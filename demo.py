import argparse
import os
import sys
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = "5"
os.environ["TORCH_COMPILE"] = "0"
os.environ['PYTHONPATH']="/mnt/ht2-nas2/00-model/00-limx/Codes/olmoearth_pretrain-main_10m:{os.environ.get('PYTHONPATH','')}"
sys.path.insert(0, "/mnt/ht2-nas2/00-model/00-limx/Codes/olmoearth_pretrain-main_10m")
 
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, ROOT_DIR)
import h5py
import hdf5plugin  # noqa: F401
from dinov3.configs import setup_config, setup_job, setup_multidistillation
from dinov3.train.ssl_meta_arch_new import SSLMetaArch_QH2
from dinov3.olmoearth_pretrain.data.constants import MAX_SEQUENCE_LENGTH, MISSING_VALUE, Modality
from dinov3.olmoearth_pretrain.data.normalize import Normalizer, Strategy
from dinov3.olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, MaskValue, OlmoEarthSample
import torch
torch.backends.cuda.matmul.allow_tf32 = True  # pytorch 1.12 sets this to false by default
torch.backends.cudnn.benchmark = False  # True

def get_args_parser(add_help: bool = True):
    parser = argparse.ArgumentParser("DINOv3 training", add_help=add_help)
    parser.add_argument("--config-file", default="/mnt/ht2-nas2/00-model/00-limx/Codes/dinov3-main/dinov3/configs/train/vitl_im1k_lin834.yaml", metavar="FILE", help="path to config file")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Whether to not attempt to resume from the checkpoint directory. ",
    )
    parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
    parser.add_argument("--eval", type=str, default="", help="Eval type to perform")
    parser.add_argument(
        "--eval_pretrained_weights",
        type=str,
        default="",
        help="Path to pretrained weights",
    )
    parser.add_argument(
        "opts",
        help="""
Modify config options at the end of the command. For Yacs configs, use
space-separated "PATH.KEY VALUE" pairs.
For python-based LazyConfig, use "path.key=value".
        """.strip(),
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--output-dir",
        default="./local_dino",
        type=str,
        help="Path to save logs and checkpoints.",
    )
    parser.add_argument("--seed", default=0, type=int, help="RNG seed")
    parser.add_argument(
        "--benchmark-codebase",
        action="store_true",
        help="test the codebase for a few iters",
    )
    parser.add_argument("--test-ibot", action="store_true", help="test ibot")
    parser.add_argument("--profiling", action="store_true", help="do profiling")
    parser.add_argument("--dump-fsdp-weights", action="store_true", help="dump fsdp weights")
    parser.add_argument("--record_ref_losses", action="store_true", help="record reference losses")
    parser.add_argument("--ref_losses_path", default="", type=str)
    parser.add_argument("--multi-distillation", action="store_true", help="run multi-distillation")

    
    return parser

def read_h5_file(h5_path: str, modalities: list[str] | None = None) -> dict:
    """读取单个 H5 文件，返回样本数据字典和缺失时间步掩码。

    Args:
        h5_path: H5 文件路径。
        modalities: 需要读取的模态列表。为 None 时读取所有模态。

    Returns:
        元组 (sample_dict, missing_timesteps_masks)。
    """
    sample_dict = {}
    missing_timesteps_masks = {}

    with h5py.File(h5_path, "r") as h5file:
        keys = list(h5file.keys())
        # logger.info(f"H5 文件包含的键: {keys}")

        for k in keys:
            if k == "missing_timesteps_masks":
                for mk, mv in h5file[k].items():
                    if modalities is None or mk in modalities:
                        missing_timesteps_masks[mk] = mv[()]
                continue

            if modalities is None or k in modalities or k == "timestamps":
                sample_dict[k] = h5file[k][()]

    return sample_dict, missing_timesteps_masks

def pad_timestamps(sample_dict: dict, max_sequence_length: int) -> tuple[dict, int]:
    """将时间戳填充到 max_sequence_length。

    Returns:
        元组 (更新后的 sample_dict, 填充前的原始序列长度)。
    """
    timestamps = sample_dict["timestamps"]
    current_length = timestamps.shape[0]
    if current_length < max_sequence_length:
        pad_width = ((0, max_sequence_length - current_length), (0, 0))
        sample_dict["timestamps"] = np.pad(timestamps, pad_width=pad_width, mode="edge")
    return sample_dict, current_length


def fill_missing_timesteps(modality_data: np.ndarray, mask: np.ndarray, max_sequence_length: int, dtype: np.dtype) -> np.ndarray:
    """填充缺失的时间步。

    Args:
        modality_data: 模态数据，形状 [H, W, T, C]。
        mask: 布尔掩码，True 表示有效时间步。
        max_sequence_length: 最大序列长度。
        dtype: 数据类型。

    Returns:
        填充后的模态数据，形状 [H, W, max_sequence_length, C]。
    """
    modality_data = modality_data.astype(dtype)
    h, w, t, c = modality_data.shape
    full_data = np.full((h, w, max_sequence_length, c), MISSING_VALUE, dtype=dtype)
    present_indices = np.where(mask)[0]
    num_to_copy = min(len(present_indices), t)
    if num_to_copy > 0:
        full_data[:, :, present_indices[:num_to_copy], :] = modality_data[:, :, :num_to_copy, :]
    return full_data


def fill_sample_with_missing_values(
    sample_dict: dict,
    inference_modalities: list[str],
    missing_timesteps_masks: dict,
    max_sequence_length: int,
    dtype: np.dtype,
) -> tuple[dict, list[str]]:
    """填充样本中缺失的模态和时间步。

    Returns:
        元组 (填充后的 sample_dict, 缺失模态名称列表)。
    """
    missing_modalities = []

    # 获取 height, width, time
    time = sample_dict["timestamps"].shape[0]
    height, width = None, None
    for mod_name, mod_data in sample_dict.items():
        if mod_name == "timestamps":
            continue
        mod_spec = Modality.get(mod_name)
        if mod_spec.is_spatial and mod_data is not None:
            height = mod_data.shape[0] // mod_spec.image_tile_size_factor
            width = mod_data.shape[1] // mod_spec.image_tile_size_factor
            break

    for modality in inference_modalities:
        if modality not in sample_dict:
            # 模态完全缺失
            mod_spec = Modality.get(modality)
            expected_shape = OlmoEarthSample.compute_expected_shape(
                modality, height, width, time
            )
            sample_dict[modality] = np.full(expected_shape, MISSING_VALUE, dtype=dtype)
            missing_modalities.append(modality)
            continue

        # 处理缺失时间步
        if modality in missing_timesteps_masks:
            mask = missing_timesteps_masks[modality]
            modality_data = sample_dict[modality].astype(dtype)
            has_missing = not np.all(mask) or len(mask) < max_sequence_length
            if has_missing:
                sample_dict[modality] = fill_missing_timesteps(
                    modality_data, mask, max_sequence_length, dtype
                )

    return sample_dict, missing_modalities


def normalize_sample(
    sample_dict: dict,
    missing_modalities: list[str],
) -> dict:
    """对样本数据进行归一化。

    优先使用 COMPUTED 策略（mean-std），失败时回退到 PREDEFINED 策略（min-max）。
    """
    normalizer_computed = Normalizer(Strategy.COMPUTED)
    normalizer_predefined = Normalizer(Strategy.PREDEFINED)

    for modality_name in sample_dict:
        if modality_name == "timestamps":
            continue
        if modality_name in missing_modalities:
            continue

        modality_data = sample_dict[modality_name]
        missing_mask = modality_data == MISSING_VALUE

        mod_spec = Modality.get(modality_name)
        try:
            normalized = normalizer_computed.normalize(mod_spec, modality_data)
        except Exception as e:
            # logger.warning(f"模态 {modality_name} COMPUTED 归一化失败 ({e})，回退到 PREDEFINED 策略")
            normalized = normalizer_predefined.normalize(mod_spec, modality_data)

        sample_dict[modality_name] = np.where(missing_mask, modality_data, normalized).astype(np.float32)

    return sample_dict

def crop_sample(
    sample: OlmoEarthSample,
    crop_h: tuple[int, int] | None = None,
    crop_w: tuple[int, int] | None = None,
    max_t: int | None = None,
) -> OlmoEarthSample:
    """对样本进行空间裁剪和时间步截取。

    Args:
        sample: 原始样本。
        crop_h: (start_h, end_h) 空间高度裁剪范围。
        crop_w: (start_w, end_w) 空间宽度裁剪范围。
        max_t: 最大时间步数。
    """
    sample_dict = sample.as_dict(include_nones=True)
    new_dict = {}

    for attr, modality in sample_dict.items():
        if modality is None:
            new_dict[attr] = None
            continue

        if attr == "timestamps":
            if max_t is not None:
                new_dict[attr] = modality[:max_t]
            else:
                new_dict[attr] = modality
            continue

        if attr == "latlon":
            new_dict[attr] = modality
            continue

        mod_spec = Modality.get(attr)
        factor = mod_spec.image_tile_size_factor

        # 构建切片
        slices = []
        if mod_spec.is_spatial:
            h_max = modality.shape[0]
            w_max = modality.shape[1]
            if crop_h:
                h_start, h_end = crop_h[0] * factor, crop_h[1] * factor
                if h_start < 0 or h_end > h_max or h_start >= h_end:
                    raise ValueError(
                        f"裁剪范围 crop_h={crop_h} (实际 {h_start}:{h_end}) 超出模态 {attr} "
                        f"高度范围 [0, {h_max})"
                    )
            if crop_w:
                w_start, w_end = crop_w[0] * factor, crop_w[1] * factor
                if w_start < 0 or w_end > w_max or w_start >= w_end:
                    raise ValueError(
                        f"裁剪范围 crop_w={crop_w} (实际 {w_start}:{w_end}) 超出模态 {attr} "
                        f"宽度范围 [0, {w_max})"
                    )
            h_slice = slice(
                (crop_h[0] * factor) if crop_h else None,
                (crop_h[1] * factor) if crop_h else None,
            )
            w_slice = slice(
                (crop_w[0] * factor) if crop_w else None,
                (crop_w[1] * factor) if crop_w else None,
            )
            slices.extend([h_slice, w_slice])

        if mod_spec.is_multitemporal and max_t is not None:
            slices.append(slice(0, max_t))

        if slices:
            new_dict[attr] = modality[tuple(slices)]
        else:
            new_dict[attr] = modality

    return OlmoEarthSample(**{k: v for k, v in new_dict.items() if v is not None})



args = get_args_parser().parse_args()

setup_job(output_dir=args.output_dir, seed=args.seed)

cfg = setup_config(args, strict_cfg=False)

h5_path = "/mnt/ht2-nas2/QH_Group/H5_DIR/h5py_data_w_missing_timesteps_zstd_3_128_x_4/cdl_landsat_openstreetmap_raster_sentinel1_sentinel2_l2a_srtm_worldcereal_worldcover_wri_canopy_height_map/3996/sample_197.h5"
modalities = ["sentinel2_l2a", "sentinel1", "landsat"]
sample_dict, missing_timesteps_masks = read_h5_file(h5_path, modalities)

# 填充时间戳
sample_dict, _ = pad_timestamps(sample_dict, MAX_SEQUENCE_LENGTH)

# 填充缺失模态和时间步
sample_dict, missing_modalities = fill_sample_with_missing_values(
    sample_dict, modalities, missing_timesteps_masks, MAX_SEQUENCE_LENGTH, np.float32
)

# 归一化
sample_dict = normalize_sample(sample_dict, missing_modalities)

# 构造OlmoEarthSample
sample = OlmoEarthSample(**sample_dict)

# 转换为MaskedOlmoEarthSample,添加batch维度
batch_dict = {}
for attr in sample.as_dict():
    val = getattr(sample, attr)
    if val is not None:
        dtype = torch.long if attr == "timestamps" else torch.float32
        batch_dict[attr] = torch.tensor(val[np.newaxis], dtype=dtype).cuda(non_blocking=True)

batch_sample = OlmoEarthSample(**batch_dict)
masked_sample = MaskedOlmoEarthSample.from_olmoearthsample(batch_sample)    # 数据这里要考虑缺失，尤其是对于模态缺失的数据

# olmoearth parameters
olmoearth_path = "/mnt/ht2-nas2/00-model/00-common/olmoearth-base"
model = SSLMetaArch_QH2(cfg, olmoearth_path, modalities)
model.prepare_for_distributed_training()    # 需要注意这里的参数有没有被覆盖，如果to_empty()的话就是设置为empty

model.train()
# model.init_weights()    # 注意olmoearth参数不要覆盖

# input
data = dict()
data["hr_image_student"] = torch.randn((1, 3, 224, 224))
data["hr_image_teacher"] = torch.randn((1, 3, 224, 224))




data["h5_data"] = masked_sample
model.forward_backward(data, teacher_temp=0.04)
