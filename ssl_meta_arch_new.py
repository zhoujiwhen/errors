# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import gc
import logging
from functools import partial
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys
import math
os.environ['PYTHONPATH']="/mnt/ht2-nas2/00-model/00-limx/Codes/olmoearth_pretrain-main_10m:{os.environ.get('PYTHONPATH','')}"
sys.path.insert(0, "/mnt/ht2-nas2/00-model/00-limx/Codes/olmoearth_pretrain-main_10m")

import torch
from omegaconf import OmegaConf
from torch import Tensor, nn

import dinov3.distributed as distributed
from dinov3.checkpointer import init_fsdp_model_from_checkpoint
from dinov3.configs import get_default_config
from dinov3.data import DataAugmentationDINO
from dinov3.fsdp.ac_compile_parallelize import ac_compile_parallelize
from dinov3.layers.dino_head import DINOHead
from dinov3.loss import DINOLoss, GramLoss, KoLeoLoss, KoLeoLossDistributed, iBOTPatchLoss
from dinov3.models import build_model_from_cfg, build_model_from_cfg_QH
from dinov3.train.cosine_lr_scheduler import linear_warmup_cosine_decay
from dinov3.train.param_groups import fuse_params_groups, get_params_groups_with_decay_fsdp
from dinov3.utils import count_parameters
from dinov3.models.RS_vision_transformer import UPHead
from dinov3.models.olmoearth import load_model_direct
from dinov3.olmoearth_pretrain.model_loader import ModelID, load_model_from_id, load_model_from_path
from dinov3.olmoearth_pretrain.datatypes import TokensAndMasks

logger = logging.getLogger("dinov3")
F = torch.nn.functional


class SSLMetaArch_QH2(nn.Module):
    """
    Modified version of SSLMetaArchCompilable including gram loss:
    - Gram loss is used only if gram.use_loss is set to true
    """

    def __init__(self, cfg, olmoearth_path, modalities):
        super().__init__()

        # assert cfg.multidistillation.enabled is False
        # assert cfg.crops.local_crops_number > 0
        # assert cfg.ibot.separate_head is True
        # assert cfg.train.centering == "sinkhorn_knopp"

        # For some reason FULL_SHARD doesn't work
        # assert cfg.compute_precision.sharding_strategy == "SHARD_GRAD_OP"

        self.cfg = cfg
        self.modalities = modalities    # 表明推理的时候要调用的模态如下

        student_model_dict = dict() # student_model中需要包含四个部分--part1 DINO VIT；part2 Olmoearth；part3 Uphead；part4 Fusion
        teacher_model_dict = dict()
        # gram_model_dict = dict()

        # 定义模型：teacher(DinoVisionTransformer) student(DinoVisionTransformer dropout),embed_dim=student.embed_dim
        ######## Part1: HR encoder ######## 原始的dinov3 vit
        hr_student_backbone, hr_teacher_backbone, hr_embed_dim = build_model_from_cfg(cfg)
        torch.cuda.empty_cache()
        gc.collect()
        # gram_backbone, _ = build_model_from_cfg(cfg, only_teacher=True)
        logger.info(f"Number of parameters: {count_parameters(hr_student_backbone)}")
        student_model_dict["backbone"] = hr_student_backbone
        teacher_model_dict["backbone"] = hr_teacher_backbone
        # gram_model_dict["backbone"] = gram_backbone
        logger.info(f"OPTIONS -- architecture : embed_dim: {hr_embed_dim}")

        self.hr_embed_dim = hr_embed_dim  # D
 
        self.dino_out_dim = cfg.dino.head_n_prototypes  # K

        # 此部分暂时待定
        logger.info("OPTIONS -- DINO")
        logger.info(f"OPTIONS -- DINO -- loss_weight: {cfg.dino.loss_weight}")
        logger.info(f"OPTIONS -- DINO -- global_ignore_diagonal: {cfg.dino.global_ignore_diagonal}")
        logger.info(f"OPTIONS -- DINO -- head_n_prototypes: {cfg.dino.head_n_prototypes}")
        logger.info(f"OPTIONS -- DINO -- head_bottleneck_dim: {cfg.dino.head_bottleneck_dim}")
        logger.info(f"OPTIONS -- DINO -- head_hidden_dim: {cfg.dino.head_hidden_dim}")
        logger.info(f"OPTIONS -- DINO -- head_norm_last_layer: {cfg.dino.head_norm_last_layer}")
        dino_head_class = partial(  # 可选项，后续计算loss时可能需要
            DINOHead,
            in_dim=hr_embed_dim,
            out_dim=cfg.dino.head_n_prototypes,
            hidden_dim=cfg.dino.head_hidden_dim,
            bottleneck_dim=cfg.dino.head_bottleneck_dim,
            nlayers=cfg.dino.head_nlayers,
        )
        student_model_dict["dino_head"] = dino_head_class()
        teacher_model_dict["dino_head"] = dino_head_class()
        self.dino_loss = DINOLoss(self.dino_out_dim)

        ######## Part2: Olmoearth encoder ########
        # olmoearth = load_model_from_path(olmoearth_path).encoder

        student_model_dict['olmoearth'] = load_model_from_path(olmoearth_path).encoder
        student_model_dict['olmoearth'].eval()
        student_model_dict['olmoearth'].requires_grad_(False)
        teacher_model_dict['olmoearth'] = load_model_from_path(olmoearth_path).encoder
        teacher_model_dict['olmoearth'].eval()
        teacher_model_dict['olmoearth'].requires_grad_(False)
        # student_model_dict['olmoearth'] = load_model_direct()
        # teacher_model_dict['olmoearth'] = load_model_direct()
        # student_model_dict['olmoearth'].eval()
        # student_model_dict['olmoearth'].requires_grad_(False)

        # teacher_model_dict['olmoearth'].eval()
        # teacher_model_dict['olmoearth'].requires_grad_(False)

        ######## Part3: embedding head ########
        embed_head = partial(
            UPHead,
            in_dim=768,
            out_dim=hr_embed_dim,
            modalities=modalities
        )
        student_model_dict["embed_head"] = embed_head()
        teacher_model_dict["embed_head"] = embed_head()

        ######## Part4: Fusion encoder ########
        # 注意：如果在时间维度上进行encoder，当前调用的是torchvision中vit encoder部分，位置编码后续需参照skysense
        fusion_student_backbone, fusion_teacher_backbone, fusion_embed_dim = build_model_from_cfg_QH()
        torch.cuda.empty_cache()
        gc.collect()
        # gram_backbone, _ = build_model_from_cfg(cfg, only_teacher=True)
        logger.info(f"Number of parameters: {count_parameters(fusion_student_backbone)}")
        student_model_dict["fusion_backbone"] = fusion_student_backbone
        teacher_model_dict["fusion_backbone"] = fusion_teacher_backbone

        # Build student and teacher models
        self.student = nn.ModuleDict(student_model_dict)
        self.teacher = nn.ModuleDict(teacher_model_dict)
        self.model_ema = self.teacher  # this may be overwritten for distillation
        logger.info(f"Student and Teacher are built: they are both {cfg.student.arch} network.")

        if cfg.distillation.enabled:
            self._setup_distillation()
        # No grad is needed for these two
        self.teacher.requires_grad_(False)
        self.model_ema.requires_grad_(False)
        self.ema_params_lists = None

        # getting config params fixed:
        # self.n_local_crops = self.cfg.crops.local_crops_number
        # self.is_distillation_enabled = self.cfg.distillation.enabled
        self.dino_global_ignore_diagonal = self.cfg.dino.global_ignore_diagonal
        self.dino_loss_weight = self.cfg.dino.loss_weight
        # self.dino_koleo_loss_weight = self.cfg.dino.koleo_loss_weight
        # self.ibot_loss_weight = self.cfg.ibot.loss_weight

        # Local loss reweighting
        if self.cfg.dino.reweight_dino_local_loss:
            iter_per_epoch = cfg.train.OFFICIAL_EPOCH_LENGTH
            total_iterations = iter_per_epoch * cfg.optim.epochs
            schedule_cfg = cfg.dino.local_loss_weight_schedule
            self.dino_local_loss_schedule = linear_warmup_cosine_decay(
                start=schedule_cfg.start,
                peak=schedule_cfg.peak,
                end=schedule_cfg.end,
                warmup_iterations=iter_per_epoch * schedule_cfg.warmup_epochs,
                total_iterations=total_iterations,
                cosine_iterations=(
                    iter_per_epoch * schedule_cfg.cosine_epochs if "cosine_epochs" in schedule_cfg else None
                ),
            )


    def _setup_distillation(self):
        logger.info(f"Performing distillation from {self.cfg.distillation.full_cfg_path}")

        default_cfg = get_default_config()
        distillation_cfg = OmegaConf.load(self.cfg.distillation.full_cfg_path)
        distillation_cfg = OmegaConf.merge(default_cfg, distillation_cfg)

        assert distillation_cfg.ibot.separate_head is True
        assert distillation_cfg.ibot.head_n_prototypes == self.cfg.ibot.head_n_prototypes, (
            f"{distillation_cfg.ibot.head_n_prototypes} != {self.cfg.ibot.head_n_prototypes}"
        )
        assert distillation_cfg.dino.head_n_prototypes == self.cfg.dino.head_n_prototypes, (
            f"{distillation_cfg.dino.head_n_prototypes} != {self.cfg.dino.head_n_prototypes}"
        )
        assert distillation_cfg.student.patch_size == self.cfg.student.patch_size

        teacher_model_dict = dict()

        backbone, embed_dim = build_model_from_cfg(distillation_cfg, only_teacher=True)
        teacher_model_dict["backbone"] = backbone

        teacher_model_dict["dino_head"] = DINOHead(
            in_dim=embed_dim,
            out_dim=distillation_cfg.dino.head_n_prototypes,
            hidden_dim=distillation_cfg.dino.head_hidden_dim,
            bottleneck_dim=distillation_cfg.dino.head_bottleneck_dim,
            nlayers=distillation_cfg.dino.head_nlayers,
        )
        teacher_model_dict["ibot_head"] = DINOHead(
            in_dim=embed_dim,
            out_dim=distillation_cfg.ibot.head_n_prototypes,
            hidden_dim=distillation_cfg.ibot.head_hidden_dim,
            bottleneck_dim=distillation_cfg.ibot.head_bottleneck_dim,
            nlayers=distillation_cfg.ibot.head_nlayers,
        )
        self.teacher = nn.ModuleDict(teacher_model_dict)

    def init_weights(self) -> None:
        # All weights are set to `nan` to ensure we initialize everything explicitly
        self.student.backbone.init_weights()
        self.student.dino_head.init_weights()
        # self.student.ibot_head.init_weights()
        self.dino_loss.init_weights()
        # self.ibot_patch_loss.init_weights()
        self.model_ema.load_state_dict(self.student.state_dict())
        # if self.has_gram_teacher:
        #     if self.gram_ckpt is not None:
        #         logger.info(f"Loading pretrained weights from {self.gram_ckpt}")
        #         init_fsdp_model_from_checkpoint(
        #             self.gram_teacher,
        #             self.gram_ckpt,
        #             skip_load_keys=[
        #                 "dino_head",
        #                 "ibot_head",
        #                 "dino_loss.center",
        #                 "ibot_patch_loss.center",
        #             ],
        #             keys_not_sharded=["backbone.rope_embed.periods", "qkv.bias_mask"],
        #             process_group=distributed.get_default_process_group(),
        #         )
        #         self.gram_teacher_initialized = True
        #     else:
        #         raise ValueError(f"Provide a correct path to {self.gram_ckpt}")
        #     self.gram_teacher.requires_grad_(False)
        #     self.gram_teacher.eval()
        if self.cfg.student.resume_from_teacher_chkpt:
            logger.info(f"Loading pretrained weights from {self.cfg.student.resume_from_teacher_chkpt}")
            init_fsdp_model_from_checkpoint(
                self.student,
                self.cfg.student.resume_from_teacher_chkpt,
                skip_load_keys=["dino_loss.center", "ibot_patch_loss.center"],
                keys_not_sharded=["backbone.rope_embed.periods", "qkv.bias_mask"],
                process_group=distributed.get_process_subgroup(),
            )
            self.model_ema.load_state_dict(self.student.state_dict())
        if self.cfg.distillation.enabled:
            if self.cfg.distillation.checkpoint_path != "ignore":
                logger.info(f"Loading teacher to distil from : {self.cfg.distillation.checkpoint_path}")
                init_fsdp_model_from_checkpoint(
                    self.teacher,
                    self.cfg.distillation.checkpoint_path,
                    skip_load_keys=["dino_loss.center", "ibot_patch_loss.center"],
                    keys_not_sharded=["backbone.rope_embed.periods", "qkv.bias_mask"],
                    process_group=distributed.get_default_process_group(),
                )
            else:
                logger.info("Init teacher to distil from, used for testing purpose only")
                self.teacher.backbone.init_weights()
                self.teacher.dino_head.init_weights()
                # self.teacher.ibot_head.init_weights()
            logger.info(f"Performing distillation from: {self.teacher}")

    def forward_backward(
        self, data, teacher_temp, iteration=0, **ignored_kwargs
    ) -> tuple[Tensor, dict[str, float | Tensor]]:
        del ignored_kwargs
        # data里面包含三个部分：1. 高分辨率图像（teacher，student）；2. olmoearth的h5数据
        # data["hr_image_student"]
        # data["hr_image_teacher"]
        # data["h5_data"]
        metrics_dict = {}

        # Shapes
        # n_global_crops = 2
        # n_local_crops = self.n_local_crops  # self.cfg.crops.local_crops_number
        # B = data["collated_local_crops"].shape[0] // n_local_crops
        # assert data["collated_global_crops"].shape[0] == n_global_crops * B
        # metrics_dict["local_batch_size"] = B
        # metrics_dict["global_batch_size"] = data["global_batch_size"]

        # global_crops = data["collated_global_crops"].cuda(non_blocking=True)
        # local_crops = data["collated_local_crops"].cuda(non_blocking=True)
        # masks = data["collated_masks"].cuda(non_blocking=True)
        # mask_indices_list = data["mask_indices_list"].cuda(non_blocking=True)
        # masks_weight = data["masks_weight"].cuda(non_blocking=True)
        # n_masked_patches_tensor = data["n_masked_patches"].cuda(non_blocking=True)

        hr_image_student = data["hr_image_student"].cuda(non_blocking=True) # [B, C, H, W]
        hr_image_teacher = data["hr_image_teacher"].cuda(non_blocking=True) # [B, C, H, W]
        # 需要对其中分别进行cuda
        h5_data = data["h5_data"]
        # Teacher output (will trigger an all-gather to unshard)
        teacher_output, teacher_fusion_output = self.get_teacher_output(
            hr_image_teacher, h5_data,
            teacher_temp = teacher_temp # 待定 主要在DINO head后面使用
        )

        # teacher_global = self.get_teacher_output(
        #     global_crops.unflatten(0, (n_global_crops, B)),
        #     teacher_temp=teacher_temp,
        #     n_masked_patches_tensor=n_masked_patches_tensor,
        #     mask_indices_list=mask_indices_list,
        #     upperbound=data["upperbound"],
        # )

        # Student output (will trigger an all-gather to unshard)
        student_output, student_fusion_output = self.get_student_output(
            hr_image_student, h5_data
        )
        # student_global, student_local = self.get_student_output(
        #     global_crops=global_crops.unflatten(0, (n_global_crops, B)),
        #     local_crops=local_crops.unflatten(0, (n_local_crops, B)),
        #     upperbound=data["upperbound"],
        #     masks=masks,
        #     mask_indices_list=mask_indices_list,
        # )


        # Compute losses and backprop
        loss_accumulator, loss_dict = self.compute_losses(
            teacher_global=teacher_global,
            student_global=student_global,
            student_local=student_local,
            gram_global=gram_global,
            masks=masks,
            mask_indices_list=mask_indices_list,
            masks_weight=masks_weight,
            iteration=iteration,
        )

        self.backprop_loss(loss_accumulator)

        # Return total weighted loss and a dict of metrics to log
        return loss_accumulator, metrics_dict | loss_dict

    def origanize_embed(self, embed, shape):    # 这个shape可以是输入图像的大小
        B, N, D = shape
        H = W = math.isqrt(N)
        modalities = self.modalities
        # fusion_feature_mask = []
        re_embed = dict()
        # 经过维度对齐的时候是融合之后对齐，后面需要拆分对应回原来的维度
        for modality in modalities:
            x_modality = getattr(embed, modality)

            masked_modality_name = embed.get_masked_modality_name(modality)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
            x_modality_mask = getattr(embed, masked_modality_name)
            # 首先将bandsets进行压缩，首先判断bandsets中是否存在数据缺失
            # 根据x_modality_mask找出有效的bandsets
            if x_modality is None:
                # mask = torch.zeros((B, 12)).to(x_modality.device)  # [B, T]
                # x = torch.ones((B, H, W, 12)).to(x_modality.device)
                # fusion_feature.append(x)
                # fusion_feature_mask.append(mask)
                re_embed[modality] = x_modality
                re_embed[masked_modality_name] = x_modality_mask
                re_embed[modality+'_timestamp'] = None
            else:
                band_missing = 1.0 - x_modality_mask.sum(dim=(1,2,3))>0     # False表示没有缺失，True表示有缺失
                x_modality_mean = (x_modality * band_missing.unsqueeze(-1).unsqueeze(1).unsqueeze(1).unsqueeze(1)).mean(dim=-2)
                # 需要判断x_modality_mean的shape是否符合要求，如果不符合要求的话需要通过UpNet
                time_missing = x_modality_mask.sum(dim=(1,2,4))>0     # True表示没有缺失，False表示有缺失
                if x_modality_mean.shape[0] == B and x_modality_mean.shape[1] == H:
                    pass
                    
                else:   # [B, H, W, T, D]
                    x_modality_mean = x_modality_mean.permute(0,3,4,1,2)
                    B, T, C, H1, W1 = x_modality_mean.shape
                    x_modality_mean = x_modality_mean.view(B*T, C, H1, W1)
                    x_modality_mean = F.interpolate(x_modality_mean, size=(H, W), mode='bilinear', align_corners=True)
                    x_modality_mean = x_modality_mean.view(B, T, C, H, W).permute(0,3,4,1,2)
                re_embed[modality] = x_modality_mean
                re_embed[masked_modality_name] = x_modality_mask
                re_embed[modality+'_timestamp'] = time_missing
                # time_missing = 1.0 - x_modality_mask.sum(dim=(1,2,4))>0     # False表示没有缺失，True表示有缺失
                # fusion_feature_mask.append(time_missing)
    
        return TokensAndMasks(**re_embed)


    @torch.no_grad()
    def get_teacher_output(
        self,
        images,
        h5_data,
        teacher_temp,
    ):
        B, rgb, H, W = images.shape
        # Part1 Output
        backbone_out = self.teacher.backbone(images, is_training=True)
        cls = backbone_out["x_norm_clstoken"]  # [n_crops * B, D]
        reg = backbone_out["x_storage_tokens"]  # [n_crops * B, R, D]
        patch = backbone_out["x_norm_patchtokens"]  # [n_crops * B, N, D]

        cls_after_head = self.teacher.dino_head(cls)  # [n_crops * B, K]

        # Center with sinkhorn-knopp
        cls_centered = self.dino_loss.sinkhorn_knopp_teacher(
            cls_after_head, teacher_temp=teacher_temp
        )  # [n_crops * B, K]

        # Part2 Olmoearth encoder
        earth_embed = self.teacher.olmoearth(h5_data, fast_pass=True, patch_size=4)['tokens_and_masks']
        # earth_embed = self.teacher.olmoearth(h5_data, fast_pass=True, patch_size=4)

        # 这里是否需要根据mask是否为0，如果mask>0表示该时刻缺失，如果mask=0表示无缺失

        # 需要对earth_embed特征进行整合
        earth_embed = self.origanize_embed(earth_embed, patch.shape)
        # Part3 Embed head
        align_feature = self.teacher.embed_head(earth_embed)

        # Part4 Fusion
        fusion_feature = self.teacher.fusion_backbone(patch, align_feature)
        # masked_patch_centered = self.ibot_patch_loss.sinkhorn_knopp_teacher(
        #     masked_patch_after_head,
        #     teacher_temp=teacher_temp,
        #     n_masked_patches_tensor=n_masked_patches_tensor,
        # )  # [n_masked_patches, K]

        return {
            'patch_pre_feature': patch,
            "cls_pre_head": cls.unflatten,  # [n_crops, B, D]
            "reg_pre_head": reg.unflatten,  # [n_crops, B, R, D]
            "cls_after_head": cls_after_head,  # [n_crops, B, K]
            "cls_centered": cls_centered,  # [n_crops, B, K]
            "fusion_feature": fusion_feature
            # "masked_patch_centered": masked_patch_centered,  # [n_masked_patches, K]
        }

    def get_gram_teacher_output(self, images, *, masks, teacher_global, student_global, student_global_crops_size):
        # Get student patch features
        student_patches = student_global["patch_pre_head"].flatten(0, 1)  # [n_crops * B, P, D]

        # Get gram targets
        if self.gram_ema_teacher:
            teacher_patches = teacher_global["patch_pre_head"].flatten(0, 1)  # [n_crops * B, P, D]
        else:
            if not self.gram_teacher_initialized:
                raise ValueError("Gram teacher has not been initialized. Load a checkpoint or from the EMA teacher.")
            n_crops, B, rgb, H, W = images.shape
            images = images.flatten(0, 1)  # [n_crops * B, rgb, H, W]

            with torch.no_grad():
                backbone_out = self.gram_teacher.backbone(images, is_training=True)
            teacher_patches = backbone_out["x_norm_patchtokens"]  # [n_crops * B, P_T, D]

            # Downsample Gram teacher features if needed
            if teacher_patches.shape[1] != student_patches.shape[1]:
                N = H // self.cfg.student.patch_size
                assert teacher_patches.shape[1] == N**2
                N_student = student_global_crops_size // self.cfg.student.patch_size
                assert student_patches.shape[1] == N_student**2
                patches_hw = teacher_patches.transpose(-2, -1).unflatten(-1, (N, N))  # [n_crops * B, D, N, N]
                patches_hw = torch.nn.functional.interpolate(
                    patches_hw,
                    size=(N_student, N_student),
                    mode=self.gram_global_teacher_resize_method,
                    align_corners=False,
                    antialias=self.gram_global_teacher_resize_antialias,
                )
                teacher_patches = patches_hw.flatten(-2, -1).transpose(
                    -2, -1
                )  # [n_crops * B, N_student * N_student, D]
                assert teacher_patches.shape == student_patches.shape

        # Select the patches to be considered in the loss
        orig_student_patches = student_patches
        orig_teacher_patches = teacher_patches
        if self.gram_tokens_used == "masked":
            student_patches = student_patches[masks]
            teacher_patches = teacher_patches[masks]
        elif self.gram_tokens_used == "unmasked":
            student_patches = student_patches[~masks]
            teacher_patches = teacher_patches[~masks]

        return {
            "student_patches": student_patches,  # [n_crops * B, P, D] or [n_selected_patches, D]
            "teacher_patches": teacher_patches,  # [n_crops * B, P, D] or [n_selected_patches, D]
            # Unmasked patches, for computing statistics
            "orig_student_patches": orig_student_patches,  # [n_crops * B, P, D]
            "orig_teacher_patches": orig_teacher_patches,  # [n_crops * B, P, D]
        }

    def get_student_output(self, images, h5_data):
        B, rgb, H, W = images.shape
        # Part1 Output
        backbone_out = self.student.backbone(images, is_training=True)
        cls = backbone_out["x_norm_clstoken"]  # [n_crops * B, D]
        reg = backbone_out["x_storage_tokens"]  # [n_crops * B, R, D]
        patch = backbone_out["x_norm_patchtokens"]  # [n_crops * B, N, D]

        cls_after_head = self.teacher.dino_head(cls)  # [n_crops * B, K]

        # # Center with sinkhorn-knopp
        # cls_centered = self.dino_loss.sinkhorn_knopp_teacher(
        #     cls_after_head, teacher_temp=teacher_temp
        # )  # [n_crops * B, K]

        # Part2 Olmoearth encoder
        earth_embed = self.teacher.olmoearth.encoder(h5_data, fast_pass=True, patch_size=4)
        # earth_embed = self.teacher.olmoearth(h5_data, fast_pass=True, patch_size=4)

        # Part3 Embed head
        align_feature = self.teacher.embed_head(earth_embed)

        # Part4 Fusion
        fusion_feature = self.teacher.fusion_backbone(patch, align_feature)
        # masked_patch_centered = self.ibot_patch_loss.sinkhorn_knopp_teacher(
        #     masked_patch_after_head,
        #     teacher_temp=teacher_temp,
        #     n_masked_patches_tensor=n_masked_patches_tensor,
        # )  # [n_masked_patches, K]

        return {
            'patch_pre_feature': patch,
            "cls_pre_head": cls.unflatten,  # [n_crops, B, D]
            "reg_pre_head": reg.unflatten,  # [n_crops, B, R, D]
            "cls_after_head": cls_after_head,  # [n_crops, B, K]
            # "cls_centered": cls_centered,  # [n_crops, B, K]
            "fusion_feature": fusion_feature
            # "masked_patch_centered": masked_patch_centered,  # [n_masked_patches, K]
        }

        # B, rgb, H, W = images.shape

        # # Forward global and local crops through the student backbone jointly
        # # Part1 Output
        # backbone_out = self.student.backbone(images, is_training=True)
        
        # cls, reg, patch = (
        #     backbone_out["x_norm_clstoken"],
        #     backbone_out["x_storage_tokens"],
        #     backbone_out["x_norm_patchtokens"],
        # )

        # # # IBOT head only on masked patches
        # # masked_patches_pre_head = torch.index_select(g_patch.flatten(0, 1), dim=0, index=mask_indices_list)
        # # global_masked_patch_after_head = self.student.ibot_head(masked_patches_pre_head)

        # # DINO head on CLS tokens (all in one pass)
        # # buffer = [
        # #     g_cls,  # [n_global_crops * B, D]
        # #     l_cls,  # [n_local_crops * B, D]
        # # ]
        # # sizes = [x.shape[0] for x in buffer]
        # # buffer = torch.cat(buffer, dim=0)  # [n_global_crops * B + n_local_crops * B, D]
        # cls_after_head = self.student.dino_head(cls)  # [n_global_crops * B + n_local_crops * B, K]
        # # buffer = torch.split_with_sizes(buffer, sizes, dim=0)

        # # Part2 Olmoearth encoder
        # earth_embed = self.student.olmoearth(h5_data)

        # # Part3 Embed head
        # align_feature = self.student.embed_head(earth_embed)

        # # Part4 Fusion
        # fusion_feature = self.student.fusion_backbone(patch, align_feature)

        # output = {
        #     "cls_pre_head": g_cls.unflatten(0, [n_global_crops, B]),  # [n_global_crops, B, D]
        #     "reg_pre_head": g_reg.unflatten(0, [n_global_crops, B]),  # [n_global_crops, B, R, D]
        #     "patch_pre_head": g_patch.unflatten(0, [n_global_crops, B]),  # [n_global_crops, B, P, D]
        #     "cls_after_head": cls_after_head.unflatten(0, [n_global_crops, B]),  # [n_global_crops, B, K],
        #     "fusion_fearture": fusion_feature,
        # }

        # return output

    def compute_losses(
        self,
        *,
        teacher_global,
        student_global,
        student_local,
        gram_global,
        masks,
        mask_indices_list,
        masks_weight,
        iteration,
    ):
        n_global_crops = student_global["cls_after_head"].shape[0]
        n_local_crops = student_local["cls_after_head"].shape[0]
        loss_dict = {}
        loss_accumulator = 0.0

        # Loss scales like in DINOv2, these are multiplied with the loss weights from the config
        dino_global_terms = (
            n_global_crops * (n_global_crops - 1) if self.dino_global_ignore_diagonal else n_global_crops**2
        )
        dino_local_terms = n_global_crops * n_local_crops
        dino_global_scale = dino_global_terms / (dino_global_terms + dino_local_terms)
        dino_local_scale = dino_local_terms / (dino_global_terms + dino_local_terms)
        koleo_scale = n_global_crops

        # DINO local loss: compare post-head CLS tokens: student(local crops) vs. teacher(global crops)
        dino_local_crops_loss = self.dino_loss(
            student_logits=student_local["cls_after_head"],
            teacher_probs=teacher_global["cls_centered"],
        )
        loss_dict["dino_local_crops_loss"] = dino_local_crops_loss

        # Reweighting of DINO loss
        if self.cfg.dino.reweight_dino_local_loss:
            local_weight = self.dino_local_loss_schedule[iteration]
        else:
            local_weight = 1.0

        loss_dict["dino_local_loss_weight"] = local_weight
        loss_accumulator += self.dino_loss_weight * dino_local_scale * local_weight * dino_local_crops_loss

        # DINO global loss: compare post-head CLS tokens: student(global crops) vs. teacher(global crops)
        dino_global_crops_loss = self.dino_loss(
            student_logits=student_global["cls_after_head"],
            teacher_probs=teacher_global["cls_centered"],
            ignore_diagonal=self.dino_global_ignore_diagonal,
        )
        loss_dict["dino_global_crops_loss"] = dino_global_crops_loss
        loss_accumulator += self.dino_loss_weight * dino_global_scale * dino_global_crops_loss

        # Koleo: regularize pre-head CLS tokens of student(global crops)
        koleo_loss = sum(self.koleo_loss(x) for x in student_global["cls_pre_head"]) / n_global_crops
        loss_dict["koleo_loss"] = koleo_loss
        loss_accumulator += self.dino_koleo_loss_weight * koleo_scale * koleo_loss

        # IBOT loss
        ibot_patch_loss = self.ibot_patch_loss.forward_masked(
            student_global["masked_patch_after_head"],
            teacher_global["masked_patch_centered"],
            student_masks_flat=masks,
            n_masked_patches=mask_indices_list.shape[0],
            masks_weight=masks_weight,
        )
        loss_dict["ibot_loss"] = ibot_patch_loss
        loss_accumulator += self.ibot_loss_weight * ibot_patch_loss

        # Gram loss
        if self.gram_use_loss:
            gram_loss = self.gram_loss(
                gram_global["student_patches"],
                gram_global["teacher_patches"],
                img_level=self.gram_img_level,
            )

            if self.gram_loss_schedule is not None:
                gram_loss_weight = self.gram_loss_schedule[iteration]
            else:
                gram_loss_weight = self.gram_loss_weight

            loss_dict["gram_loss_weight"] = gram_loss_weight
            loss_accumulator += gram_loss * gram_loss_weight
            loss_dict["gram_loss"] = gram_loss

            if self.gram_compute_stats:
                with torch.no_grad():
                    # Save stats over masked / unmasked tokens
                    gram_loss_masked = self.gram_loss(
                        gram_global["orig_student_patches"][masks].detach(),
                        gram_global["orig_teacher_patches"][masks],
                        img_level=False,
                    )
                    loss_dict["stats_only/masked_gram_loss"] = gram_loss_masked
                    gram_loss_unmasked = self.gram_loss(
                        gram_global["orig_student_patches"][~masks].detach(),
                        gram_global["orig_teacher_patches"][~masks],
                        img_level=False,
                    )
                    loss_dict["stats_only/unmasked_gram_loss"] = gram_loss_unmasked

        return loss_accumulator, loss_dict

    @torch.no_grad()
    def gram_load_ema_teacher(self):
        if self.has_gram_teacher:
            skip_load_prefixes = ["dino_head.", "ibot_head."]
            self.gram_teacher.load_state_dict(
                {
                    k: v
                    for k, v in self.model_ema.state_dict().items()
                    if not any(k.startswith(prefix) for prefix in skip_load_prefixes)
                }
            )
            self.gram_teacher.requires_grad_(False)
            self.gram_teacher.eval()
            self.gram_teacher_initialized = True

    def train(self):
        super().train()
        self.teacher.eval()

    def forward(self, inputs):
        raise NotImplementedError

    def backprop_loss(self, loss):
        loss.backward()

    def update_ema(self, m):
        if self.ema_params_lists is None:
            student_param_list = []
            teacher_param_list = []
            for k in self.student.keys():
                for ms, mt in zip(self.student[k].parameters(), self.model_ema[k].parameters()):
                    student_param_list += [ms]
                    teacher_param_list += [mt]
            self.ema_params_lists = (student_param_list, teacher_param_list)
        else:
            student_param_list, teacher_param_list = self.ema_params_lists
        with torch.no_grad():
            torch._foreach_mul_(teacher_param_list, m)
            torch._foreach_add_(teacher_param_list, student_param_list, alpha=1 - m)

    def update_gram(self, m=0):
        if not self.has_gram_teacher:
            return
        logger.info("Updating gram teacher with teacher weights.")
        if self.gram_params_lists is None:
            teacher_param_list = []
            gramteacher_param_list = []
            for k in self.gram_teacher.keys():
                for mgt, mt in zip(self.gram_teacher[k].parameters(), self.teacher[k].parameters()):
                    gramteacher_param_list += [mgt]
                    teacher_param_list += [mt]
            self.gram_params_lists = (gramteacher_param_list, teacher_param_list)
        else:
            gramteacher_param_list, teacher_param_list = self.gram_params_lists

        with torch.no_grad():
            torch._foreach_mul_(gramteacher_param_list, m)
            torch._foreach_add_(gramteacher_param_list, teacher_param_list, alpha=1 - m)

    def build_data_augmentation_dino(self, cfg):
        return DataAugmentationDINO(
            cfg.crops.global_crops_scale,
            cfg.crops.local_crops_scale,
            cfg.crops.local_crops_number,
            global_crops_size=cfg.crops.global_crops_size,
            local_crops_size=cfg.crops.local_crops_size,
            gram_teacher_crops_size=cfg.crops.gram_teacher_crops_size,
            gram_teacher_no_distortions=cfg.crops.gram_teacher_no_distortions,
            local_crops_subset_of_global_crops=cfg.crops.localcrops_subset_of_globalcrops,
            share_color_jitter=cfg.crops.share_color_jitter,
            horizontal_flips=cfg.crops.horizontal_flips,
            mean=cfg.crops.rgb_mean,
            std=cfg.crops.rgb_std,
        )

    def get_maybe_fused_params_for_submodel(self, m: nn.Module):
        params_groups = get_params_groups_with_decay_fsdp(
            model=m,
            lr_decay_rate=self.cfg.optim.layerwise_decay,
            patch_embed_lr_mult=self.cfg.optim.patch_embed_lr_mult,
            dino_head_wd_multiplier=self.cfg.optim.dino_head_wd_multiplier,
        )
        if self.cfg.optim.multi_tensor_optim:
            fused_params_groups = fuse_params_groups(params_groups)
            logger.info("fusing param groups")

            for g in fused_params_groups:
                g["foreach"] = True
                g["fused"] = True
            return fused_params_groups
        else:
            return params_groups

    def get_params_groups(self):
        all_params_groups = []
        for name, m in self.student.items():
            logger.info(f"Getting paramer groups for {name}")
            all_params_groups += self.get_maybe_fused_params_for_submodel(m)
        return all_params_groups

    def prepare_for_distributed_training(self) -> None:
        process_subgroup = distributed.get_process_subgroup()
        default_process_group = distributed.get_default_process_group()
        inference_only_models = [self.model_ema]
        inference_only_models_process_groups = [process_subgroup]
        if self.cfg.distillation.enabled:
            inference_only_models.append(self.teacher)
            inference_only_models_process_groups.append(default_process_group)
        ac_compile_parallelize(
            trained_model=self.student,
            inference_only_models=inference_only_models,
            cfg=self.cfg,
            trained_model_process_group=process_subgroup,
            inference_only_models_process_groups=inference_only_models_process_groups,
        )

    def broadcast_to_subgroups(self, tensor, over_dim, global_batch_size=None):
        """
        This is an operation that takes a tensor from the default process group, gathers it, stacks it, then scatters it within a smaller process subgroup
        """
        world_size = distributed.get_world_size()
        subgroup_size = distributed.get_subgroup_size()
        gathered = [torch.zeros_like(tensor) for _ in range(world_size)]

        torch.distributed.all_gather(gathered, tensor)
        catted = torch.cat(gathered, dim=over_dim)
        if global_batch_size is not None:
            catted = catted.narrow(dim=over_dim, start=0, length=global_batch_size)

        return catted.chunk(subgroup_size, dim=over_dim)[distributed.get_subgroup_rank()].clone()
