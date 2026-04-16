# Copyright (c) 2025 Haian Jin. Created for the LVSM project (ICLR 2025).
import copy
import os
import numpy as np
import torch
import torch.nn as nn
from easydict import EasyDict as edict
from einops.layers.torch import Rearrange
from einops import rearrange, repeat
import traceback
from utils import camera_utils, data_utils
from .transformer import (
    QK_Norm_TransformerBlock,
    init_weights,
    _init_weights,
    _init_weights_layerwise,
)
from .loss import LossComputer_official
import torch.nn.functional as F
from utils.pe_utils_official import get_1d_sincos_pos_emb_from_grid, get_2d_sincos_pos_embed
from utils.pose_utils import rot6d2mat, quat2mat
from utils.metric_utils import compute_psnr, compute_lpips, compute_ssim
from utils.data_utils import create_video_from_frames
from PIL import Image


class Images2Latent4D(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.process_data = data_utils.ProcessData(config)
        # self.transform_input = TransformInput(config)

        # image tokenizer
        if not config.model.image_tokenizer.get("use_off_the_shelf", False):
            self.image_tokenizer = nn.Sequential(
                Rearrange(
                    "b v c (hh ph) (ww pw) -> (b v) (hh ww) (ph pw c)",
                    ph=self.config.model.image_tokenizer.patch_size,
                    pw=self.config.model.image_tokenizer.patch_size,
                ),
                nn.Linear(
                    config.model.image_tokenizer.in_channels
                    * (config.model.image_tokenizer.patch_size**2),
                    config.model.transformer.d,
                    bias=False,
                ),
            )
            self.image_tokenizer.apply(_init_weights)
        else:
            # self.image_tokenizer = CustomImageTokenizer(config)
            pass

        # image positional embedding embedder
        self.use_pe_embedding_layer = config.model.get("input_with_pe", True)
        self.pe_embedder = (
            nn.Sequential(
                nn.Linear(
                    config.model.transformer.d,
                    config.model.transformer.d,
                ),
                nn.SiLU(),
                nn.Linear(
                    config.model.transformer.d,
                    config.model.transformer.d,
                ),
            )
            if self.use_pe_embedding_layer
            else nn.Identity()
        )
        self.pe_embedder.apply(_init_weights)

        # latent scene representation
        self.scene_code = nn.Parameter(
            torch.randn(
                config.model.scene_latent.length,
                config.model.transformer.d,
            )
        )
        nn.init.trunc_normal_(self.scene_code, std=0.02)

        # pose tokens
        self.cam_code = nn.Parameter(
            torch.randn(
                self.config.model.pose_latent.get("length", 1),
                config.model.transformer.d,
            )
        )
        nn.init.trunc_normal_(self.cam_code, std=0.02)

        # pose pe temporal embedder
        self.temporal_pe_embedder = (
            nn.Sequential(
                nn.Linear(
                    config.model.transformer.d,
                    config.model.transformer.d,
                ),
                nn.SiLU(),
                nn.Linear(
                    config.model.transformer.d,
                    config.model.transformer.d,
                ),
            )
            if self.use_pe_embedding_layer
            else nn.Identity()
        )
        self.temporal_pe_embedder.apply(_init_weights)

        # qk norm settings
        use_qk_norm = config.model.transformer.get("use_qk_norm", False)

        # transformer encoder and init
        self.transformer_encoder = [
            QK_Norm_TransformerBlock(
                config.model.transformer.d,
                config.model.transformer.d_head,
                use_qk_norm=use_qk_norm,
            )
            for _ in range(config.model.transformer.encoder_n_layer)
        ]
        if config.model.transformer.get("special_init", False):
            if config.model.transformer.get("depth_init", False):
                for idx in range(len(self.transformer_encoder)):
                    weight_init_std = 0.02 / (2 * (idx + 1)) ** 0.5
                    self.transformer_encoder[idx].apply(
                        lambda module: _init_weights_layerwise(module, weight_init_std)
                    )
            else:
                for idx in range(len(self.transformer_encoder)):
                    weight_init_std = 0.02 / (2 * config.model.transformer.encoder_n_layer) ** 0.5
                    self.transformer_encoder[idx].apply(
                        lambda module: _init_weights_layerwise(module, weight_init_std)
                    )
            self.transformer_encoder = nn.ModuleList(self.transformer_encoder)
        else:
            self.transformer_encoder = nn.ModuleList(self.transformer_encoder)
            self.transformer_encoder.apply(_init_weights)

        # transformer encoder2 and init
        self.transformer_encoder_geom = [
            QK_Norm_TransformerBlock(
                config.model.transformer.d,
                config.model.transformer.d_head,
                use_qk_norm=use_qk_norm,
            )
            for _ in range(config.model.transformer.encoder_geom_n_layer)
        ]
        if config.model.transformer.get("special_init", False):
            if config.model.transformer.get("depth_init", False):
                for idx in range(len(self.transformer_encoder_geom)):
                    weight_init_std = 0.02 / (2 * (idx + 1)) ** 0.5
                    self.transformer_encoder_geom[idx].apply(
                        lambda module: _init_weights_layerwise(module, weight_init_std)
                    )
            else:
                for idx in range(len(self.transformer_encoder_geom)):
                    weight_init_std = (
                        0.02 / (2 * config.model.transformer.encoder_geom_n_layer) ** 0.5
                    )
                    self.transformer_encoder_geom[idx].apply(
                        lambda module: _init_weights_layerwise(module, weight_init_std)
                    )
            self.transformer_encoder_geom = nn.ModuleList(self.transformer_encoder_geom)
        else:
            self.transformer_encoder_geom = nn.ModuleList(self.transformer_encoder_geom)
            self.transformer_encoder_geom.apply(_init_weights)

        # ln before decoder
        self.decoder_ln = nn.LayerNorm(config.model.transformer.d, bias=False)

        # transformer decoder and init
        self.transformer_decoder = [
            QK_Norm_TransformerBlock(
                config.model.transformer.d,
                config.model.transformer.d_head,
                use_qk_norm=use_qk_norm,
            )
            for _ in range(config.model.transformer.decoder_n_layer)
        ]
        if config.model.transformer.get("special_init", False):
            if config.model.transformer.depth_init:
                for idx in range(len(self.transformer_decoder)):
                    weight_init_std = 0.02 / (2 * (idx + 1)) ** 0.5
                    self.transformer_decoder[idx].apply(
                        lambda module: _init_weights_layerwise(module, weight_init_std)
                    )
            else:
                for idx in range(len(self.transformer_decoder)):
                    weight_init_std = 0.02 / (2 * config.model.transformer.decoder_n_layer) ** 0.5
                    self.transformer_decoder[idx].apply(
                        lambda module: _init_weights_layerwise(module, weight_init_std)
                    )
            self.transformer_decoder = nn.ModuleList(self.transformer_decoder)
        else:
            self.transformer_decoder = nn.ModuleList(self.transformer_decoder)
            self.transformer_decoder.apply(_init_weights)

        # pose predictor
        self.pose_predictor = PoseEstimator(config)

        # target pose tokenizer
        self.target_latent_h = (
            config.model.target_image.height // config.model.target_image.patch_size
        )
        self.target_latent_w = (
            config.model.target_image.width // config.model.target_image.patch_size
        )
        self.target_pose_tokenizer = nn.Sequential(
            Rearrange(
                "b v c (hh ph) (ww pw) -> (b v) (hh ww) (ph pw c)",
                ph=self.config.model.target_pose_tokenizer.patch_size,
                pw=self.config.model.target_pose_tokenizer.patch_size,
            ),
            nn.Linear(
                config.model.target_pose_tokenizer.in_channels
                * (config.model.target_pose_tokenizer.patch_size**2),
                config.model.transformer.d,
                bias=False,
            ),
        )
        self.target_pose_tokenizer.apply(_init_weights)

        self.target_pose_tokenizer2 = nn.Sequential(
            Rearrange(
                "b v c (hh ph) (ww pw) -> (b v) (hh ww) (ph pw c)",
                ph=self.config.model.target_pose_tokenizer.patch_size,
                pw=self.config.model.target_pose_tokenizer.patch_size,
            ),
            nn.Linear(
                config.model.target_pose_tokenizer.in_channels
                * (config.model.target_pose_tokenizer.patch_size**2),
                config.model.transformer.d,
                bias=False,
            ),
        )
        self.target_pose_tokenizer2.apply(_init_weights)

        # fuse mlp
        self.mlp_fuse = nn.Sequential(
            nn.LayerNorm(config.model.transformer.d * 2, bias=False),
            nn.Linear(
                config.model.transformer.d * 2,
                config.model.transformer.d,
                bias=True,
            ),
            nn.SiLU(),
            nn.Linear(
                config.model.transformer.d,
                config.model.transformer.d,
                bias=True,
            ),
        )
        self.target_pose_tokenizer.apply(_init_weights)

        # output regresser
        self.image_token_decoder = nn.Sequential(
            nn.LayerNorm(config.model.transformer.d, bias=False),
            nn.Linear(
                config.model.transformer.d,
                (config.model.target_image.patch_size**2) * 3,
                bias=False,
            ),
            nn.Sigmoid(),
        )
        self.image_token_decoder.apply(_init_weights)

        # loss
        self.loss_computer = LossComputer_official(config)

        # config backup
        self.config_bk = copy.deepcopy(config)
        self.render_interpolate = config.training.get("render_interpolate", False)

        if config.model.transformer.get("fix_decoder", False):
            self.freeze_weights()

        # training settings
        if config.inference or config.get("evaluation", False):
            if config.training.get("random_inputs", False):
                self.random_index = True
            else:
                self.random_index = False
        else:
            self.random_index = config.training.get("random_split", False)

        print("Use random index:", self.random_index)

    def freeze_weights(self):
        for param in self.target_pose_tokenizer.parameters():
            param.requires_grad = False

        for param in self.image_token_decoder.parameters():
            param.requires_grad = False

        for param in self.transformer_decoder.parameters():
            param.requires_grad = False

    def train(self, mode=True):
        # override the train method to keep the fronzon modules in eval mode
        super().train(mode)
        self.loss_computer.eval()

    def get_overview(self):
        count_train_params = lambda model: sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        overview = edict(
            image_tokenizer=count_train_params(self.image_tokenizer),
            scene_latent=self.scene_code.data.numel(),
            transformer=count_train_params(self.transformer_decoder)
            + count_train_params(self.transformer_encoder),
            image_token_decoder=count_train_params(self.image_token_decoder),
        )
        return overview

    @torch.no_grad()
    def reconstruct_images(self, data_batch):
        """
        Minimal API used by train.py for visualization.
        Delegates to forward with create_visual=True to produce renderings and visuals.
        Returns an edict with at least: input, target, render, and optional render_video.
        """
        return self.forward(data_batch, create_visual=True, render_video=False)

    def forward(self, data, create_visual=False, render_video=False, iter=0):
        """Split all images into two sets, use one set to get scene representation, use the other to render & train"""
        input, target = self.process_data(
            data, has_target_image=True, target_has_input=True, compute_rays=False
        )
        image = input.image * 2.0 - 1.0  # [b, v, c, h, w], range (0,1) to (-1,1)
        b, v_input, c, h, w = image.shape
        image_all = data["image"] * 2.0 - 1.0  # [b, v_all, c, h, w], range (0,1) to (-1,1)
        v_all = image_all.shape[1]
        v_target = (
            target.image.shape[1] if hasattr(target, "image") and target.image is not None else 0
        )
        device = image.device
        batch_idx = torch.arange(b).unsqueeze(1).to(device)

        """se3 pose prediction for all views"""
        # tokenize images, add spatial-temporal p.e.
        img_tokens = self.image_tokenizer(image_all)  # [b * v, n, d]
        _, n, d = img_tokens.shape
        if self.use_pe_embedding_layer:
            img_tokens = self.add_sptial_temporal_pe(img_tokens, b, v_all, h, w)
        img_tokens = rearrange(img_tokens, "(b v) n d -> b (v n) d", b=b, v=v_all)  # [b, v * n, d]

        # get camera tokens, add temporal p.e.
        cam_tokens = self.get_camera_tokens(b, v_all)  # [b, v_all * n_cam, d]
        n_cam = cam_tokens.shape[1] // v_all
        assert n_cam == 1
        cam_tokens = rearrange(cam_tokens, "b (v n) d -> b v n d", v=v_all)  # [b, v_all, n_cam, d]
        cam_tokens = rearrange(cam_tokens, "b v n d -> b (v n) d")  # [b, v_all * n_cam, d]

        # pose estimation for all views
        all_tokens = torch.cat([cam_tokens, img_tokens], dim=1)
        all_tokens = self.run_encoder(all_tokens)
        cam_tokens, _ = all_tokens.split([v_all * n_cam, v_all * n], dim=1)

        # get se3 poses and intrinsics
        cam_tokens = rearrange(cam_tokens, "b (v n) d -> (b v) n d", b=b, v=v_all, n=n_cam)[
            :, 0
        ]  # [b * v_all, d]
        cam_info = self.pose_predictor(
            cam_tokens, v_all
        )  # [b * v_all, num_pose_element+3+4], rot, 3d trans, 4d fxfycxcy
        c2w, fxfycxcy = get_cam_se3(cam_info)  # [b * v_all,4,4], [b * v_all,4]
        normalized = True

        # get plucker ray and embeddings
        plucker_rays = cam_info_to_plucker(
            c2w, fxfycxcy, self.config.model.target_image, normalized=normalized
        )
        plucker_rays = rearrange(plucker_rays, "(b v) c h w -> b v c h w", b=b, v=v_all)
        # Split plucker rays into input and target
        plucker_input = plucker_rays[:, :v_input]  # [b, v_input, 6, h, w]
        plucker_target = (
            plucker_rays[:, v_input : v_input + v_target] if v_target > 0 else None
        )  # [b, v_target, 6, h, w]

        plucker_emb_input = self.target_pose_tokenizer(plucker_input)  # [b * v_input, n, d]
        plucker_emb_target = (
            self.target_pose_tokenizer2(plucker_target) if plucker_target is not None else None
        )  # [b * v_target, n, d]
        plucker_emb_input = rearrange(
            plucker_emb_input, "(b v) n d -> b (v n) d", v=v_input
        )  # [b, v_input * n, d]

        """predict scene representation using (posed) input views"""
        # get posed image representation
        img_tokens_input = rearrange(img_tokens, "b (v n) d -> b v n d", v=v_all)[
            :, :v_input
        ]  # [b, v_input, n, d]
        img_tokens_input = rearrange(img_tokens_input, "b v n d -> b (v n) d")
        img_tokens_input = torch.cat(
            [img_tokens_input, plucker_emb_input], dim=-1
        )  # [b, v_input * n, 2d]
        img_tokens_input = self.mlp_fuse(img_tokens_input)  # [b, v_input * n, d]

        # replicate scene tokens
        scene_tokens = self.scene_code.expand(b, -1, -1)  # [b, n_scene, d]
        n_scene = scene_tokens.shape[1]
        all_tokens = torch.cat(
            [scene_tokens, img_tokens_input], dim=1
        )  # [b, n_scene + v_input * n, d]

        # encoder layers, update scene representation
        all_tokens = self.run_encoder_geom(all_tokens)
        scene_tokens, _ = all_tokens.split(
            [n_scene, v_input * n], dim=1
        )  # [b, n_scene, d], [b, v_input * n, d]

        """render with scene representation from input views and pose of target views"""
        # render target images
        if plucker_emb_target is not None and v_target > 0:
            render_results = self.render_images(scene_tokens, plucker_emb_target)  # dict

            # compute loss
            loss_metrics = self.loss_computer(
                render_results.rendered_images,
                target.image,
                input,
                create_visual=create_visual,
            )
        else:
            # No target views to render - create dummy loss for compatibility
            render_results = edict(rendered_images=None)
            device = scene_tokens.device
            loss_metrics = edict(
                loss=torch.tensor(0.0, device=device, requires_grad=True),
                l2_loss=torch.tensor(0.0, device=device),
                psnr=torch.tensor(0.0, device=device),
                lpips_loss=torch.tensor(0.0, device=device),
                perceptual_loss=torch.tensor(0.0, device=device),
                ssim_loss=torch.tensor(0.0, device=device),
                visual=None,
            )

        if create_visual and v_target > 0:
            with torch.no_grad():
                c2w_all_reshaped = rearrange(c2w, "(b v) c d -> b v c d", v=v_all)
                fxfycxcy_all_reshaped = rearrange(fxfycxcy, "(b v) c -> b v c", v=v_all)

                c2w_target = c2w_all_reshaped[
                    :, v_input : v_input + v_target
                ]  # [b, v_target, 4, 4]
                fxfycxcy_target = fxfycxcy_all_reshaped[
                    :, v_input : v_input + v_target
                ]  # [b, v_target, 4]
                c2w_target = rearrange(c2w_target, "b v c d -> (b v) c d")
                fxfycxcy_target = rearrange(fxfycxcy_target, "b v c -> (b v) c")
                vis_only_results = self.render_images_video(
                    scene_tokens, c2w_target, fxfycxcy_target, normalized=normalized
                )

        # return results
        result = edict(
            input=input,
            target=target,
            loss_metrics=loss_metrics,
            render=render_results.rendered_images,
            c2w=rearrange(c2w, "(b v) c d -> b v c d", b=b, v=v_all),
            fxfycxcy=fxfycxcy,
        )

        if create_visual and v_target > 0:
            result.render_video = vis_only_results.rendered_images_video.detach()

        return result

    def render_images(self, scene_tokens, target_tokens):
        """
        Render target views based on the scene representation, target view pose tokens and target view tokens
        Args:
            scene_tokens: [b, n_scene, d]
            target_tokens: plucker embedding tokens of input images [b*v, n_target, d]
        Return:
            rendered_images: rendered target views in [b,v,c,h,w]
        """
        b, _, d = scene_tokens.shape
        bv = target_tokens.shape[0]
        v = bv // b

        # repeat scene tokens
        scene_tokens = scene_tokens.unsqueeze(1).repeat(1, v, 1, 1)
        scene_tokens = rearrange(scene_tokens, "b v n d -> (b v) n d")  # [b*v, n_scene, d]

        # get all tokens
        n_target, n_scene = target_tokens.shape[1], scene_tokens.shape[1]
        all_tokens = torch.cat(
            [target_tokens, scene_tokens], dim=1
        )  # [b*v, n_target+n_pose+n_scene, d]

        # render
        rendered_images_all = self.render(all_tokens, n_target, n_scene, v)  # [b,v,c,h,w]

        render_results = edict(rendered_images=rendered_images_all)
        return render_results

    def render(self, all_tokens, n_target, n_scene, v):
        """
        Run decoder layers and output mlp to render target views
        Args:
            all_tokens: [b*v, n_target+n_pose+n_scene, d]
        Return:
            rendered views: [b, v, c, h, w]
        """
        all_tokens = self.decoder_ln(all_tokens)
        all_tokens = self.run_decoder(all_tokens)

        # split tokens
        target_tokens, _ = all_tokens.split([n_target, n_scene], dim=1)  # [b*v, n_target, d]

        # regress image
        rendered_images_all = self.image_token_decoder(target_tokens)  # [b*v, n_target, p*p*3]
        patch_size = self.config.model.target_image.patch_size
        rendered_images_all = rearrange(
            rendered_images_all,
            "(b v) (h w) (p1 p2 c) -> b v c (h p1) (w p2)",
            v=v,
            h=self.target_latent_h,
            w=self.target_latent_w,
            p1=patch_size,
            p2=patch_size,
            c=3,
        )
        return rendered_images_all

    def get_camera_tokens(self, b, v):
        n, d = self.cam_code.shape[-2:]

        cam_tokens = rearrange(self.cam_code, "n d -> 1 1 n d")  # [1, 1, n_cam, d]
        cam_tokens = repeat(cam_tokens, "1 1 n d -> 1 v n d", v=v)  # [1, v, n_cam, d]
        cam_tokens = rearrange(cam_tokens, "1 v n d -> 1 (v n) d")  # [1, v*n_cam, d]

        # get temporal pe
        img_indices = torch.arange(v).repeat_interleave(n)  # [v*n_cam]
        img_indices = img_indices.to(cam_tokens.device)
        temporal_pe = get_1d_sincos_pos_emb_from_grid(
            embed_dim=d, pos=img_indices, device=cam_tokens.device
        ).to(
            cam_tokens.dtype
        )  # [v*n_cam, d]
        temporal_pe = temporal_pe.reshape(1, v * n, d)  # [1, v*n_cam, d]
        temporal_pe = self.temporal_pe_embedder(temporal_pe)  # [1, v*n_cam, d]

        return (cam_tokens + temporal_pe).repeat(b, 1, 1)  # [b, v*n_cam, d]

    def add_sptial_temporal_pe(self, img_tokens, b, v, h_origin, w_origin):
        """
        Adding spatial-temporal pe to input image tokens
        Args:
            img_tokens: shape [b*v, n, d]
        Return:
            image tokens with positional embedding
        """
        patch_size = self.config.model.image_tokenizer.patch_size
        num_h_tokens = h_origin // patch_size
        num_w_tokens = w_origin // patch_size
        assert (num_h_tokens * num_w_tokens) == img_tokens.shape[1]
        bv, n, d = img_tokens.shape

        # get temporal pe
        img_indices = torch.arange(v).repeat_interleave(n)  # [v*n]
        img_indices = img_indices.unsqueeze(0).repeat(b, 1).reshape(-1)  # [b*v*n]
        img_indices = img_indices.to(img_tokens.device)
        temporal_pe = get_1d_sincos_pos_emb_from_grid(
            embed_dim=d // 2, pos=img_indices, device=img_tokens.device
        ).to(
            img_tokens.dtype
        )  # [b*v*n, d2]
        temporal_pe = temporal_pe.reshape(b, v, n, d // 2)  # [b,v,n,d2]

        # get spatial pe
        spatial_pe = get_2d_sincos_pos_embed(
            embed_dim=d // 2,
            grid_size=(num_h_tokens, num_w_tokens),
            device=img_tokens.device,
        ).to(
            img_tokens.dtype
        )  # [n, d3]
        spatial_pe = spatial_pe.reshape(1, 1, n, d // 2).repeat(b, v, 1, 1)  # [b,v,n,d3]

        # embed pe
        pe = self.pe_embedder(
            torch.cat([spatial_pe, temporal_pe], dim=-1).reshape(bv, n, d)
        )  # [b*v,n,d]

        return img_tokens + pe

    def run_layers_encoder(self, start, end):
        def custom_forward(concat_nerf_img_tokens):
            for i in range(start, min(end, len(self.transformer_encoder))):
                concat_nerf_img_tokens = self.transformer_encoder[i](concat_nerf_img_tokens)
            return concat_nerf_img_tokens

        return custom_forward

    def run_layers_encoder_geom(self, start, end):
        def custom_forward(concat_nerf_img_tokens):
            for i in range(start, min(end, len(self.transformer_encoder_geom))):
                concat_nerf_img_tokens = self.transformer_encoder_geom[i](concat_nerf_img_tokens)
            return concat_nerf_img_tokens

        return custom_forward

    def run_layers_decoder(self, start, end):
        def custom_forward(concat_nerf_img_tokens):
            for i in range(start, min(end, len(self.transformer_decoder))):
                concat_nerf_img_tokens = self.transformer_decoder[i](concat_nerf_img_tokens)
            return concat_nerf_img_tokens

        return custom_forward

    def run_encoder(self, all_tokens_encoder):
        checkpoint_every = self.config.training.grad_checkpoint_every
        for i in range(0, len(self.transformer_encoder), checkpoint_every):
            all_tokens_encoder = torch.utils.checkpoint.checkpoint(
                self.run_layers_encoder(i, i + 1),
                all_tokens_encoder,
                use_reentrant=False,
            )
            if checkpoint_every > 1:
                all_tokens_encoder = self.run_layers_encoder(i + 1, i + checkpoint_every)(
                    all_tokens_encoder
                )
        return all_tokens_encoder

    def run_encoder_geom(self, all_tokens_encoder):
        checkpoint_every = self.config.training.grad_checkpoint_every
        for i in range(0, len(self.transformer_encoder_geom), checkpoint_every):
            all_tokens_encoder = torch.utils.checkpoint.checkpoint(
                self.run_layers_encoder_geom(i, i + 1),
                all_tokens_encoder,
                use_reentrant=False,
            )
            if checkpoint_every > 1:
                all_tokens_encoder = self.run_layers_encoder_geom(i + 1, i + checkpoint_every)(
                    all_tokens_encoder
                )
        return all_tokens_encoder

    def run_decoder(self, all_tokens_encoder):
        checkpoint_every = self.config.training.grad_checkpoint_every
        for i in range(0, len(self.transformer_decoder), checkpoint_every):
            all_tokens_encoder = torch.utils.checkpoint.checkpoint(
                self.run_layers_decoder(i, i + 1),
                all_tokens_encoder,
                use_reentrant=False,
            )
            if checkpoint_every > 1:
                all_tokens_encoder = self.run_layers_decoder(i + 1, i + checkpoint_every)(
                    all_tokens_encoder
                )
        return all_tokens_encoder

    def render_images_video(self, scene_tokens_all, c2w_all, fxfycxcy_all, normalized=False):
        """
        scene_tokens_all: [b, n_scene, d]
        c2w_all: [b*v, 4, 4]
        fxfycxcy_all: [b*v, 4]
        """
        with torch.no_grad():
            scene_tokens_all = scene_tokens_all.detach()
            c2w_all = c2w_all.detach()
            fxfycxcy_all = fxfycxcy_all.detach()

            b, _, d = scene_tokens_all.shape
            bv = c2w_all.shape[0]
            v = bv // b
            c2w_all = rearrange(c2w_all, "(b v) x y -> b v x y", v=v)
            fxfycxcy_all = rearrange(fxfycxcy_all, "(b v) x -> b v x", v=v)
            device = scene_tokens_all.device

            all_renderings = []
            num_frames = 30
            traj_type = "interpolate"
            order_poses = False

            for i in range(b):
                scene_tokens = scene_tokens_all[i]  # [n_scene, d]
                c2ws = c2w_all[i]  # [v, 4, 4]
                fxfycxcy = fxfycxcy_all[i]  # [v, 4]
                if traj_type == "interpolate":
                    # build Ks from fxfycxcy
                    Ks = torch.zeros((c2ws.shape[0], 3, 3), device=device)
                    Ks[:, 0, 0] = fxfycxcy[:, 0]
                    Ks[:, 1, 1] = fxfycxcy[:, 1]
                    Ks[:, 0, 2] = fxfycxcy[:, 2]
                    Ks[:, 1, 2] = fxfycxcy[:, 3]
                    c2ws, Ks = camera_utils.get_interpolated_poses_many(
                        c2ws[:, :3, :4], Ks, num_frames, order_poses=order_poses
                    )
                    frame_c2ws = torch.cat(
                        [
                            c2ws.to(device),
                            torch.tensor([[[0, 0, 0, 1]]], device=device).repeat(
                                c2ws.shape[0], 1, 1
                            ),
                        ],
                        dim=1,
                    )  # [v',4,4]
                    frame_fxfycxcy = torch.zeros((c2ws.shape[0], 4), device=device)
                    frame_fxfycxcy[:, 0] = Ks[:, 0, 0]  # [v',4]
                    frame_fxfycxcy[:, 1] = Ks[:, 1, 1]
                    frame_fxfycxcy[:, 2] = Ks[:, 0, 2]
                    frame_fxfycxcy[:, 3] = Ks[:, 1, 2]
                elif traj_type == "same":
                    frame_c2ws = c2ws.clone()
                    frame_fxfycxcy = fxfycxcy.clone()
                else:
                    raise NotImplementedError

                plucker_rays = cam_info_to_plucker(
                    frame_c2ws,
                    frame_fxfycxcy,
                    self.config.model.target_image,
                    normalized=normalized,
                )  # [v',6,h,w]
                plucker_embeddings = self.target_pose_tokenizer2(
                    plucker_rays.unsqueeze(0)
                )  # [v',n_target,d]

                v_render = plucker_embeddings.shape[0]

                scene_tokens = scene_tokens.unsqueeze(0).repeat(v_render, 1, 1)  # [v',n_scene,d]
                all_tokens = torch.cat(
                    [plucker_embeddings, scene_tokens], dim=1
                )  # [v', n_target+n_scene, d]

                # render
                n_target, n_scene = plucker_embeddings.shape[1], scene_tokens.shape[1]
                rendered_images_all = self.render(all_tokens, n_target, n_scene, v_render).squeeze(
                    0
                )  # [v',c,h,w]
                all_renderings.append(rendered_images_all)

            all_renderings = torch.stack(all_renderings)  # [b,v',c,h,w]

        render_results = edict(rendered_images_video=all_renderings)
        return render_results

    @torch.no_grad()
    def save_visuals(self, out_dir, result, batch, save_all=False):
        os.makedirs(out_dir, exist_ok=True)

        input, target = result.input, result.target

        # save comparison
        if result.loss_metrics.visual is not None:
            uids = [target.index[b, 0, -1].item() for b in range(target.index.size(0))]

            uid_firstlast = f"{uids[0]:08}_{uids[-1]:08}"
            Image.fromarray(result.loss_metrics.visual).save(
                os.path.join(out_dir, f"supervision_{uid_firstlast}.jpg")
            )
            with open(os.path.join(out_dir, f"uids.txt"), "w") as f:
                uids = "_".join([f"{uid:08}" for uid in uids])
                f.write(uids)

        # save input, rendered interpolated imaged, video
        for b in range(input.image.size(0)):
            uid = input.index[b, 0, -1].item()

            # vis input
            vis_image = rearrange(input.image[b], "v c h w -> h (v w) c")
            vis_image = (vis_image.cpu().numpy() * 255.0).clip(0.0, 255.0).astype(np.uint8)
            Image.fromarray(vis_image).save(os.path.join(out_dir, f"input_{uid}.png"))

            if ("render_video" in result.keys()) and torch.is_tensor(result.render_video):
                render_video = result.render_video[b].detach().cpu()  # [v, c, h, w]
                render_video = np.ascontiguousarray(np.array(render_video.to(torch.float32)))
                render_video = rearrange(render_video, "v c h w -> v h w c")
                create_video_from_frames(
                    render_video, f"{out_dir}/render_video_{uid}.mp4", framerate=30
                )

            if not save_all:
                break

    @torch.no_grad()
    def save_evaluations(self, out_dir, result, batch, dataset):
        os.makedirs(out_dir, exist_ok=True)
        input, target = result.input, result.target
        for b in range(input.image.size(0)):
            uid = input.index[b, 0, -1].item()
            curr_out_dir = os.path.join(out_dir, "%06d" % (uid))
            os.makedirs(curr_out_dir, exist_ok=True)
            curr_out_color_dir = curr_out_dir
            os.makedirs(curr_out_color_dir, exist_ok=True)

            target_idx = target.index[b, :, 0].cpu().numpy()
            _, v, _, h, w = input.image.size()

            # vis gt and predicted target views
            im = torch.cat((result.target.image[b], result.render[b]), dim=2).detach().cpu()
            im = rearrange(im, "v c h w -> h (v w) c")
            im = (im.cpu().numpy() * 255.0).clip(0.0, 255.0).astype(np.uint8)
            Image.fromarray(im).save(os.path.join(curr_out_color_dir, f"gt_vs_pred.png"))

            # vis input views
            input_im = result.input.image[b]  # V C H W
            input_im = rearrange(input_im, "v c h w -> h (v w) c")
            input_im = (input_im.cpu().numpy() * 255.0).clip(0.0, 255.0).astype(np.uint8)
            Image.fromarray(input_im).save(os.path.join(curr_out_color_dir, f"input.png"))

            # # vis interpolated views
            # if torch.is_tensor(result.render_interpolate):
            #     interpolate_im = result.render_interpolate[b].detach().cpu().to(torch.float32)
            #     interpolate_im = rearrange(interpolate_im, 'v c h w -> h (v w) c')
            #     interpolate_im = (interpolate_im.cpu().numpy() * 255.0).clip(0.0, 255.0).astype(np.uint8)
            #     Image.fromarray(interpolate_im).save(
            #         os.path.join(curr_out_color_dir, f"pred_interpolate.png"))

            # get metrics
            psnr = compute_psnr(target.image[b], result.render[b])  # .mean()
            lpips = compute_lpips(target.image[b], result.render[b])  # .mean()
            ssim = compute_ssim(
                target.image[b].to(torch.float32), result.render[b].to(torch.float32)
            )  # .mean()

            with open(os.path.join(curr_out_dir, "per_view_metrics.txt"), "w") as f:
                for i in range(psnr.shape[0]):
                    f.write(f"view: {target_idx[i]}, psnr: {psnr[i]},")
                    f.write(f"lpips: {lpips[i]},")
                    f.write(f"ssim: {ssim[i]}\n")

            with open(os.path.join(curr_out_dir, "metrics.txt"), "w") as f:
                metrics = f"psnr: {psnr.mean()}\nlpips: {lpips.mean()}\nssim: {ssim.mean()}\n"
                f.write(metrics)

            # save rendered video
            if torch.is_tensor(result.get("render_video", None)):
                all_frames = result.render_video[b].detach().cpu()  # [v h w c]
                all_frames = np.ascontiguousarray(np.array(all_frames.to(torch.float32)))
                all_frames = rearrange(all_frames, "v c h w -> v h w c")
                create_video_from_frames(all_frames, f"{curr_out_dir}/input_traj.mp4", framerate=30)

            # save rendered video
            if torch.is_tensor(result.get("render_video", None)):
                all_frames = result.render_video[b].detach().cpu()  # [v h w c]
                all_frames = np.ascontiguousarray(np.array(all_frames.to(torch.float32)))
                all_frames = rearrange(all_frames, "v c h w -> v h w c")
                create_video_from_frames(all_frames, f"{curr_out_dir}/input_traj.mp4", framerate=30)

            # visualize cameras (placeholder - implement if needed)
            # c2w = result.c2w[b].detach().cpu()
            # input_idx_cur = input_idx[b].detach().cpu()
            # c2w_input = c2w[input_idx_cur]
            # TODO: Implement camera trajectory visualization if needed
            # pass


def get_cam_se3(cam_info):
    """
    cam_info: [b,num_pose_element+3+4], rot, 3d trans, 4d fxfycxcy
    """
    b, n = cam_info.shape

    if n == 13:
        rot_6d = cam_info[:, :6]
        R = rot6d2mat(rot_6d)  # [b,3,3]
        t = cam_info[:, 6:9].unsqueeze(-1)  # [b,3,1]
        fxfycxcy = cam_info[:, 9:]  # normalized by resolution / shift from average, [b,4]
    elif n == 11:
        rot_quat = cam_info[:, :4]
        R = quat2mat(rot_quat)
        t = cam_info[:, 4:7].unsqueeze(-1)  # [b,3,1]
        fxfycxcy = cam_info[:, 7:]  # normalized by resolution / shift from average, [b,4]
    else:
        raise NotImplementedError

    Rt = torch.cat([R, t], dim=2)  # [b,3,4]
    c2w = torch.cat(
        [
            Rt,
            torch.tensor([0, 0, 0, 1], dtype=R.dtype, device=R.device)
            .view(1, 1, 4)
            .repeat(b, 1, 1),
        ],
        dim=1,
    )  # [b,4,4]
    return c2w, fxfycxcy


def cam_info_to_plucker(c2w, fxfycxcy, target_imgs_info, normalized=True):
    """
    c2w: [b,4,4]
    fxfycxcy: [b,4]
    """
    b = c2w.shape[0]
    device = c2w.device
    h, w = target_imgs_info.height, target_imgs_info.width

    fxfycxcy = fxfycxcy.clone()
    if normalized:
        fxfycxcy[:, 0] *= w
        fxfycxcy[:, 1] *= h
        fxfycxcy[:, 2] *= w
        fxfycxcy[:, 3] *= h

    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    y, x = y.to(c2w), x.to(c2w)
    x = x[None, :, :].expand(b, -1, -1).reshape(b, -1)
    y = y[None, :, :].expand(b, -1, -1).reshape(b, -1)
    x = (x + 0.5 - fxfycxcy[:, 2:3]) / fxfycxcy[:, 0:1]
    y = (y + 0.5 - fxfycxcy[:, 3:4]) / fxfycxcy[:, 1:2]
    z = torch.ones_like(x)
    ray_d = torch.stack([x, y, z], dim=2)  # [b, h*w, 3]
    ray_d = torch.bmm(ray_d, c2w[:, :3, :3].transpose(1, 2))  # [b, h*w, 3]
    ray_d = ray_d / torch.norm(ray_d, dim=2, keepdim=True)  # [b, h*w, 3]
    ray_o = c2w[:, :3, 3][:, None, :].expand_as(ray_d)  # [b, h*w, 3]

    ray_o = ray_o.reshape(b, h, w, 3).permute(0, 3, 1, 2)  # [b,3,h,w]
    ray_d = ray_d.reshape(b, h, w, 3).permute(0, 3, 1, 2)

    plucker = torch.cat(
        [
            torch.cross(ray_o, ray_d, dim=1),
            ray_d,
        ],
        dim=1,
    )
    return plucker  # [b,c=6,h,w]


class PoseEstimator(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.pose_rep = self.config.model.pose_latent.get("representation", "6d")
        print("Pose representation:", self.pose_rep)
        if self.pose_rep == "6d":
            self.num_pose_element = 6
        elif self.pose_rep == "quat":
            self.num_pose_element = 4
        else:
            raise NotImplementedError

        self.rel_head = nn.Sequential(
            nn.Linear(
                config.model.transformer.d * 2,
                config.model.transformer.d,
                bias=True,
            ),
            nn.SiLU(),
            nn.Linear(
                config.model.transformer.d,
                self.num_pose_element + 3,
                bias=True,
            ),
        )
        self.rel_head.apply(_init_weights)

        self.canonical_k_head = nn.Sequential(
            nn.Linear(
                config.model.transformer.d,
                config.model.transformer.d,
                bias=True,
            ),
            nn.SiLU(),
            nn.Linear(
                config.model.transformer.d,
                1,
                bias=True,
            ),
        )
        self.canonical_k_head.apply(_init_weights)

        self.f_bias = 1.25

    def forward(self, x, v):
        """
        x: [b*v, d]
        """
        canonical = self.config.model.pose_latent.get("canonical", "first")
        x = rearrange(x, "(b v) d -> b v d", v=v)
        b = x.shape[0]
        if canonical == "first":
            x_canonical = x[:, 0:1]  # [b,1,d]
            x_rel = x[:, 1:]  # [b,v-1,d]
        elif canonical == "middle":
            cano_idx = v // 2
            rel_indices = torch.cat([torch.arange(cano_idx), torch.arange(cano_idx + 1, v)])
            x_canonical = x[:, cano_idx : cano_idx + 1]  # [b,1,d]
            x_rel = x[:, rel_indices]  # [b,v-1,d]
        else:
            raise NotImplementedError

        fxfy_canonical = self.canonical_k_head(x_canonical[:, 0]) + self.f_bias  # [b,1]
        fxfy_canonical = fxfy_canonical.unsqueeze(1).repeat(1, 1, 2)  # [b,1,2]

        if self.pose_rep == "6d":
            rt_canonical = (
                torch.tensor([1, 0, 0, 0, 1, 0, 0, 0, 0])
                .reshape(1, 1, 9)
                .to(fxfy_canonical)
                .repeat(b, 1, 1)
            )  # [b,1,9]
        elif self.pose_rep == "quat":
            rt_canonical = (
                torch.tensor([1, 0, 0, 0, 0, 0, 0])
                .reshape(1, 1, 7)
                .to(fxfy_canonical)
                .repeat(b, 1, 1)
            )  # [b,1,7]
        info_canonical = torch.cat([rt_canonical, fxfy_canonical], dim=-1)  # [b,1,11]

        feat_rel = torch.cat([x_canonical.repeat(1, v - 1, 1), x_rel], dim=-1)  # [b,v-1,2*d]
        info_rel = self.rel_head(feat_rel)  # [b,v-1,num_pose_element+3]
        info_all = info_canonical.repeat(1, v, 1)  # [b,v,num_pose_element+3+2]

        if canonical == "first":
            info_all[:, 1:, : self.num_pose_element + 3] += info_rel
        elif canonical == "middle":
            info_all[:, rel_indices, : self.num_pose_element + 3] += info_rel
        else:
            raise NotImplementedError

        cxcy_all = torch.tensor([0.5, 0.5]).reshape(1, 1, 2).repeat(b, v, 1).to(info_all)
        info_all = torch.cat([info_all, cxcy_all], dim=-1)
        return rearrange(info_all, "b v d -> (b v) d")
