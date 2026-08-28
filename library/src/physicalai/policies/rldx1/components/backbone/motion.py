# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Vendored from RLWRLD/RLDX-1 (Apache-2.0)

"""Motion module for RLDX-1 backbone."""

import torch
import torch.nn.functional as F  # noqa: N812
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn


class STSSTransformation(nn.Module):
    """Compute local spatio-temporal self-similarity (STSS) correlation volumes.

    For each token in a video feature map, computes pairwise cosine or dot-product
    similarity against a spatial neighbourhood in nearby frames, producing a local
    (temporal-window x spatial-window x spatial-window) correlation tensor.
    """

    def __init__(self, window: tuple[int, int, int] = (5, 9, 9), corr_func: str = "cosine") -> None:
        """Initialize STSSTransformation.

        Args:
            window: (T, H, W) local window sizes for the temporal, height, and width axes.
                H and W must be equal.
            corr_func: Correlation function to use. One of ``"cosine"``,
                ``"dotproduct"``, or ``"dotproduct_softmax"``.

        Raises:
            ValueError: If ``window[1] != window[2]`` (height and width window sizes differ).
        """
        super().__init__()
        self.window = window
        if window[1] != window[2]:
            msg = f"window height and width must be equal, got {window[1]} and {window[2]}"
            raise ValueError(msg)
        self.corr_func = corr_func
        if self.corr_func == "cosine":
            self.pad_value: float = 0
        elif self.corr_func == "dotproduct_softmax":
            self.pad_value = -float("Inf")
        else:
            self.pad_value = 0

    def _convert_global_to_local(self, corr_g: torch.Tensor) -> torch.Tensor:
        """Convert absolute correlation to relative (local window) correlation.

        Args:
            corr_g: Global correlation tensor of shape ``(B, H, W, H, W)``.

        Returns:
            Local correlation tensor of shape ``(B, H, W, window_h, window_w)``.
        """
        max_d = self.window[1] // 2

        corr_l = [
            F.pad(
                torch.diagonal(corr_g, offset=i, dim1=1, dim2=3),
                (abs(i) if i < 0 else 0, abs(i) if i >= 0 else 0),
                value=self.pad_value,
            )
            for i in range(-max_d, max_d + 1)
        ]
        corr_l = torch.stack(corr_l, dim=-1)  # B, W1, W2, H1, H2 -> U

        corr_l = [
            F.pad(
                torch.diagonal(corr_l, offset=i, dim1=1, dim2=2),
                (abs(i) if i < 0 else 0, abs(i) if i >= 0 else 0),
                value=self.pad_value,
            )
            for i in range(-max_d, max_d + 1)
        ]
        corr_l = torch.stack(corr_l, dim=-1)  # B, H1, H2 -> U, W1, W2 -> V
        corr_l = corr_l.transpose(2, 3).contiguous()  # B, H1, W1, U, V

        return corr_l

    def _correlation(self, feat1: torch.Tensor, feat2: torch.Tensor) -> torch.Tensor:
        """Compute pairwise correlation between two spatial feature maps.

        Applies the configured similarity function and crops the full global
        correlation to the local spatial window via ``_convert_global_to_local``.

        Args:
            feat1: Source feature map of shape ``(B, C, H, W)``.
            feat2: Target feature map of shape ``(B, C, H, W)``.

        Returns:
            Local correlation tensor of shape ``(B, H, W, window_h, window_w)``.
        """
        if self.corr_func == "cosine":
            feat1 = F.normalize(feat1, p=2, dim=1)
            feat2 = F.normalize(feat2, p=2, dim=1)
        elif self.corr_func in {"dotproduct", "dotproduct_softmax"}:
            scale = feat1.size(1) ** -0.5
            feat1 *= scale

        corr = torch.einsum("bchw,bcuv->bhwuv", feat1, feat2)
        corr = self._convert_global_to_local(corr)

        if self.corr_func == "dotproduct_softmax":
            corr_shape = corr.shape
            corr = rearrange(corr, "b h w u v -> b h w (u v)")
            corr = F.softmax(corr, dim=-1)
            corr = corr.reshape(corr_shape)

        return corr

    def forward(self, x: torch.Tensor, grid_sizes: torch.Tensor) -> torch.Tensor:
        """Compute STSS correlation volumes for a batch of video tokens.

        Args:
            x: Flattened token tensor of shape ``(B*T*H*W, C)``.
            grid_sizes: Tensor of shape ``(B, 3)`` where each row is ``[t, h, w]``.
                All rows are assumed equal (single video size per batch).

        Returns:
            Correlation volume of shape ``(B, T, H, W, 1, window_t, window_h, window_w)``.
        """
        t, h, w = grid_sizes[0]
        if self.window[0] > 1:
            x = rearrange(x, "(b t h w) c -> b t c h w", t=t, h=h, w=w)
            x_src = repeat(x, "b t c h w -> (b t l) c h w", l=self.window[0])
            # Replicate-pad the temporal axis: edge frames repeat instead of being
            # zero-padded, so boundary correlations reflect "no motion" rather than
            # "motion against a blank frame".
            pad_t = self.window[0] // 2
            x_pad = torch.cat(
                [
                    x[:, :1].expand(-1, pad_t, -1, -1, -1),
                    x,
                    x[:, -1:].expand(-1, pad_t, -1, -1, -1),
                ],
                dim=1,
            )
            x_tgt = x_pad.unfold(1, self.window[0], 1)
            x_tgt = rearrange(x_tgt, "b t c h w l -> (b t l) c h w")
        else:
            x_src = x
            x = rearrange(x, "(b t h w) c -> b t c h w", t=t, h=h, w=w)
            x_tgt = torch.cat((x[:, 0].unsqueeze(1), x[:, :-1]), 1)
            x_tgt = rearrange(x_tgt, "b t c h w -> (b t) c h w")

        stss = self._correlation(x_src, x_tgt)
        return rearrange(stss, "(b t l) h w u v -> b t h w 1 l u v", t=t, l=self.window[0])


class STSSExtraction(nn.Module):
    """Project STSS correlation volumes to compact channel features.

    Flattens the spatial-window dimensions of the correlation volume and applies
    a 1x1x1 Conv3d to project them to a learnable feature space.
    """

    def __init__(
        self,
        window: tuple[int, int, int] = (5, 9, 9),
        chnls: tuple[int, ...] = (256,),
        use_layernorm: bool = False,  # noqa: FBT001, FBT002
        use_syncbn: bool = False,  # noqa: FBT001, FBT002
    ) -> None:
        """Initialize STSSExtraction.

        Args:
            window: (T, H, W) local window sizes matching those used in
                :class:`STSSTransformation`.
            chnls: Tuple with a single int specifying the output channel count
                of the projection conv.
            use_layernorm: Use ``GroupNorm(1, chnls[0])`` instead of BatchNorm.
            use_syncbn: Use ``SyncBatchNorm`` instead of ``BatchNorm3d``.
                Ignored when ``use_layernorm`` is ``True``.
        """
        super().__init__()
        self.window = window
        self.chnls = chnls

        if use_layernorm:
            norm_layer = nn.GroupNorm(1, chnls[0])
        elif use_syncbn:
            norm_layer = nn.SyncBatchNorm(chnls[0])
        else:
            norm_layer = nn.BatchNorm3d(chnls[0])

        self.conv0 = nn.Sequential(
            nn.Conv3d(
                self.window[1] * self.window[2],
                chnls[0],
                kernel_size=(1, 1, 1),
                stride=(1, 1, 1),
                padding=(0, 0, 0),
                bias=False,
            ),
            norm_layer,
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from a correlation volume.

        Args:
            x: Correlation volume of shape
                ``(B, T, H, W, 1, window_t, window_h, window_w)``.

        Returns:
            Feature tensor of shape ``(B*window_t, chnls[0], T, H, W)``.
        """
        _, t, h, w, _, _, _, _ = x.size()
        x = rearrange(x, "b t h w 1 l u v -> (b l) (u v) t h w", t=t, h=h, w=w)
        return self.conv0(x)


class STSSIntegration(nn.Module):
    """Fuse temporal-window STSS feature slices into a single motion feature map.

    In ``"lite"`` mode a single 1x1x1 Conv3d fuses all temporal slices at once.
    In the default mode a three-layer 3x3 Conv3d stack is used for richer spatial
    mixing before the final temporal fusion.
    """

    def __init__(
        self,
        d_in: int,
        window: tuple[int, int, int] = (5, 9, 9),
        chnls: tuple[int, int, int] = (64, 64, 64),
        use_layernorm: bool = False,  # noqa: FBT001, FBT002
        use_syncbn: bool = False,  # noqa: FBT001, FBT002
        mode: str = "lite",
    ) -> None:
        """Initialize STSSIntegration.

        Args:
            d_in: Input channel count (output of :class:`STSSExtraction`).
            window: (T, H, W) local window sizes.
            chnls: Three-element tuple with the channel counts for the three
                conv layers. In ``"lite"`` mode only ``chnls[-1]`` is used.
            use_layernorm: Use ``GroupNorm(1, ...)`` instead of BatchNorm.
            use_syncbn: Use ``SyncBatchNorm`` instead of ``BatchNorm3d``.
                Ignored when ``use_layernorm`` is ``True``.
            mode: ``"lite"`` for a single 1x1x1 fusion conv, or any other value
                for the full three-layer spatial conv stack.
        """
        super().__init__()
        self.window = window
        self.mode = mode

        if mode == "lite":
            # Single 1x1 Conv3d: L fuse + channel projection, no spatial mixing, no norm.
            # Replaces the 3-layer 3x3 conv stack so motion module contributes from step 1
            # (no residual-layer-scale warm-up).
            self.fuse = nn.Sequential(
                Rearrange("(b l) c t h w -> b (l c) t h w", l=self.window[0]),
                nn.Conv3d(d_in * self.window[0], chnls[-1], kernel_size=(1, 1, 1), bias=False),
                nn.GELU(),
            )
            return

        if use_layernorm:
            norm_layer0 = nn.GroupNorm(1, chnls[0])
            norm_layer1 = nn.GroupNorm(1, chnls[1])
            norm_layer2 = nn.GroupNorm(1, chnls[2])
        elif use_syncbn:
            norm_layer0 = nn.SyncBatchNorm(chnls[0])
            norm_layer1 = nn.SyncBatchNorm(chnls[1])
            norm_layer2 = nn.SyncBatchNorm(chnls[2])
        else:
            norm_layer0 = nn.BatchNorm3d(chnls[0])
            norm_layer1 = nn.BatchNorm3d(chnls[1])
            norm_layer2 = nn.BatchNorm3d(chnls[2])

        self.conv0 = nn.Sequential(
            nn.Conv3d(
                d_in,
                chnls[0],
                kernel_size=(1, 3, 3),
                stride=(1, 1, 1),
                padding=(0, 1, 1),
                bias=False,
            ),
            norm_layer0,
            nn.GELU(),
        )
        self.conv1 = nn.Sequential(
            nn.Conv3d(
                chnls[0],
                chnls[1],
                kernel_size=(1, 3, 3),
                stride=(1, 1, 1),
                padding=(0, 1, 1),
                bias=False,
            ),
            norm_layer1,
            nn.GELU(),
        )
        self.conv2_fuse = nn.Sequential(
            Rearrange("(b l) c t h w -> b (l c) t h w", l=self.window[0]),
            nn.Conv3d(
                chnls[1] * self.window[0],
                chnls[2],
                kernel_size=(1, 3, 3),
                stride=(1, 1, 1),
                padding=(0, 1, 1),
                bias=False,
            ),
            norm_layer2,
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Integrate temporal STSS feature slices.

        Args:
            x: Feature tensor of shape ``(B*window_t, C, T, H, W)`` produced by
                :class:`STSSExtraction`.

        Returns:
            Integrated feature map of shape ``(B, chnls[-1], T, H, W)``.
        """
        if self.mode == "lite":
            return self.fuse(x)
        x = self.conv0(x)
        x = self.conv1(x)
        return self.conv2_fuse(x)


class STSSEncoder(nn.Module):
    """Single STSS encoder block combining projection, transformation, extraction, and integration.

    Applies a linear input projection, computes the STSS correlation volume, extracts
    compact features, integrates temporal slices, and projects to the output dimension.
    """

    def __init__(
        self,
        d_in: int,
        d_hid: int,
        d_out: int,
        window: tuple[int, int, int] = (7, 11, 11),
        ext_chnls: tuple[int, ...] = (256,),
        int_chnls: tuple[int, int, int] = (256, 256, 512),
        corr_func: str = "cosine",
        use_layernorm: bool = False,  # noqa: FBT001, FBT002
        use_syncbn: bool = False,  # noqa: FBT001, FBT002
        int_mode: str = "lite",
    ) -> None:
        """Initialize STSSEncoder.

        Args:
            d_in: Input token feature dimension.
            d_hid: Hidden dimension after the input projection.
            d_out: Output feature dimension.
            window: (T, H, W) local window sizes passed to :class:`STSSTransformation`.
            ext_chnls: Channel counts for :class:`STSSExtraction`.
            int_chnls: Channel counts for :class:`STSSIntegration`.
            corr_func: Correlation function; see :class:`STSSTransformation`.
            use_layernorm: Use GroupNorm instead of BatchNorm in extraction/integration.
            use_syncbn: Use SyncBatchNorm instead of BatchNorm in extraction/integration.
            int_mode: Integration mode passed to :class:`STSSIntegration`.
        """
        super().__init__()
        self.window = window
        self.ln_pre = nn.LayerNorm(d_in, eps=1e-6)
        self.in_proj = nn.Linear(d_in, d_hid)
        self.stss_transformation = STSSTransformation(window=window, corr_func=corr_func)
        self.stss_extraction = STSSExtraction(
            window=window,
            chnls=ext_chnls,
            use_layernorm=use_layernorm,
            use_syncbn=use_syncbn,
        )
        self.stss_integration = STSSIntegration(
            ext_chnls[-1],
            window=window,
            chnls=int_chnls,
            use_layernorm=use_layernorm,
            use_syncbn=use_syncbn,
            mode=int_mode,
        )
        self.out_proj = nn.Linear(int_chnls[-1], d_out)

    def forward(self, x: torch.Tensor, grid_sizes: torch.Tensor) -> torch.Tensor:
        """Run the STSS encoder on a batch of video tokens.

        Args:
            x: Flattened token tensor of shape ``(B*T*H*W, d_in)``.
            grid_sizes: Tensor of shape ``(B, 3)`` with rows ``[t, h, w]``.

        Returns:
            Motion feature tensor of shape ``(B*T*H*W, d_out)``.
        """
        x = self.in_proj(self.ln_pre(x))
        x = self.stss_transformation(x, grid_sizes)
        x = self.stss_extraction(x)
        x = self.stss_integration(x)
        return self.out_proj(rearrange(x, "b c t h w -> (b t h w) c"))


class MotionModule(nn.Module):
    """Motion feature extractor built from a stack of :class:`STSSEncoder` blocks.

    Accepts flattened video tokens with heterogeneous or homogeneous spatial grids and
    produces per-token motion features by summing the outputs of all STSS encoders.
    Optionally applies a learnable layer-scale parameter instead of a final linear
    projection, and supports gradient monitoring during training.
    """

    def __init__(
        self,
        d_in: int,
        d_hid: int,
        d_out: int,
        window: tuple[int, int, int] = (5, 9, 9),
        ext_chnls: tuple[int, ...] = (256,),
        int_chnls: tuple[int, int, int] = (256, 256, 512),
        corr_func: str = "cosine",
        n_encoders: int = 1,
        use_layerscale: bool = False,  # noqa: FBT001, FBT002
        layerscale_init: float = 1e-5,
        use_layernorm: bool = False,  # noqa: FBT001, FBT002
        use_syncbn: bool = False,  # noqa: FBT001, FBT002
        gradient_check: bool = False,  # noqa: FBT001, FBT002
        int_mode: str = "lite",
    ) -> None:
        """Initialize MotionModule.

        Args:
            d_in: Input token feature dimension.
            d_hid: Hidden dimension inside each :class:`STSSEncoder`.
            d_out: Output feature dimension.
            window: (T, H, W) local window sizes for all STSS encoders.
            ext_chnls: Extraction channel counts passed to each encoder.
            int_chnls: Integration channel counts passed to each encoder.
            corr_func: Correlation function; see :class:`STSSTransformation`.
            n_encoders: Number of stacked :class:`STSSEncoder` blocks.
            use_layerscale: When ``True``, replace the output linear projection with a
                learnable per-channel scale parameter.
            layerscale_init: Initial value for the layer-scale parameter.
            use_layernorm: Use GroupNorm instead of BatchNorm inside encoders.
            use_syncbn: Use SyncBatchNorm inside encoders.
            gradient_check: When ``True``, register a backward hook that logs
                gradient statistics of the module output during training.
            int_mode: Integration mode passed to each :class:`STSSEncoder`.
        """
        super().__init__()
        self.use_layerscale = use_layerscale
        self.layerscale_init = layerscale_init
        self.gradient_check = gradient_check

        self.stss_encoders = nn.ModuleList(
            [
                STSSEncoder(
                    d_in=d_in if i == 0 else (d_out if self.use_layerscale else d_hid),
                    d_hid=d_hid,
                    d_out=d_out if self.use_layerscale else d_hid,
                    window=window,
                    ext_chnls=ext_chnls,
                    int_chnls=int_chnls,
                    corr_func=corr_func,
                    use_layernorm=use_layernorm,
                    use_syncbn=use_syncbn,
                    int_mode=int_mode,
                )
                for i in range(n_encoders)
            ],
        )

        if self.use_layerscale:
            self.layerscale = nn.Parameter(torch.ones(d_out) * layerscale_init, requires_grad=True)
        else:
            self.out_proj = nn.Linear(d_hid, d_out)

        # Gradient monitoring
        self._grad_check_counter = 0
        self._grad_check_steps = {1, 2, 5, 10}  # log at these steps
        self._grad_check_interval = 50  # then every N steps

    def _gradient_check_hook(self, grad: torch.Tensor | None) -> torch.Tensor | None:
        """Backward hook: log gradient stats flowing through motion module output.

        Args:
            grad: Gradient tensor of shape ``(B*T*H*W, d_out)`` or ``None`` if no gradient is flowing.

        Returns:
            The same gradient tensor, unmodified.
        """
        self._grad_check_counter += 1
        step = self._grad_check_counter
        if step not in self._grad_check_steps and step % self._grad_check_interval != 0:
            return grad
        return grad

    def initialize_weights(self) -> None:
        """Initialize all weights in MotionModule."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

        if self.use_layerscale:
            self.layerscale.data.fill_(self.layerscale_init)
        # else: out_proj inherits the trunc_normal_(std=0.02) weight + zero-bias init
        # from the nn.Linear loop above. Previously we zero-inited out_proj so motion module
        # started as a no-op — dropped so motion module contributes from step 1 instead of
        # warming up behind a residual shortcut.

    def forward(self, x: torch.Tensor, grid_sizes: torch.Tensor) -> torch.Tensor:
        """Compute motion features for a batch of video tokens.

        Handles both homogeneous batches (all videos share the same grid) and
        heterogeneous batches (each video has a different ``[t, h, w]``) by
        splitting, processing per video, and concatenating the results.

        Args:
            x: Flattened token tensor of shape
                ``(sum(t_i * h_i * w_i), d_in)``.
            grid_sizes: Int tensor of shape ``(B, 3)`` where each row is
                ``[t, h, w]`` for the corresponding video.

        Returns:
            Motion feature tensor of shape ``(sum(t_i * h_i * w_i), d_out)``.
        """
        # x: (total_tokens, C) where total_tokens = sum(t_i * h_i * w_i)
        # grid_sizes: (batch_size, 3) where each row is [t, h, w]
        all_same_grid = (grid_sizes == grid_sizes[0]).all()

        if all_same_grid:
            out = x
            encoder_outputs = []
            for stss_encoder in self.stss_encoders:
                out = stss_encoder(out, grid_sizes=grid_sizes)
                encoder_outputs.append(out)
            out = torch.stack(encoder_outputs, dim=0).sum(dim=0)
        else:
            num_tokens_per_video = grid_sizes.prod(dim=1).tolist()
            x_splits = x.split(num_tokens_per_video, dim=0)

            processed_videos = []
            for x_video, grid_size in zip(x_splits, grid_sizes, strict=False):
                video_out = x_video
                encoder_outputs = []
                for stss_encoder in self.stss_encoders:
                    video_out = stss_encoder(video_out, grid_sizes=grid_size.unsqueeze(0))
                    encoder_outputs.append(video_out)
                video_out = torch.stack(encoder_outputs, dim=0).sum(dim=0)
                processed_videos.append(video_out)
            out = torch.cat(processed_videos, dim=0)

        if self.use_layerscale:
            out *= self.layerscale
        else:
            out = self.out_proj(out)

        # Register backward hook for gradient monitoring (only when enabled)
        if self.gradient_check and self.training and out.requires_grad:
            out.register_hook(self._gradient_check_hook)

        return out
