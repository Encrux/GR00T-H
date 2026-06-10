from typing import Tuple

from gr00t.configs.model.gr00t_n1d6 import Gr00tN1d6Config
from gr00t.model.modules.dit import AlternateVLDiT, DiT
from gr00t.model.modules.eagle_backbone import EagleBackbone
from gr00t.model.modules.embodiment_conditioned_mlp import (
    CategorySpecificMLP,
    MultiEmbodimentActionEncoder,
)
import torch
from torch import nn
from torch.distributions import Beta
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, PreTrainedModel
from transformers.feature_extraction_utils import BatchFeature
import tree


class Gr00tN1d6ActionHead(nn.Module):
    """Action head component for flow matching diffusion policy."""

    supports_gradient_checkpointing = True

    def __init__(self, config: Gr00tN1d6Config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.input_embedding_dim = config.input_embedding_dim

        # Initialize components directly from config
        if config.use_alternate_vl_dit:
            self.model = AlternateVLDiT(
                **config.diffusion_model_cfg,
                cross_attention_dim=config.backbone_embedding_dim,
                attend_text_every_n_blocks=config.attend_text_every_n_blocks,
            )
            print("Using AlternateVLDiT for diffusion model")
        else:
            self.model = DiT(
                **config.diffusion_model_cfg, cross_attention_dim=config.backbone_embedding_dim
            )
            print("Using DiT for diffusion model")
        self.action_dim = config.max_action_dim
        self.action_horizon = config.action_horizon
        self.num_inference_timesteps = config.num_inference_timesteps

        self.state_encoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=config.max_state_dim,
            hidden_dim=self.hidden_size,
            output_dim=self.input_embedding_dim,
        )
        self.action_encoder = MultiEmbodimentActionEncoder(
            action_dim=self.action_dim,
            hidden_size=self.input_embedding_dim,
            num_embodiments=config.max_num_embodiments,
        )
        self.action_decoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=self.hidden_size,
            hidden_dim=self.hidden_size,
            output_dim=self.action_dim,
        )

        self.vlln = (
            nn.LayerNorm(config.backbone_embedding_dim) if config.use_vlln else nn.Identity()
        )

        if config.add_pos_embed:
            self.position_embedding = nn.Embedding(config.max_seq_len, self.input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        # State dropout parameters
        self.state_dropout_prob = config.state_dropout_prob
        self.state_dropout_prob_per_embodiment = getattr(
            config, "state_dropout_prob_per_embodiment", None
        )

        # Build per-embodiment dropout lookup buffer
        if self.state_dropout_prob_per_embodiment:
            from .processing_gr00t_n1d6 import EMBODIMENT_TAG_TO_PROJECTOR_INDEX

            dropout_buf = torch.zeros(config.max_num_embodiments)
            for tag, prob in self.state_dropout_prob_per_embodiment.items():
                if tag in EMBODIMENT_TAG_TO_PROJECTOR_INDEX:
                    dropout_buf[EMBODIMENT_TAG_TO_PROJECTOR_INDEX[tag]] = prob
            self.register_buffer("dropout_prob_by_embodiment", dropout_buf)

        has_any_dropout = self.state_dropout_prob > 0 or self.state_dropout_prob_per_embodiment
        self.mask_token = (
            nn.Parameter(0.02 * torch.randn(1, 1, self.input_embedding_dim))
            if has_any_dropout
            else None
        )

        # State noise parameters
        self.state_additive_noise_scale = config.state_additive_noise_scale

        self.beta_dist = Beta(config.noise_beta_alpha, config.noise_beta_beta)
        self.num_timestep_buckets = config.num_timestep_buckets

        # BEAST: load the B-spline tokenizer's BSpline class (we don't keep the
        # outer processor — its __init__ hardcodes device='cuda' and we want a
        # CPU-friendly construction with our own K / num_dof / degree).
        self.use_bspline = getattr(config, "use_bspline", False)
        if self.use_bspline:
            from transformers import AutoProcessor
            beast_proc = AutoProcessor.from_pretrained(
                config.beast_processor_id, trust_remote_code=True
            )
            BSplineClass = type(beast_proc.bsp)
            self.bspline_init_cond_order = getattr(config, "bspline_init_cond_order", 0)
            assert self.bspline_init_cond_order in (0, 1, 2), (
                f"bspline_init_cond_order must be 0 (free), 1 (clamp start pos) or "
                f"2 (clamp start pos+vel); got {self.bspline_init_cond_order}"
            )
            self.bspline = BSplineClass(
                num_basis=config.bspline_num_basis,
                num_dof=config.max_action_dim,
                degree=config.bspline_degree,
                init_cond_order=self.bspline_init_cond_order,
            )
            # Action sample times in spline-phase. With a start clamp
            # (init_cond_order>0) the boundary control point sits at phase 0
            # (= the current state); the T action samples must live at phases
            # (0, 1] so no predicted action is forced onto the clamp. Without a
            # clamp, the usual [0, 1] grid (action at phase 0) is fine.
            if self.bspline_init_cond_order > 0:
                t_grid = torch.linspace(
                    0, 1, config.action_horizon + 1, dtype=torch.float32
                )[1:]
            else:
                t_grid = torch.linspace(
                    0, 1, config.action_horizon, dtype=torch.float32
                )
            self.register_buffer("t_grid", t_grid)
            # Per-channel std for normalizing control points to ~unit variance
            # before flow matching. Pretrained DiT + action_decoder + the unit-
            # variance noise schedule all assume O(1) targets; unnormalized CPs
            # are ~5-10× larger per channel, which breaks SNR per timestep.
            # Estimated as a running mean over the first `cp_std_warmup_steps`
            # training batches, then frozen (saved in state_dict).
            self.cp_std_warmup_steps = 50
            self.register_buffer(
                "cp_std",
                torch.ones(config.max_action_dim, dtype=torch.float32),
            )
            self.register_buffer(
                "cp_std_init_steps",
                torch.zeros(1, dtype=torch.long),
            )

        self.set_trainable_parameters(
            config.tune_projector, config.tune_diffusion_model, config.tune_vlln
        )

    def set_trainable_parameters(
        self, tune_projector: bool, tune_diffusion_model: bool, tune_vlln: bool
    ):
        self.tune_projector = tune_projector
        self.tune_diffusion_model = tune_diffusion_model
        self.tune_vlln = tune_vlln
        for p in self.parameters():
            p.requires_grad = True
        if not tune_projector:
            self.state_encoder.requires_grad_(False)
            self.action_encoder.requires_grad_(False)
            self.action_decoder.requires_grad_(False)
            if self.config.add_pos_embed:
                self.position_embedding.requires_grad_(False)
            if self.mask_token is not None:
                self.mask_token.requires_grad_(False)
        if not tune_diffusion_model:
            self.model.requires_grad_(False)
        if not tune_vlln:
            self.vlln.requires_grad_(False)
        print(f"Tune action head projector: {self.tune_projector}")
        print(f"Tune action head diffusion model: {self.tune_diffusion_model}")
        print(f"Tune action head vlln: {self.tune_vlln}")
        # Check if any parameters are still trainable. If not, print a warning.
        if not tune_projector and not tune_diffusion_model and not tune_vlln:
            for name, p in self.named_parameters():
                if p.requires_grad:
                    print(f"Action head trainable parameter: {name}")
        if not any(p.requires_grad for p in self.parameters()):
            print("Warning: No action head trainable parameters found.")

    def set_frozen_modules_to_eval_mode(self):
        """
        Huggingface will call model.train() at each training_step. To ensure
        the expected behaviors for modules like dropout, batchnorm, etc., we
        need to call model.eval() for the frozen modules.
        """
        if self.training:
            if not self.tune_projector:
                self.state_encoder.eval()
                self.action_encoder.eval()
                self.action_decoder.eval()
                if self.config.add_pos_embed:
                    self.position_embedding.eval()
            if not self.tune_diffusion_model:
                self.model.eval()

    def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        sample = (1 - sample) * self.config.noise_s
        return sample

    def _bspline_boundary_conditions(self, state: torch.Tensor | None) -> dict:
        """BSpline start-clamp kwargs, shared by training encode, the R4 loss
        decode and inference decode so train/inference cannot diverge.

        order 1: pin the chunk-start position to the current state s_t.
        order 2: additionally pin the start velocity, estimated from a 2-step
        state history (modality config state delta_indices=[-1, 0]).

        Velocity is in spline-phase units: consecutive states are one env step
        apart, which equals the phase gap t_grid[0] between the clamp (phase 0)
        and the first action sample — the same dt beast.py's own auto-extract
        uses. State-normalizer mean offsets cancel in the difference, so the
        velocity is only sensitive to the (near-identical) std scaling.
        """
        order = self.bspline_init_cond_order
        if order == 0:
            return {}
        # Order 1 keeps the shipped behavior bit-for-bit, including the silent
        # no-clamp fallback when no state is provided at inference.
        if state is None:
            assert order < 2, (
                "bspline_init_cond_order=2 requires the raw state at decode "
                "time; got None"
            )
            return {}
        bc = {"init_pos": state[:, -1, :].to(torch.float32)}
        if order >= 2:
            # Never fall back to beast.py's default init_vel (first diff of the
            # GT chunk, beast.py learn_mp_params_from_trajs) — that quantity
            # does not exist at inference and would train a clamp the policy
            # can't reproduce.
            assert state.shape[1] >= 2, (
                "bspline_init_cond_order=2 requires a 2-step state history "
                "(state delta_indices=[-1, 0]); got state shape "
                f"{tuple(state.shape)}"
            )
            s_prev = state[:, -2, :].to(torch.float32)
            bc["init_vel"] = (bc["init_pos"] - s_prev) / float(self.t_grid[0])
        return bc

    def process_backbone_output(self, backbone_output: BatchFeature) -> BatchFeature:
        backbone_features = backbone_output["backbone_features"]
        backbone_features = self.vlln(backbone_features)
        backbone_output["backbone_features"] = backbone_features
        return backbone_output

    def forward(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:
        """
        Forward pass through the action head.

        Args:
            backbone_output: Output from the backbone model containing:
                - backbone_features: [B, seq_len, backbone_embedding_dim]
                - backbone_attention_mask: [B, seq_len]
            action_input: Input containing:
                - state: [B, state_dim]
                - action: [B, action_horizon, action_dim] (during training)
                - embodiment_id: [B] (embodiment IDs)
                - action_mask: [B, action_horizon, action_dim]

        Returns:
            BatchFeature containing:
                - loss: action prediction loss
        """
        # Set frozen modules to eval
        self.set_frozen_modules_to_eval_mode()

        backbone_output = self.process_backbone_output(backbone_output)

        # Get vision and language embeddings.
        vl_embeds = backbone_output.backbone_features
        device = vl_embeds.device

        # Get embodiment ID.
        embodiment_id = action_input.embodiment_id

        # Per-embodiment state dropout: zero state BEFORE encoding.
        if self.state_dropout_prob_per_embodiment and hasattr(self, "dropout_prob_by_embodiment"):
            dropout_probs = self.dropout_prob_by_embodiment[embodiment_id]  # (B,)
            if self.training:
                do_dropout = (
                    torch.rand(action_input.state.shape[0], device=action_input.state.device)
                    < dropout_probs
                )
            else:
                do_dropout = dropout_probs > 0.999  # deterministic at inference
            do_dropout = do_dropout[:, None, None].to(dtype=action_input.state.dtype)
            action_input.state = action_input.state * (1 - do_dropout)

        # Embed state.
        state_features = self.state_encoder(action_input.state, embodiment_id)

        # Global state dropout: replace encoded features with learned mask_token.
        if self.mask_token is not None and self.state_dropout_prob > 0:
            if self.training:
                do_dropout = (
                    torch.rand(state_features.shape[0], device=state_features.device)
                    < self.state_dropout_prob
                )
                do_dropout = do_dropout[:, None, None].to(dtype=state_features.dtype)
                state_features = state_features * (1 - do_dropout) + self.mask_token * do_dropout

        # Add Gaussian noise to state features.
        if self.training and self.state_additive_noise_scale > 0:
            print(
                f"Adding Gaussian noise to state features with scale {self.state_additive_noise_scale}"
            )
            noise = torch.randn_like(state_features) * self.state_additive_noise_scale
            state_features = state_features + noise

        # Embed noised action trajectory.
        actions = action_input.action                                # [B, T, D]
        if self.use_bspline:
            # Encode GT trajectory → K control points. No gradient needed for
            # the target side; this is a fixed-per-batch supervised target.
            # torch.linalg.solve (used inside BSpline) doesn't support bf16,
            # but model.to(bf16) auto-casts buffers + child-module params.
            # So force float32 on: the BSpline's internal state, the t_grid,
            # and the trajectory we encode. Cast result back to bf16 for the
            # downstream flow-matching loop.
            orig_dtype = actions.dtype
            B = actions.shape[0]
            self.bspline.float()                                     # idempotent
            times_b = (
                self.t_grid.to(actions.device, dtype=torch.float32)
                .expand(B, -1)
            )                                                        # [B, T] f32
            # Start-clamp (init_cond_order>0): anchor the spline start to the
            # CURRENT robot state, so consecutive chunks join continuously. We
            # override the BEAST default (which clamps to the chunk's first
            # action) because at inference the first action is unknown — only
            # the current state is. State-normalized ≈ action-normalized for
            # absolute-action embodiments (sofa_ll: action[t]=state[t+1]).
            # Order 2 additionally clamps the start velocity (2-step history).
            bc_kwargs = self._bspline_boundary_conditions(action_input.state)
            # Disable autocast — trainer wraps the whole forward in bf16
            # autocast, which intercepts matmuls/einsums inside the BSpline
            # solve and casts them down to bf16 even though our inputs are
            # float32. torch.linalg.solve doesn't support bf16.
            with torch.no_grad(), torch.amp.autocast(
                device_type=actions.device.type, enabled=False
            ):
                params_dict = self.bspline.learn_mp_params_from_trajs(
                    times_b, actions.to(torch.float32), **bc_kwargs
                )
                # BSpline returns params flat in (D, K)-major order — each
                # contiguous block of K values is one DOF's basis. View as
                # (B, D, K) then transpose to put K on the temporal axis the
                # DiT expects. .view(B, K, D) directly would scramble DOFs
                # across tokens and break per-token action_decoder weights +
                # position embeddings.
                K = self.config.bspline_num_basis
                actions_f32 = (
                    params_dict["params"]
                    .view(B, -1, K)
                    .transpose(-1, -2)
                    .contiguous()
                )                                                    # [B, K, D] fp32

                # Update / freeze per-channel CP std (running mean over the
                # first cp_std_warmup_steps batches).
                if self.training:
                    n_done = int(self.cp_std_init_steps.item())
                    if n_done < self.cp_std_warmup_steps:
                        batch_std = (
                            actions_f32.std(dim=(0, 1)).clamp(min=1e-2)
                        )                                            # [D]
                        if n_done == 0:
                            self.cp_std.copy_(batch_std)
                        else:
                            self.cp_std.mul_(n_done / (n_done + 1)).add_(
                                batch_std / (n_done + 1)
                            )
                        self.cp_std_init_steps.add_(1)

                # Normalize CPs to ~unit variance per channel before flow loss.
                cp_std_view = self.cp_std.view(1, 1, -1)             # [1, 1, D]
                actions = (actions_f32 / cp_std_view).to(orig_dtype)
        noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
        t = self.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)
        t = t[:, None, None]  # shape (B,1,1) for broadcast

        noisy_trajectory = (1 - t) * noise + t * actions
        velocity = actions - noise

        # Convert (continuous) t -> discrete if needed
        t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()
        action_features = self.action_encoder(noisy_trajectory, t_discretized, embodiment_id)

        # Maybe add position embedding.
        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        # Join vision, language, state and action embedding along sequence dimension.
        sa_embs = torch.cat((state_features, action_features), dim=1)
        vl_attn_mask = backbone_output.backbone_attention_mask

        if self.config.use_alternate_vl_dit:
            image_mask = backbone_output.image_mask
            backbone_attention_mask = backbone_output.backbone_attention_mask
            model_output, _ = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embeds,
                encoder_attention_mask=vl_attn_mask,
                timestep=t_discretized,
                return_all_hidden_states=True,
                image_mask=image_mask,
                backbone_attention_mask=backbone_attention_mask,
            )
        else:
            model_output, _ = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embeds,
                encoder_attention_mask=vl_attn_mask,
                timestep=t_discretized,
                return_all_hidden_states=True,
            )

        pred = self.action_decoder(model_output, embodiment_id)
        pred_actions = pred[:, -actions.shape[1] :]

        # Loss.
        if self.use_bspline:
            # R4: compute loss in trajectory space, not CP space. Decoding
            # is linear, so MSE(B·pred_cp, B·target_cp) just re-weights the
            # CP-space MSE by basis structure — directly minimizes the metric
            # we evaluate (per-step action MAE) instead of an indirect proxy.
            # Also: action_input.action_mask is [B, T, D] in trajectory space,
            # which matches the decoded shape — no expand/slice gymnastics.
            B_ = pred_actions.shape[0]
            cp_std_view = self.cp_std.view(1, 1, -1)                     # [1, 1, D]
            # Denormalize velocities back to encoded CP scale before decode,
            # mirroring inference. Cast to fp32 for the BSpline matmul.
            pred_cp_real = (pred_actions.float() * cp_std_view)          # [B, K, D] f32
            target_cp_real = (velocity.float() * cp_std_view)            # [B, K, D] f32
            # Flatten (B, K, D) → (B, D, K) → (B, D*K) to match BSpline's
            # native (D, K)-major flat layout.
            pred_flat = pred_cp_real.transpose(-1, -2).contiguous().reshape(B_, -1)
            target_flat = target_cp_real.transpose(-1, -2).contiguous().reshape(B_, -1)
            times_traj = self.t_grid.to(
                pred_actions.device, dtype=torch.float32
            ).expand(B_, -1)
            # Start-clamp: pass the SAME boundary conditions to both decodes.
            # Decoding is affine, decode(v)=C(init)+L(v); identical init → C
            # cancels in MSE(C+L(pred), C+L(target)), so the loss is the
            # basis-weighted comparison exactly as in the no-clamp case
            # (verified numerically; holds for order 2 as well — both boundary
            # CPs live in C).
            bc_loss = bc_kwargs
            self.bspline.float()
            with torch.amp.autocast(device_type=pred_actions.device.type, enabled=False):
                pred_traj = self.bspline.get_traj_pos(times=times_traj, params=pred_flat, **bc_loss)
                target_traj = self.bspline.get_traj_pos(times=times_traj, params=target_flat, **bc_loss)
            pred_traj = pred_traj.to(pred_actions.dtype)                 # [B, T, D]
            target_traj = target_traj.to(pred_actions.dtype)             # [B, T, D]
            action_mask = action_input.action_mask                       # [B, T, D]
            action_loss = (
                F.mse_loss(pred_traj, target_traj, reduction="none") * action_mask
            )
        else:
            action_mask = action_input.action_mask
            action_loss = F.mse_loss(pred_actions, velocity, reduction="none") * action_mask
        loss = action_loss.sum() / (action_mask.sum() + 1e-6)

        return {
            "loss": loss,
            "action_loss": action_loss,
            "action_mask": action_mask,
            "backbone_features": vl_embeds,
            "state_features": state_features,
        }

    def _encode_features(
        self, backbone_output: BatchFeature, action_input: BatchFeature
    ) -> BatchFeature:
        """
        Encode features for the action head.

        Args:
            backbone_output: Output from the backbone model containing:
                - backbone_features: [B, seq_len, backbone_embedding_dim]
                - backbone_attention_mask: [B, seq_len]
            action_input: Input containing:
                - state: [B, state_dim]
                - embodiment_id: [B] (embodiment IDs)

        Returns:
            BatchFeature containing:
                - backbone_features: [B, seq_len, backbone_embedding_dim]
                - state_features: [B, state_horizon, input_embedding_dim]
        """
        backbone_output = self.process_backbone_output(backbone_output)

        # Get vision and language embeddings.
        vl_embeds = backbone_output.backbone_features
        embodiment_id = action_input.embodiment_id

        state = action_input.state

        # Per-embodiment state dropout: zero state before encoding (deterministic at inference)
        if self.state_dropout_prob_per_embodiment and hasattr(self, "dropout_prob_by_embodiment"):
            dropout_probs = self.dropout_prob_by_embodiment[embodiment_id]
            do_dropout = (dropout_probs > 0.999)[:, None, None].to(dtype=state.dtype)
            state = state * (1 - do_dropout)

        # Embed state.
        state_features = self.state_encoder(state, embodiment_id)

        return BatchFeature(data={"backbone_features": vl_embeds, "state_features": state_features})

    @torch.no_grad()
    def get_action_with_features(
        self,
        backbone_features: torch.Tensor,
        state_features: torch.Tensor,
        embodiment_id: torch.Tensor,
        backbone_output: BatchFeature,
        raw_state: torch.Tensor = None,
    ) -> BatchFeature:
        """
        Generate actions using the flow matching diffusion process.

        Args:
            backbone_features: [B, seq_len, backbone_embedding_dim]
            state_features: [B, state_horizon, input_embedding_dim]
            embodiment_id: [B] (embodiment IDs)
            backbone_output: Output from the backbone model
        """
        vl_embeds = backbone_features

        # Set initial actions as the sampled noise.
        # When use_bspline, the DiT operates on K control-point tokens (one per
        # spline basis) instead of T raw timesteps; we decode to a T-step
        # trajectory after Euler integration.
        batch_size = vl_embeds.shape[0]
        device = vl_embeds.device
        horizon = (
            self.config.bspline_num_basis
            if self.use_bspline
            else self.config.action_horizon
        )
        actions = torch.randn(
            size=(batch_size, horizon, self.action_dim),
            dtype=vl_embeds.dtype,
            device=device,
        )

        dt = 1.0 / self.num_inference_timesteps

        # Run denoising steps.
        for t in range(self.num_inference_timesteps):
            t_cont = t / float(self.num_inference_timesteps)  # e.g. goes 0, 1/N, 2/N, ...
            t_discretized = int(t_cont * self.num_timestep_buckets)

            # Embed noised action trajectory.
            timesteps_tensor = torch.full(
                size=(batch_size,), fill_value=t_discretized, device=device
            )
            action_features = self.action_encoder(actions, timesteps_tensor, embodiment_id)
            # Add position embedding.
            if self.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs

            # Join vision, language, state and action embedding along sequence dimension.
            sa_embs = torch.cat((state_features, action_features), dim=1)

            # Run model forward.
            if self.config.use_alternate_vl_dit:
                model_output = self.model(
                    hidden_states=sa_embs,
                    encoder_hidden_states=vl_embeds,
                    timestep=timesteps_tensor,
                    image_mask=backbone_output.image_mask,
                    backbone_attention_mask=backbone_output.backbone_attention_mask,
                )
            else:
                model_output = self.model(
                    hidden_states=sa_embs,
                    encoder_hidden_states=vl_embeds,
                    timestep=timesteps_tensor,
                )
            pred = self.action_decoder(model_output, embodiment_id)

            pred_velocity = pred[:, -horizon:]

            # Update actions using euler integration.
            actions = actions + dt * pred_velocity

        # If BEAST, decode predicted control points to a T-step trajectory.
        # Same dtype dance as in forward() — keep BSpline state + inputs in
        # float32, cast result back to bf16 for downstream tensors.
        if self.use_bspline:
            B = actions.shape[0]
            self.bspline.float()
            times_b = (
                self.t_grid.to(actions.device, dtype=torch.float32)
                .expand(B, -1)
            )                                                             # [B, T] f32
            # Denormalize predicted CPs back to original (encoded) scale before
            # BSpline decode — mirrors the per-channel division in forward().
            cp_std_view = self.cp_std.view(1, 1, -1).to(
                device=actions.device, dtype=actions.dtype
            )                                                             # [1, 1, D]
            actions = actions * cp_std_view
            # Mirror the forward()-side layout: model emits (B, K, D); BSpline
            # decoder expects flat in (D, K)-major. Transpose before flatten.
            cp_flat = (
                actions.transpose(-1, -2).contiguous().reshape(B, -1).to(torch.float32)
            )                                                              # [B, D*K] f32
            # Start-clamp: anchor the decoded trajectory to the current robot
            # state (same as training). raw_state is [B, state_horizon, D].
            # Order 1 falls back to no clamp if raw_state is missing (shipped
            # behavior); order 2 asserts — a malformed spline otherwise.
            bc_infer = self._bspline_boundary_conditions(raw_state)
            with torch.amp.autocast(device_type=actions.device.type, enabled=False):
                traj = self.bspline.get_traj_pos(times=times_b, params=cp_flat, **bc_infer)
            actions = traj.to(dtype=vl_embeds.dtype)                      # [B, T, D]
        return BatchFeature(
            data={
                "action_pred": actions,
                "backbone_features": vl_embeds,
                "state_features": state_features,
            }
        )

    @torch.no_grad()
    def get_action(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:
        """
        Generate actions using the flow matching diffusion process.

        Args:
            backbone_output: Output from the backbone model containing:
                - backbone_features: [B, seq_len, backbone_embedding_dim]
                - backbone_attention_mask: [B, seq_len]
            action_input: Input containing:
                - state: [B, state_dim]
                - embodiment_id: [B] (embodiment IDs)

        Returns:
            BatchFeature containing:
                - action_pred: [B, action_horizon, action_dim] predicted actions
        """
        features = self._encode_features(backbone_output, action_input)
        return self.get_action_with_features(
            backbone_features=features.backbone_features,
            state_features=features.state_features,
            embodiment_id=action_input.embodiment_id,
            backbone_output=backbone_output,
            raw_state=getattr(action_input, "state", None),
        )

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype

    def prepare_input(self, batch: dict) -> BatchFeature:
        """Prepare input batch for the action head."""
        return BatchFeature(data=batch)


def get_backbone_cls(config: Gr00tN1d6Config):
    if "NVEagle" in config.model_name or "nvidia/Eagle" in config.model_name:
        return EagleBackbone
    else:
        raise ValueError(f"Unsupported model name: {config.model_name}")


class Gr00tN1d6(PreTrainedModel):
    """Gr00tN1d6: Vision-Language-Action model with backbone."""

    config_class = Gr00tN1d6Config
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: Gr00tN1d6Config,
        transformers_loading_kwargs: dict = {"trust_remote_code": True},
    ):
        """
        Initialize Gr00tN1d6 model.

        Args:
            config: Model configuration
            transformers_loading_kwargs: Dict with transformers loading parameters:
                - transformers_trust_remote_code: Whether to trust remote code when loading from HF Hub
                - transformers_local_files_only: Whether to only use local files
                - model_revision: Specific model revision to use
                - transformers_cache_dir: Directory to cache downloaded models
                - transformers_access_token: HuggingFace access token for gated models

        Note: During training, transformers parameters are passed from training config.
              During inference (e.g., from_pretrained), defaults are used.
        """
        super().__init__(config)
        self.config = config

        backbone_cls = get_backbone_cls(config)
        self.backbone = backbone_cls(
            model_name=config.model_name,
            tune_llm=config.tune_llm,
            tune_visual=config.tune_visual,
            select_layer=config.select_layer,
            reproject_vision=config.reproject_vision,
            use_flash_attention=config.use_flash_attention,
            load_bf16=config.load_bf16,
            tune_top_llm_layers=config.tune_top_llm_layers,
            trainable_params_fp32=config.backbone_trainable_params_fp32,
            transformers_loading_kwargs=transformers_loading_kwargs,
        )

        # Initialize action head
        self.action_head = Gr00tN1d6ActionHead(config)
        from .processing_gr00t_n1d6 import Gr00tN1d6DataCollator

        self.collator = Gr00tN1d6DataCollator(
            model_name=config.model_name,
            model_type=config.backbone_model_type,
            transformers_loading_kwargs=transformers_loading_kwargs,
        )

    def prepare_input(self, inputs: dict) -> Tuple[BatchFeature, BatchFeature]:
        """Prepare inputs for backbone and action head."""

        # NOTE -- currently the eval code doesn't use collator, so we need to add it here
        # this should ideally be fixed upstream
        if "vlm_content" in inputs:
            # Fix for n_envs > 1: Process all environments' VLM content, not just the first
            vlm_content_list = inputs["vlm_content"]
            # Ensure vlm_content_list is always a list for consistent processing
            if not isinstance(vlm_content_list, list):
                vlm_content_list = [vlm_content_list]

            # Process all VLM contents through the collator
            prep = self.collator([{"vlm_content": vlm} for vlm in vlm_content_list])["inputs"]
            inputs.pop("vlm_content")
            inputs.update(prep)

        backbone_inputs = self.backbone.prepare_input(inputs)
        action_inputs = self.action_head.prepare_input(inputs)

        # Move to device and dtype
        def to_device_with_dtype(x):
            if torch.is_floating_point(x):
                return x.to(self.device, dtype=self.dtype)
            else:
                return x.to(self.device)

        backbone_inputs = tree.map_structure(to_device_with_dtype, backbone_inputs)
        action_inputs = tree.map_structure(to_device_with_dtype, action_inputs)

        return backbone_inputs, action_inputs

    def forward(self, inputs: dict) -> BatchFeature:
        """
        Forward pass through the complete model.

        Args:
            inputs: Dictionary containing:
                - Eagle inputs (prefixed with 'eagle_')
                - Action inputs (state, action, embodiment_id, etc.)

        Returns:
            BatchFeature containing loss and other outputs
        """
        # Prepare inputs for backbone and action head
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        backbone_outputs = self.backbone(backbone_inputs)
        action_outputs = self.action_head(backbone_outputs, action_inputs)

        return action_outputs

    def get_action(self, inputs: dict) -> BatchFeature:
        """
        Generate actions using the complete model.
        """
        # Prepare inputs for backbone and action head
        backbone_inputs, action_inputs = self.prepare_input(inputs)

        # Forward through backbone
        backbone_outputs = self.backbone(backbone_inputs)
        action_outputs = self.action_head.get_action(backbone_outputs, action_inputs)

        return action_outputs

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype


# Register the model with HuggingFace
AutoConfig.register("Gr00tN1d6", Gr00tN1d6Config)
AutoModel.register(Gr00tN1d6Config, Gr00tN1d6)
