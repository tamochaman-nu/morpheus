from typing import Optional

import torch
from einops import rearrange
from torch.nn import functional as F

from morpheus.utils.data_utils import Frame
from morpheus.utils.mesh_utils import MeshWarp


def compute_attn_allkv(
    attn, query, key, value, video_length, attention_mask, extra_attn_mult=None, self_attn_value=1.0, chunk_size: int = 1024
):
    """
    Takes queries from the frame to be generated and does attention between those and keys/values
    drawn from all frames, both ref and non-ref.
    query: (BxF_non_ref, D_q, C)
    key / value: (BxF_total, D_kv, C)
    """

    d_query = query.shape[1]

    # The new matrices must be catted along the f-dimension
    if extra_attn_mult is not None:
        extra_attn_mult = rearrange(extra_attn_mult, "r k q -> (r k) q")
        self_attn_mult = torch.full(
            (d_query, d_query), self_attn_value, device=extra_attn_mult.device
        )
        extra_attn_mult = torch.cat([extra_attn_mult, self_attn_mult], dim=0)

    # Flatten K and V
    key = rearrange(key, "(b f) d c -> b (f d) c", f=video_length)
    value = rearrange(value, "(b f) d c -> b (f d) c", f=video_length)

    key = attn.head_to_batch_dim(key)
    value = attn.head_to_batch_dim(value)
    
    scale = attn.scale
    dtype = query.dtype
    
    if attn.upcast_attention:
        query = query.float()
        key = key.float()
        
    num_queries = query.shape[1]
    out_hidden_states = torch.zeros(query.shape[0], query.shape[1], value.shape[2], dtype=dtype, device=query.device)

    # Apply slicing to prevent materializing the full (query_len x key_len) attention matrix
    for i in range(0, num_queries, chunk_size):
        query_chunk = query[:, i : i + chunk_size, :]
        
        if attention_mask is None:
            baddbmm_input = torch.empty(
                query_chunk.shape[0], query_chunk.shape[1], key.shape[1], dtype=query_chunk.dtype, device=query_chunk.device
            )
            beta = 0
        else:
            baddbmm_input = attention_mask[:, i : i + chunk_size, :]
            beta = 1

        attention_scores = torch.baddbmm(
            baddbmm_input,
            query_chunk,
            key.transpose(-1, -2),
            beta=beta,
            alpha=scale,
        )
        del baddbmm_input

        if attn.upcast_softmax:
            attention_scores = attention_scores.float()

        if extra_attn_mult is not None:
            extra_attn_shift = torch.log(extra_attn_mult)
            shift_chunk = extra_attn_shift.transpose(0, 1)[i : i + chunk_size, :]
            attention_scores = attention_scores + shift_chunk[None, ...]

        attention_probs = attention_scores.softmax(dim=-1)
        del attention_scores

        attention_probs = attention_probs.to(value.dtype)
        
        # Multiply by V for the current chunk
        out_hidden_states[:, i : i + chunk_size, :] = torch.bmm(attention_probs, value)
        del attention_probs

    return out_hidden_states


def get_attention_scores(
    query: torch.Tensor,
    key: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    upcast_attention: bool = False,
    upcast_softmax: bool = False,
    scale: float = 1.0,
    extra_attn_mult=None,
    chunk_size: int = 1024,
) -> torch.Tensor:
    # This is now only a fallback or utility, most logic moved to compute_attn_allkv for memory efficiency
    dtype = query.dtype
    if upcast_attention:
        query = query.float()
        key = key.float()

    num_queries = query.shape[1]
    all_attention_probs = []

    for i in range(0, num_queries, chunk_size):
        query_chunk = query[:, i : i + chunk_size, :]
        if attention_mask is None:
            baddbmm_input = torch.empty(query_chunk.shape[0], query_chunk.shape[1], key.shape[1], dtype=query_chunk.dtype, device=query_chunk.device)
            beta = 0
        else:
            baddbmm_input = attention_mask[:, i : i + chunk_size, :]
            beta = 1

        attention_scores = torch.baddbmm(baddbmm_input, query_chunk, key.transpose(-1, -2), beta=beta, alpha=scale)
        if upcast_softmax: attention_scores = attention_scores.float()
        if extra_attn_mult is not None:
            extra_attn_shift = torch.log(extra_attn_mult)
            shift_chunk = extra_attn_shift.transpose(0, 1)[i : i + chunk_size, :]
            attention_scores = attention_scores + shift_chunk[None, ...]
        attention_probs = attention_scores.softmax(dim=-1).to(dtype)
        all_attention_probs.append(attention_probs)

    return torch.cat(all_attention_probs, dim=1)


class AutoregressiveCrossAttentionProcessor:
    def __init__(
        self,
        unet_num_chunks=1,
        extra_attention_maps=None,
        ref_mixing_amount=0.0,
        self_attn_shift=1.0,
        pooling_mode="max",
        xattn_min_downscale_factor: int = 1,
        xattn_max_downscale_factor: int = 8,
        xattn_in_encoder=True,
        xattn_in_decoder=True,
    ):
        self.unet_num_chunks = unet_num_chunks
        self.ref_latents = None
        self.extra_attention_maps = extra_attention_maps
        self.ref_mixing_amount = ref_mixing_amount
        self.self_attn_shift = self_attn_shift
        self.pooling_mode = pooling_mode
        self.xattn_min_downscale_factor = xattn_min_downscale_factor
        self.xattn_max_downscale_factor = xattn_max_downscale_factor

        self.xattn_in_encoder = xattn_in_encoder
        self.xattn_in_decoder = xattn_in_decoder

    def set_reference_latents(self, ref_latents):
        self.ref_latents = ref_latents

    @property
    def num_ref_latents(self):
        return len(self.ref_latents) if self.ref_latents is not None else 0

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        scale=1.0,
    ):

        residual = hidden_states

        args = ()

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query_pre_reshape = attn.to_q(hidden_states, *args)

        is_cross_attention_with_prompt = encoder_hidden_states is not None
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        # Compute the key and value tensors
        key = attn.to_k(encoder_hidden_states, *args)
        value = attn.to_v(encoder_hidden_states, *args)

        query = attn.head_to_batch_dim(query_pre_reshape)

        if not is_cross_attention_with_prompt:
            ################## Perform self attention (with slicing to prevent OOM)
            key_self = attn.head_to_batch_dim(key)
            value_self = attn.head_to_batch_dim(value)
            
            # Sliced self-attention
            chunk_size = 1024
            num_queries = query.shape[1]
            hidden_states_self = torch.zeros(query.shape[0], query.shape[1], value_self.shape[2], dtype=query.dtype, device=query.device)
            
            for i in range(0, num_queries, chunk_size):
                query_chunk = query[:, i : i + chunk_size, :]
                mask_chunk = attention_mask[:, i : i + chunk_size, :] if attention_mask is not None else None
                
                # Use simplified slicing for self-attention
                probs_chunk = get_attention_scores(
                    query_chunk, key_self, mask_chunk, 
                    upcast_attention=attn.upcast_attention, 
                    upcast_softmax=attn.upcast_softmax, 
                    scale=attn.scale
                )
                hidden_states_self[:, i : i + chunk_size, :] = torch.bmm(probs_chunk, value_self)
                del probs_chunk
            #######################################

            video_length = key.size(0) // self.unet_num_chunks
            num_tokens = hidden_states_self.shape[-2]

            # The deeper layers of the UNet operate at lower resolution. Our reference latents
            # are at the original stable diffusion latent resolution (which is itself (H/8, W/8) in
            # terms of the original image resolution). So we need to downscale our reference latents
            # to match the current layer's resolution.
            # For non-square images, we use the aspect ratio from the attention maps.
            if self.extra_attention_maps is not None:
                _, ref_h, ref_w, tgt_h, tgt_w = self.extra_attention_maps.shape
                aspect_ratio = tgt_w / tgt_h
                latent_height = int(round((num_tokens / aspect_ratio) ** 0.5))
                # latent_width = int(round(num_tokens / latent_height))
                ref_latent_height = ref_h
            else:
                # Fallback to square if no maps provided
                latent_height = int(round(num_tokens**0.5))
                ref_latent_height = int(round(self.ref_latents[0].shape[-2] ** 0.5)) if self.ref_latents is not None else 0

            downscale_factor = ref_latent_height // latent_height

            # Only do xattn for certain layers
            if (
                downscale_factor < self.xattn_min_downscale_factor
                or downscale_factor > self.xattn_max_downscale_factor
            ):
                do_xattn_with_ref = False
            else:
                do_xattn_with_ref = True

            # As well as downscaling the reference latents, we also need to downscale the extra
            # attention maps (if present). We will do this by max-pooling each of them.
            if self.extra_attention_maps is not None:
                extra_attention_maps = independent_max_pooling(
                    self.extra_attention_maps,
                    downscale_factor,
                    downscale_factor,
                    mode=self.pooling_mode,
                )

                # Now reshape to match key & query dims by flattening spatial axes (H, W) and (P, Q) to a
                # single dimension each.
                extra_attention_maps_flat = rearrange(
                    extra_attention_maps, "b h w p q -> b (h w) (p q)"
                )

            else:
                extra_attention_maps_flat = None

            # Reshape the hidden states from the self-attn ready for mixing in cross-attention:
            hidden_states = attn.batch_to_head_dim(hidden_states_self)  # (B, H*W, C)
            hidden_states = rearrange(hidden_states, "(b f) d c -> b f d c", b=self.unet_num_chunks)

            if do_xattn_with_ref:
                # We want to do attention betwen the non-ref frames (queries) and all frames, reference and non-reference (keys/values).
                # This allows us to mix in information from the reference frames into the non-reference frames.
                # This involves selecting the ref frames as the keys and values for the current frame, and the target frame as the query.

                # Reshape to (b, f, d, c) & pick out the non-ref idxs only to obtain the queries:
                query_for_xattn = rearrange(
                    query_pre_reshape, "(b f) d c -> b f d c", f=video_length
                )
                query_for_xattn = query_for_xattn[:, self.num_ref_latents :]
                # Now reshape back to (b * f, d, c)
                query_for_xattn = rearrange(query_for_xattn, "b f d c -> (b f) d c")
                # Finally move batch to head dim ready to compute attn
                query_for_xattn = attn.head_to_batch_dim(query_for_xattn)

                # Do xattn
                attn_mult_for_ref = (
                    extra_attention_maps_flat if extra_attention_maps_flat is not None else None
                )

                mean_hidden_states_refs = compute_attn_allkv(
                    attn,
                    query_for_xattn,
                    key,
                    value,
                    video_length,
                    attention_mask,
                    extra_attn_mult=attn_mult_for_ref,
                    self_attn_value=self.self_attn_shift,
                )
                mean_hidden_states_refs = attn.batch_to_head_dim(
                    mean_hidden_states_refs
                )  # (B, H*W, C)
                # Reshape things to (b f d c) for the mean_hidden_states_refs
                mean_hidden_states_refs = rearrange(
                    mean_hidden_states_refs, "(b f) d c -> b f d c", b=self.unet_num_chunks
                )

                # hidden_states_self's f-dimension will equal video_length (i.e. total num frames),
                # whereas mean_hidden_states_refs' f-dimension will be the number of non-ref images.
                # We want to mix some of mean_hidden_states_refs into the non-ref hidden states,
                # while leaving the ref hidden states untouched
                hidden_states[:, self.num_ref_latents :] = mean_hidden_states_refs

            # We don't do feature injection at the outermost layers because it can produce visual
            # artifacts; we want to inject deep features, not pixel values.
            should_do_feature_injection = (
                self.ref_mixing_amount > 0.0
                and downscale_factor > 1
                and extra_attention_maps_flat is not None
            )
            if should_do_feature_injection:
                hidden_states = mix_reference_into_target(
                    hidden_states,
                    extra_attention_maps_flat,
                    self.ref_mixing_amount,
                    self.num_ref_latents,
                )

            # Now reshape back to (b * f, d, c) for the output
            hidden_states = rearrange(hidden_states, "b f d c -> (b f) d c")

        else:
            # Cross-attention case: compute hidden_states_ref3
            key = attn.head_to_batch_dim(key)
            value = attn.head_to_batch_dim(value)
            attention_probs = attn.get_attention_scores(query, key, attention_mask)
            hidden_states_ref3 = torch.bmm(attention_probs, value)

            hidden_states = hidden_states_ref3
            hidden_states = attn.batch_to_head_dim(hidden_states)

        # Linear projection
        hidden_states = attn.to_out[0](hidden_states, *args)
        # Dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


def independent_max_pooling(tensor, downscale_factor_first, downscale_factor_last, mode="avg"):
    """
    Apply max pooling independently over the first two dimensions and the last two dimensions
    of a tensor with shape (B, H1, W1, H2, W2).

    Parameters:
        tensor (torch.Tensor): Input tensor of shape (B, H1, W1, H2, W2).
        downscale_factor_first (int): Pool size for the first two dimensions (H1, W1).
        downscale_factor_last (int): Pool size for the last two dimensions (H2, W2).

    Returns:
        torch.Tensor: The result after applying independent max pooling.
    """
    B, H1, W1, H2, W2 = tensor.shape
    if mode == "max":
        pool_func = F.max_pool2d
    elif mode == "avg":
        pool_func = F.avg_pool2d
    else:
        raise ValueError(f"Unknown pooling mode: {mode}")

    # Torch's pool functions don't let you explicitly specify which dimensions are to be pooled
    # over, so instead we need to reshape the tensor to move the dimensions to be pooled to the
    # end.

    # 1) Max pooling over (H1, W1)
    # Permute to bring H1 and W1 to the last dimensions
    tensor = tensor.permute(0, 3, 4, 1, 2)  # Shape: (B, H2, W2, H1, W1)
    # Reshape to merge batch and (H2, W2) dimensions
    tensor = tensor.reshape(-1, 1, H1, W1)  # Shape: (B * H2 * W2, 1, H1, W1)
    # Apply max pooling over H1 and W1
    tensor = pool_func(tensor, kernel_size=downscale_factor_first)
    # Get the new spatial dimensions after pooling
    _, _, pooled_H1, pooled_W1 = tensor.shape
    # Reshape back to separate B, H2, W2 dimensions
    tensor = tensor.reshape(B, H2, W2, pooled_H1, pooled_W1)
    # Permute back to original order but with pooled H1 and W1
    tensor = tensor.permute(0, 3, 4, 1, 2)  # Shape: (B, pooled_H1, pooled_W1, H2, W2)

    # 2) Max pooling over (H2, W2)
    # Reshape to merge batch and (pooled_H1, pooled_W1) dimensions
    B, pooled_H1, pooled_W1, H2, W2 = tensor.shape
    tensor = tensor.reshape(-1, 1, H2, W2)  # Shape: (B * pooled_H1 * pooled_W1, 1, H2, W2)
    # Apply max pooling over H2 and W2
    tensor = pool_func(tensor, kernel_size=downscale_factor_last)
    # Get the new spatial dimensions after pooling
    _, _, pooled_H2, pooled_W2 = tensor.shape
    # Reshape back to separate B, pooled_H1, pooled_W1 dimensions
    tensor = tensor.reshape(B, pooled_H1, pooled_W1, pooled_H2, pooled_W2)

    return tensor


def mix_reference_into_target(hidden_states, extra_attn_mult, ref_mixing_amount, num_ref_latents):
    b, f, d, c = hidden_states.shape
    r = num_ref_latents
    t = f - r

    # Split hidden_states into reference and target
    ref_hidden_states = hidden_states[:, :r, :, :]  # shape (b, r, d, c)
    target_hidden_states = hidden_states[:, r:, :, :]  # shape (b, t, d, c)

    # Flatten the spatial and frame dimensions
    ref_hidden_states_flat = ref_hidden_states.reshape(b, -1, c)  # shape (b, P, c), where P = r * d
    target_hidden_states_flat = target_hidden_states.reshape(
        b, -1, c
    )  # shape (b, Q, c), where Q = t * d
    extra_attn_mult_flat = extra_attn_mult.reshape(r * d, t * d)  # shape (r * d, c)

    # Pull out a batch dimension for the extra attn mult
    extra_attn_mult = extra_attn_mult_flat.unsqueeze(0)  # shape (1, P, Q)

    # Compute the maximum over all the reference frames (over the P-dimension)
    max_attn_across_all_ref = torch.amax(extra_attn_mult, dim=1, keepdim=False)  # shape (1, Q)

    # Do a softmax on extra_attn_mult so that now the weights sum to 1 across all the states that we're mixing in
    # The temperature controls the locality of the mixing
    softmax_temp = 1e-3
    extra_attn_mult = F.softmax(extra_attn_mult / softmax_temp, dim=1)

    # Compute the weighted sum of reference hidden states
    # extra_attn_mult has shape (1, P, Q)
    # We need to compute: mix = extra_attn_mult^T @ ref_hidden_states_flat
    # Resulting in shape (b, Q, c)
    mix = torch.matmul(extra_attn_mult.transpose(1, 2), ref_hidden_states_flat)  # shape (b, Q, c)

    ref_mixing_amount = ref_mixing_amount * max_attn_across_all_ref.unsqueeze(-1)  # shape (b, Q, 1)

    # Apply the mixing with the scaling factor ref_mixing_amount
    new_target_hidden_states_flat = (
        1 - ref_mixing_amount
    ) * target_hidden_states_flat + ref_mixing_amount * mix

    # Reshape back to the original target hidden_states shape
    new_target_hidden_states = new_target_hidden_states_flat.reshape(b, t, d, c)

    # Update the hidden_states tensor with the new target hidden states
    hidden_states[:, r:, :, :] = new_target_hidden_states

    return hidden_states


def attn_multiplier_pixel_dist_piecewise(dist, dropoff_start_distance, dropoff_stop_dist):
    # Simple function: 1 for dist < dropoff_start_distance, 0 for dist > dropoff_stop_dist, linear in between
    return torch.clamp(
        (dropoff_stop_dist - dist) / (dropoff_stop_dist - dropoff_start_distance), min=0, max=1
    )


def compute_mesh_warped_attention_map(
    source_frame, target_frame, s: int = 1, window_size=100, min_xattn_weight=0.1
):
    # Construct mesh warper
    mesh_warp = MeshWarp(
        height=source_frame.image_bchw.shape[2],
        width=source_frame.image_bchw.shape[3],
        device=torch.device("cuda"),
    )

    flow_field = compute_flow_field(source_frame, target_frame, mesh_warp)

    attention_map = make_attention_map_from_flow_field(
        flow_field=flow_field.squeeze(0).permute(1, 2, 0),
        scale=s,
        smearing_window_radius=int(window_size),
        min_weight=min_xattn_weight,
    )
    # Sanity-check attention map:
    assert (
        torch.all(attention_map >= 0)
        and torch.all(attention_map <= 1)
        and torch.all(torch.isfinite(attention_map))
    )

    return attention_map


def compute_flow_field(source_frame, target_frame, mesh_warp):
    # Get the height and width from the source frame's image
    _, _, h, w = source_frame.image_bchw.shape

    # Create coordinate grids for x and y
    grid_y, grid_x = torch.meshgrid(
        torch.arange(w, device=source_frame.image_bchw.device, dtype=torch.float32),
        torch.arange(h, device=source_frame.image_bchw.device, dtype=torch.float32),
        indexing="xy",
    )

    # Stack x and y into image_bchw (batch_size=1, channels=3, height, width)
    # The third channel can be zeros
    image_bchw = torch.zeros(1, 3, h, w, device=source_frame.image_bchw.device, dtype=source_frame.image_bchw.dtype)
    image_bchw[0, 0, :, :] = grid_x
    image_bchw[0, 1, :, :] = grid_y

    # Create a pseudo source frame with the coordinate image
    pseudo_source_frame = Frame(
        idx=source_frame.idx,
        image_bchw=image_bchw,
        depth_b1hw=source_frame.depth_b1hw,
        K_b44=source_frame.K_b44,
        invK_b44=source_frame.invK_b44,
        cam_to_world_b44=source_frame.cam_to_world_b44,
        world_to_cam_b44=source_frame.world_to_cam_b44,
    )

    # Perform the mesh warp
    warped_frame = mesh_warp.forward(target_frame, pseudo_source_frame)

    # Extract the warped image containing the new coordinates
    warped_image_bchw = warped_frame.image_bchw  # Shape: (1, 3, h, w)

    # Extract x and y channels to form the flow field
    flow_field = warped_image_bchw[:, :2, :, :]  # Shape: (1, 2, h, w)

    # Identify invalid pixels where depth is NaN
    invalid_mask = torch.isnan(warped_frame.depth_b1hw)  # Shape: (1, 1, h, w)

    # Set invalid flow field entries to NaN
    flow_field = flow_field.masked_fill(invalid_mask, float("nan"))

    return flow_field


def make_attention_map_from_flow_field(flow_field, scale, smearing_window_radius, min_weight=0.1):
    """
    Create an attention map from a flow field.

    Parameters:
    - flow_field: torch tensor of shape (H, W, 2), where flow_field[i, j] gives
      the (source_i, source_j) coordinates that map to (i, j) in the target frame.
    - scale: integer scale factor to downscale the flow field and attention map.

    Returns:
    - attention_map: torch tensor of shape (H_s, W_s, H_s, W_s), where H_s = H // scale,
      and W_s = W // scale. The attention map represents the mapping between the
      downscaled source and target pixel coordinates.
    """
    H, W = flow_field.shape[:2]
    H_s = H // scale
    W_s = W // scale

    device = flow_field.device
    dtype = flow_field.dtype

    # Initialize the attention map
    attention_map = torch.zeros((H_s, W_s, H_s, W_s), dtype=dtype, device=device)

    # Create grid for downsampled target indices
    target_i = torch.arange(H_s, device=device)
    target_j = torch.arange(W_s, device=device)
    target_i_grid, target_j_grid = torch.meshgrid(
        target_i, target_j, indexing="ij"
    )  # shape (H_s, W_s)

    # Scale back to full resolution
    target_i_full = (target_i_grid * scale).long()  # shape (H_s, W_s)
    target_j_full = (target_j_grid * scale).long()

    # Ensure indices are within bounds
    target_i_full = torch.clamp(target_i_full, 0, H - 1)
    target_j_full = torch.clamp(target_j_full, 0, W - 1)

    # Get flow values at the downscaled target positions
    flow_value = flow_field[target_i_full, target_j_full]  # shape (H_s, W_s, 2)

    # Identify valid flow values (non-NaN)
    valid_mask = ~torch.isnan(flow_value).any(dim=2)  # shape (H_s, W_s)
    valid_target_i_indices, valid_target_j_indices = torch.where(valid_mask)

    # Extract valid flow values
    valid_flow_value = flow_value[valid_target_i_indices, valid_target_j_indices]  # shape (N, 2)

    source_i_full = valid_flow_value[:, 0]  # shape (N,)
    source_j_full = valid_flow_value[:, 1]

    # Scale source positions down to match attention map resolution
    source_i = source_i_full / scale  # shape (N,)
    source_j = source_j_full / scale

    # Parameters for smearing
    k = smearing_window_radius // scale  # Radius for the smearing window

    # Create smearing offsets
    offsets = torch.arange(-k, k + 1, device=device)
    offsets_i, offsets_j = torch.meshgrid(
        offsets, offsets, indexing="ij"
    )  # shape (window_size, window_size)
    offsets_i_flat = offsets_i.reshape(-1)  # shape (window_size ** 2,)
    offsets_j_flat = offsets_j.reshape(-1)

    window_size = offsets_i_flat.shape[0]

    N = valid_target_i_indices.shape[0]  # Number of valid positions

    # Compute integer source indices and apply offsets
    src_i_floor = torch.floor(source_i).long()  # shape (N,)
    src_j_floor = torch.floor(source_j).long()

    src_i_indices = src_i_floor.unsqueeze(1) + offsets_i_flat.unsqueeze(0)  # shape (N, window_size)
    src_j_indices = src_j_floor.unsqueeze(1) + offsets_j_flat.unsqueeze(0)

    # Compute distances for Gaussian weights
    # Round to eliminate moire-style patterns that otherwise appear:
    dist_i = src_i_indices.float() - src_i_floor.unsqueeze(1)  # shape (N, window_size)
    dist_j = src_j_indices.float() - src_j_floor.unsqueeze(1)

    # now switch to our usual linear window
    dists = torch.sqrt(dist_i**2 + dist_j**2)
    assert k > 0, "k must be greater than 0 for piecewise window"
    weights = attn_multiplier_pixel_dist_piecewise(
        dists, dropoff_start_distance=0.0, dropoff_stop_dist=k
    )

    # Expand target indices
    target_i_indices = valid_target_i_indices.unsqueeze(1).expand(
        -1, window_size
    )  # shape (N, window_size)
    target_j_indices = valid_target_j_indices.unsqueeze(1).expand(-1, window_size)

    # Flatten all indices and weights
    src_i_indices_flat = src_i_indices.reshape(-1)  # shape (N * window_size,)
    src_j_indices_flat = src_j_indices.reshape(-1)
    target_i_indices_flat = target_i_indices.reshape(-1)
    target_j_indices_flat = target_j_indices.reshape(-1)
    weights_flat = weights.reshape(-1)

    # Filter out indices that are out of bounds
    valid_indices = (
        (src_i_indices_flat >= 0)
        & (src_i_indices_flat < H_s)
        & (src_j_indices_flat >= 0)
        & (src_j_indices_flat < W_s)
    )

    src_i_indices_flat = src_i_indices_flat[valid_indices]
    src_j_indices_flat = src_j_indices_flat[valid_indices]
    target_i_indices_flat = target_i_indices_flat[valid_indices]
    target_j_indices_flat = target_j_indices_flat[valid_indices]
    weights_flat = weights_flat[valid_indices]

    # Compute linear indices into attention_map
    src_linear_indices = src_i_indices_flat * W_s + src_j_indices_flat  # shape (M,)
    target_linear_indices = target_i_indices_flat * W_s + target_j_indices_flat

    # Flatten attention map to 1D
    attention_map_flat = attention_map.view(-1)  # shape (H_s * W_s * H_s * W_s,)

    # Compute combined indices for scatter_add_
    total_size = (H_s * W_s) * (H_s * W_s)
    flat_indices = src_linear_indices * (H_s * W_s) + target_linear_indices  # shape (M,)

    # Use scatter_add_ to accumulate weights
    attention_map_flat = torch.zeros(total_size, dtype=dtype, device=device)
    attention_map_flat.scatter_add_(0, flat_indices, weights_flat)

    # Reshape attention_map_flat back to (H_s * W_s, H_s * W_s)
    attention_map_2d = attention_map_flat.view(H_s * W_s, H_s * W_s)

    # Reshape to (H_s, W_s, H_s, W_s)
    attention_map = attention_map_2d.view(H_s, W_s, H_s, W_s)

    # Attend a little bit to the whole image
    attention_map = attention_map.clamp(min=min_weight)

    return attention_map
