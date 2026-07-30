# Copyright © 2023-2024 Apple Inc.

import math
import os

import mlx.core as mx
import mlx.nn as nn

from .activations import swiglu


_DISABLE_SORTED_QMM = os.environ.get("MLX_LM_DISABLE_SORTED_QMM", "").lower() in {
    "1",
    "true",
    "yes",
}

SORTED_QMM_PACKED_CONFIG_KEY = "mlx_cuda_sorted_qmm_packed"
SORTED_QMM_PACKED_FORMAT_VERSION = 1
SORTED_QMM_PACKED_LAYOUT = "mxfp4-n256-k64-xor-v1"
_CUDA_SORTED_WEIGHTED_REDUCE = None


def _packed_sorted_qmm_enabled() -> bool:
    return os.environ.get("MLX_CUDA_SORTED_QMM_PACKED", "").lower() in {
        "1",
        "true",
        "yes",
    }


def _fused_moe_reduce_enabled() -> bool:
    return os.environ.get("MLX_CUDA_FUSED_MOE_REDUCE", "").lower() in {
        "1",
        "true",
        "yes",
    }


def _fused_moe_reduce_min_assignments() -> int:
    value = os.environ.get(
        "MLX_CUDA_FUSED_MOE_REDUCE_MIN_ASSIGNMENTS",
        "2048",
    )
    try:
        return max(1, int(value))
    except ValueError:
        return 2048


def _cuda_vector_width(dtype) -> int:
    if dtype == mx.float32:
        return 4
    if dtype in (mx.float16, mx.bfloat16):
        return 8
    return 0


def _can_fuse_sorted_weighted_reduce(
    sorted_rows: mx.array,
    indices: mx.array,
    scores: mx.array,
    *,
    training: bool,
) -> bool:
    """Check the inference-only contract for the CUDA MoE output kernel."""
    vector_width = _cuda_vector_width(sorted_rows.dtype)
    return (
        _fused_moe_reduce_enabled()
        and not training
        and mx.cuda.is_available()
        and vector_width > 0
        and indices.size >= _fused_moe_reduce_min_assignments()
        and indices.shape == scores.shape
        and sorted_rows.ndim >= 2
        and sorted_rows.size == indices.size * sorted_rows.shape[-1]
        and sorted_rows.shape[-1] % vector_width == 0
    )


def _cuda_sorted_weighted_reduce(
    sorted_rows: mx.array,
    inverse: mx.array,
    scores: mx.array,
) -> mx.array:
    """Unsort and reduce top-k expert outputs in one vectorized CUDA pass."""
    global _CUDA_SORTED_WEIGHTED_REDUCE
    if _CUDA_SORTED_WEIGHTED_REDUCE is None:
        _CUDA_SORTED_WEIGHTED_REDUCE = mx.fast.cuda_kernel(
            name="mlx_lm_sorted_moe_weighted_reduce_vec16",
            input_names=["sorted_rows", "inverse", "scores"],
            output_names=["out"],
            source=r"""
                struct alignas(16) Vector {
                  T values[VEC];
                };

                auto vector_index =
                    cooperative_groups::this_grid().thread_rank();
                long long tokens = 1;
                for (int dim = 0; dim < scores_ndim - 1; ++dim) {
                  tokens *= static_cast<long long>(scores_shape[dim]);
                }
                long long output_vectors =
                    tokens * VECTORS_PER_ROW;
                if (vector_index >= output_vectors) {
                  return;
                }
                int token = vector_index / VECTORS_PER_ROW;
                int vector_column =
                    vector_index - token * VECTORS_PER_ROW;
                int column = vector_column * VEC;
                float accumulators[VEC];
                #pragma unroll
                for (int lane = 0; lane < VEC; ++lane) {
                  accumulators[lane] = 0.0f;
                }
                #pragma unroll
                for (int slot = 0; slot < TOPK; ++slot) {
                  int assignment = token * TOPK + slot;
                  long long sorted_row =
                      static_cast<long long>(inverse[assignment]);
                  const T* source =
                      sorted_rows + sorted_row * HIDDEN + column;
                  Vector packed =
                      *reinterpret_cast<const Vector*>(source);
                  float score =
                      static_cast<float>(scores[assignment]);
                  #pragma unroll
                  for (int lane = 0; lane < VEC; ++lane) {
                    accumulators[lane] +=
                        static_cast<float>(packed.values[lane])
                        * score;
                  }
                }
                Vector result;
                #pragma unroll
                for (int lane = 0; lane < VEC; ++lane) {
                  result.values[lane] =
                      static_cast<T>(accumulators[lane]);
                }
                *reinterpret_cast<Vector*>(
                    out + token * HIDDEN + column) = result;
            """,
        )

    hidden = sorted_rows.shape[-1]
    topk = scores.shape[-1]
    tokens = scores.size // topk
    vector_width = _cuda_vector_width(sorted_rows.dtype)
    output_shape = tuple(scores.shape[:-1]) + (hidden,)
    vectors_per_row = hidden // vector_width
    output_vectors = tokens * vectors_per_row
    return _CUDA_SORTED_WEIGHTED_REDUCE(
        inputs=[sorted_rows, inverse, scores],
        output_shapes=[output_shape],
        output_dtypes=[sorted_rows.dtype],
        grid=(output_vectors, 1, 1),
        threadgroup=(256, 1, 1),
        template=[
            ("T", sorted_rows.dtype),
            ("VEC", vector_width),
            ("HIDDEN", hidden),
            ("TOPK", topk),
            ("VECTORS_PER_ROW", vectors_per_row),
        ],
        stream=mx.gpu,
    )[0]


def validate_sorted_qmm_packed_config(config: dict) -> bool:
    """Validate and detect an offline-packed sorted-QMM checkpoint."""
    packed = config.get(SORTED_QMM_PACKED_CONFIG_KEY)
    if packed is None:
        return False
    if not isinstance(packed, dict):
        raise ValueError(
            f"{SORTED_QMM_PACKED_CONFIG_KEY} must be a JSON object"
        )
    expected = {
        "format_version": SORTED_QMM_PACKED_FORMAT_VERSION,
        "layout": SORTED_QMM_PACKED_LAYOUT,
    }
    for key, value in expected.items():
        if packed.get(key) != value:
            raise ValueError(
                f"unsupported packed sorted-QMM checkpoint {key}: "
                f"{packed.get(key)!r}; expected {value!r}"
            )
    return True


def _pack_mxfp4_weight_for_sorted_qmm(weight: mx.array) -> mx.array:
    """Pack one expert weight for MLX-CUDA's N256/K64 sorted-QMM path."""
    if weight.dtype != mx.uint32 or weight.ndim != 3:
        raise ValueError(
            "packed sorted QMM requires a 3D uint32 MXFP4 expert weight"
        )
    experts, output_dims, packed_input_dims = weight.shape
    if output_dims % 256 or packed_input_dims % 8:
        raise ValueError(
            "packed sorted QMM requires output dimensions divisible by 256 "
            "and input dimensions divisible by 64"
        )

    # The shared-memory Swizzle<3,3,3> changes only the row-word coordinate:
    # physical_row = logical_row ^ k_word. Each group of eight MXFP4 values is
    # already one uint32, so no nibble-level repacking is needed.
    n_tiles = output_dims // 256
    k_tiles = packed_input_dims // 8
    tiled = weight.reshape(experts, n_tiles, 32, 8, k_tiles, 8)
    tiled = mx.transpose(tiled, (0, 1, 4, 2, 5, 3))
    # Build the fixed 8x8 XOR permutation from views and concatenations.
    # This avoids materializing a full-size index tensor for the very large
    # GLM down projections while emitting the same physical order.
    k_word_planes = []
    for k_word in range(8):
        plane = tiled[..., k_word, :]
        physical_rows = [
            plane[..., k_word ^ packed_row : (k_word ^ packed_row) + 1]
            for packed_row in range(8)
        ]
        k_word_planes.append(
            mx.concatenate(physical_rows, axis=-1)[..., None, :]
        )
    packed = mx.concatenate(k_word_planes, axis=-2)
    return mx.contiguous(packed).reshape(weight.shape)


def _pack_mxfp4_scales_for_sorted_qmm(scales: mx.array) -> mx.array:
    """Pack E8M0 scales into MLX-CUDA's N256/K64 sorted-QMM tile order."""
    if scales.dtype != mx.uint8 or scales.ndim != 3:
        raise ValueError(
            "packed sorted QMM requires 3D uint8 MXFP4 scales"
        )
    experts, output_dims, input_groups = scales.shape
    if output_dims % 256 or input_groups % 2:
        raise ValueError(
            "packed sorted QMM scales require output dimensions divisible "
            "by 256 and an even number of K/32 groups"
        )

    n_tiles = output_dims // 256
    k_tiles = input_groups // 2
    tiled = scales.reshape(experts, n_tiles, 256, k_tiles, 2)
    packed = mx.transpose(tiled, (0, 1, 3, 2, 4))
    return mx.contiguous(packed).reshape(scales.shape)


def pack_mxfp4_switch_weights(
    model: nn.Module,
    weights: dict[str, mx.array],
    *,
    prepacked: bool = False,
) -> dict[str, mx.array]:
    """Pack every MXFP4 SwitchLinear weight when the CUDA opt-in is enabled."""
    enabled = _packed_sorted_qmm_enabled()
    if prepacked and not enabled:
        raise ValueError(
            "this checkpoint uses the CUDA packed sorted-QMM layout; set "
            "MLX_CUDA_SORTED_QMM_PACKED=1 and use a compatible MLX-CUDA build"
        )
    if not enabled:
        return weights

    for path, module in model.named_modules():
        if not isinstance(module, QuantizedSwitchLinear):
            continue
        if (
            module.mode != "mxfp4"
            or module.bits != 4
            or module.group_size != 32
        ):
            continue
        weight_key = f"{path}.weight"
        scales_key = f"{path}.scales"
        if weight_key not in weights:
            continue
        if scales_key not in weights:
            raise ValueError(
                f"missing MXFP4 scales for packed SwitchLinear {path}"
            )
        original_weight = weights[weight_key]
        original_scales = weights[scales_key]
        if (
            original_weight.ndim != 3
            or original_scales.ndim != 3
            or original_weight.shape[-2] % 256
            or original_weight.shape[-1] % 8
            or original_scales.shape[-2] % 256
            or original_scales.shape[-1] % 2
        ):
            continue
        if prepacked:
            # Offline-packed safetensors are already in the physical layout
            # consumed by the CUDA kernel. Keeping these file-backed arrays
            # intact is the entire point of the checkpoint format.
            module._requires_sorted_qmm = True
            continue
        packed_weight = _pack_mxfp4_weight_for_sorted_qmm(original_weight)
        packed_scales = _pack_mxfp4_scales_for_sorted_qmm(original_scales)
        # Materialize one projection at a time. Keeping this lazy until the
        # whole model is evaluated would retain both layouts for every MoE
        # layer and can exceed a 128 GB Spark's unified memory.
        mx.eval(packed_weight, packed_scales)
        weights[weight_key] = packed_weight
        weights[scales_key] = packed_scales
        module._requires_sorted_qmm = True
        del original_weight, original_scales
        mx.clear_cache()

    return weights


def _gather_sort(x, indices):
    *_, M = indices.shape
    indices = indices.flatten()
    order = mx.argsort(indices)
    inv_order = mx.argsort(order)
    return x.flatten(0, -3)[order // M], indices[order], inv_order


def _scatter_unsort(x, inv_order, shape=None):
    x = x[inv_order]
    if shape is not None:
        x = mx.unflatten(x, 0, shape)
    return x


def _should_sort_switch(indices, *projections) -> bool:
    # Packed CUDA weights are only meaningful to the sorted-QMM kernel.  A
    # token-by-token decode normally falls below the throughput-oriented
    # sorting threshold, so retain the marker established while loading the
    # checkpoint and force the compatible path for every decode step.
    return indices.size >= 64 or any(
        getattr(projection, "_requires_sorted_qmm", False)
        for projection in projections
    )


class QuantizedSwitchLinear(nn.Module):
    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        num_experts: int,
        bias: bool = True,
        group_size: int = 64,
        bits: int = 4,
        mode: str = "affine",
    ):
        super().__init__()

        scale = math.sqrt(1 / input_dims)
        self.weight, self.scales, *biases = mx.quantize(
            mx.random.uniform(
                low=-scale,
                high=scale,
                shape=(num_experts, output_dims, input_dims),
            ),
            group_size=group_size,
            bits=bits,
            mode=mode,
        )
        self.biases = biases[0] if biases else None

        if bias:
            self.bias = mx.zeros((num_experts, output_dims))

        self.group_size = group_size
        self.bits = bits
        self.mode = mode
        self._requires_sorted_qmm = False

        # Freeze this model's parameters
        self.freeze()

    @property
    def input_dims(self):
        return self.scales.shape[2] * self.group_size

    @property
    def output_dims(self):
        return self.weight.shape[1]

    @property
    def num_experts(self):
        return self.weight.shape[0]

    def __call__(self, x, indices, sorted_indices=False):
        x = mx.gather_qmm(
            x,
            self["weight"],
            self["scales"],
            self.get("biases"),
            rhs_indices=indices,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
            sorted_indices=sorted_indices and not _DISABLE_SORTED_QMM,
        )
        if "bias" in self:
            x = x + mx.expand_dims(self["bias"][indices], -2)
        return x


class SwitchLinear(nn.Module):
    def __init__(
        self, input_dims: int, output_dims: int, num_experts: int, bias: bool = True
    ):
        super().__init__()
        scale = math.sqrt(1 / input_dims)
        self.weight = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(num_experts, output_dims, input_dims),
        )

        if bias:
            self.bias = mx.zeros((num_experts, output_dims))

    @property
    def input_dims(self):
        return self.weight.shape[2]

    @property
    def output_dims(self):
        return self.weight.shape[1]

    @property
    def num_experts(self):
        return self.weight.shape[0]

    def __call__(self, x, indices, sorted_indices=False):
        x = mx.gather_mm(
            x,
            self["weight"].swapaxes(-1, -2),
            rhs_indices=indices,
            sorted_indices=sorted_indices,
        )
        if "bias" in self:
            x = x + mx.expand_dims(self["bias"][indices], -2)
        return x

    def to_quantized(self, group_size: int = 64, bits: int = 4, mode: str = "affine"):
        num_experts, output_dims, input_dims = self.weight.shape
        ql = QuantizedSwitchLinear(
            input_dims,
            output_dims,
            num_experts,
            False,
            group_size,
            bits,
            mode=mode,
        )
        ql.weight, ql.scales, *biases = mx.quantize(
            self.weight, group_size, bits, mode=mode
        )
        ql.biases = biases[0] if biases else None

        if "bias" in self:
            ql.bias = self.bias
        return ql


class SwiGLU(nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, x, gate):
        return swiglu(gate, x)


class SwitchGLU(nn.Module):
    def __init__(
        self,
        input_dims: int,
        hidden_dims: int,
        num_experts: int,
        activation=SwiGLU(),
        bias: bool = False,
    ):
        super().__init__()

        self.gate_proj = SwitchLinear(input_dims, hidden_dims, num_experts, bias=bias)
        self.up_proj = SwitchLinear(input_dims, hidden_dims, num_experts, bias=bias)
        self.down_proj = SwitchLinear(hidden_dims, input_dims, num_experts, bias=bias)
        self.activation = activation

    def _forward_sorted(self, x, indices):
        x = mx.expand_dims(x, (-2, -3))

        # When we have many tokens, then sort them to make sure that the access
        # of different experts is in order.
        do_sort = _should_sort_switch(
            indices,
            self.gate_proj,
            self.up_proj,
            self.down_proj,
        )
        idx = indices
        inv_order = None
        if do_sort:
            x, idx, inv_order = _gather_sort(x, indices)
        if self.training:
            idx = mx.stop_gradient(idx)
        x_up = self.up_proj(x, idx, sorted_indices=do_sort)
        x_gate = self.gate_proj(x, idx, sorted_indices=do_sort)
        x = self.down_proj(
            self.activation(x_up, x_gate),
            idx,
            sorted_indices=do_sort,
        )

        return x, inv_order, do_sort

    def __call__(self, x, indices) -> mx.array:
        x, inv_order, do_sort = self._forward_sorted(x, indices)
        if do_sort:
            x = _scatter_unsort(x, inv_order, indices.shape)

        return x.squeeze(-2)

    def weighted_call(self, x, indices, scores) -> mx.array:
        """Run routed experts and combine their weighted outputs."""
        x, inv_order, do_sort = self._forward_sorted(x, indices)
        if do_sort and _can_fuse_sorted_weighted_reduce(
            x,
            indices,
            scores,
            training=self.training,
        ):
            return _cuda_sorted_weighted_reduce(x, inv_order, scores)

        if do_sort:
            x = _scatter_unsort(x, inv_order, indices.shape)
        x = x.squeeze(-2)
        return (x * scores[..., None]).sum(axis=-2).astype(x.dtype)


class SwitchMLP(nn.Module):
    def __init__(
        self,
        input_dims: int,
        hidden_dims: int,
        num_experts: int,
        activation=nn.GELU(approx="precise"),
        bias: bool = False,
    ):
        super().__init__()

        self.fc1 = SwitchLinear(input_dims, hidden_dims, num_experts, bias=bias)
        self.fc2 = SwitchLinear(hidden_dims, input_dims, num_experts, bias=bias)
        self.activation = activation

    def __call__(self, x, indices) -> mx.array:
        x = mx.expand_dims(x, (-2, -3))

        # When we have many tokens, then sort them to make sure that the access
        # of different experts is in order.
        do_sort = _should_sort_switch(indices, self.fc1, self.fc2)
        idx = indices
        inv_order = None
        if do_sort:
            x, idx, inv_order = _gather_sort(x, indices)
        if self.training:
            idx = mx.stop_gradient(idx)
        x = self.fc1(x, idx, sorted_indices=do_sort)
        x = self.activation(x)
        x = self.fc2(x, idx, sorted_indices=do_sort)

        if do_sort:
            x = _scatter_unsort(x, inv_order, indices.shape)

        return x.squeeze(-2)
