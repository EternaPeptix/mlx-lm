import os
from functools import partial
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn


@partial(mx.compile, shapeless=True)
def compute_g(A_log, a, dt_bias):
    return mx.exp(-mx.exp(A_log.astype(mx.float32)) * nn.softplus(a + dt_bias))


@partial(mx.compile, shapeless=True)
def compute_g_safe(A_log, a, dt_bias, lower_bound):
    return mx.exp(
        lower_bound * mx.sigmoid(mx.exp(A_log.astype(mx.float32)) * (a + dt_bias))
    )


def _make_gated_delta_kernel(
    has_mask=False,
    vectorized=False,
    return_state_history=False,
):
    if not mx.metal.is_available():
        return None
    mask_source = "mask[b_idx * T + t]" if has_mask else "true"

    # Configure g indexing based on whether gating is vectorized
    if vectorized:
        g_comment = "// g: [B, T, Hv, Dk]"
        g_setup = "auto g_ = g + (b_idx * T * Hv + hv_idx) * Dk;"
        g_access = "g_[s_idx]"
        g_advance = "g_ += Hv * Dk;"
    else:
        g_comment = "// g: [B, T, Hv]"
        g_setup = "auto g_ = g + b_idx * T * Hv;"
        g_access = "g_[hv_idx]"
        g_advance = "g_ += Hv;"

    state_history_write = (
        """
            auto h_state =
                state_history + ((n * T + t) * Dv + dv_idx) * Dk;
            for (int i = 0; i < n_per_t; ++i) {
              auto s_idx = n_per_t * dk_idx + i;
              h_state[s_idx] = static_cast<StT>(state[i]);
            }
        """
        if return_state_history
        else ""
    )

    source = f"""
        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);
        constexpr int n_per_t = Dk / 32;

        // q, k: [B, T, Hk, Dk]
        auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
        auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;

        // v, y: [B, T, Hv, Dv]
        auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
        y += b_idx * T * Hv * Dv + hv_idx * Dv;

        auto dk_idx = thread_position_in_threadgroup.x;
        auto dv_idx = thread_position_in_grid.y;

        // state_in, state_out: [B, Hv, Dv, Dk]
        auto i_state = state_in + (n * Dv + dv_idx) * Dk;
        auto o_state = state_out + (n * Dv + dv_idx) * Dk;

        float state[n_per_t];
        for (int i = 0; i < n_per_t; ++i) {{
          auto s_idx = n_per_t * dk_idx + i;
          state[i] = static_cast<float>(i_state[s_idx]);
        }}

        {g_comment}
        {g_setup}
        auto beta_ = beta + b_idx * T * Hv;

        for (int t = 0; t < T; ++t) {{
          if ({mask_source}) {{
            float kv_mem = 0.0f;
            for (int i = 0; i < n_per_t; ++i) {{
              auto s_idx = n_per_t * dk_idx + i;
              state[i] = state[i] * {g_access};
              kv_mem += state[i] * k_[s_idx];
            }}
            kv_mem = simd_sum(kv_mem);

            auto delta = (v_[dv_idx] - kv_mem) * beta_[hv_idx];

            float out = 0.0f;
            for (int i = 0; i < n_per_t; ++i) {{
              auto s_idx = n_per_t * dk_idx + i;
              state[i] = state[i] + k_[s_idx] * delta;
              out += state[i] * q_[s_idx];
            }}
            out = simd_sum(out);
            if (thread_index_in_simdgroup == 0) {{
              y[dv_idx] = static_cast<InT>(out);
            }}
          }} else {{
            y[dv_idx] = static_cast<InT>(0);
          }}
          // Increment data pointers to next time step
          q_ += Hk * Dk;
          k_ += Hk * Dk;
          v_ += Hv * Dv;
          y += Hv * Dv;
          {g_advance}
          beta_ += Hv;
          {state_history_write}
        }}
        for (int i = 0; i < n_per_t; ++i) {{
          auto s_idx = n_per_t * dk_idx + i;
          o_state[s_idx] = static_cast<StT>(state[i]);
        }}
    """
    inputs = ["q", "k", "v", "g", "beta", "state_in", "T"]
    if has_mask:
        inputs.append("mask")

    suffix = ""
    if vectorized:
        suffix += "_vec"
    if has_mask:
        suffix += "_mask"
    if return_state_history:
        suffix += "_history"

    return mx.fast.metal_kernel(
        name=f"gated_delta_step{suffix}",
        input_names=inputs,
        output_names=(
            ["y", "state_out", "state_history"]
            if return_state_history
            else ["y", "state_out"]
        ),
        source=source,
    )


_gated_delta_kernel = _make_gated_delta_kernel(has_mask=False, vectorized=False)
_gated_delta_kernel_masked = _make_gated_delta_kernel(has_mask=True, vectorized=False)
_gated_delta_kernel_vec = _make_gated_delta_kernel(has_mask=False, vectorized=True)
_gated_delta_kernel_vec_masked = _make_gated_delta_kernel(
    has_mask=True, vectorized=True
)
_gated_delta_kernel_history = _make_gated_delta_kernel(return_state_history=True)
_gated_delta_kernel_masked_history = _make_gated_delta_kernel(
    has_mask=True,
    return_state_history=True,
)
_gated_delta_kernel_vec_history = _make_gated_delta_kernel(
    vectorized=True,
    return_state_history=True,
)
_gated_delta_kernel_vec_masked_history = _make_gated_delta_kernel(
    has_mask=True,
    vectorized=True,
    return_state_history=True,
)


_EXPERIMENTAL_KDA_ROW_PREFILL_ENV = "MLX_LM_EXPERIMENTAL_KDA_ROW_PREFILL"
_EXPERIMENTAL_KDA_ROW_PREFILL_MIN_TOKENS = 128
_EXPERIMENTAL_KDA_ROWS_PER_SIMD = 4


def _make_experimental_kda_row_prefill_kernel():
    """Make the exact row-tiled KDA prefill kernel.

    The existing recurrent kernel assigns one SIMD-group to one value row.
    This specialization assigns four value rows to each SIMD-group so q, k,
    the vector gate, and beta are loaded once and reused across those rows.
    It does not materialize chunk intermediates and keeps the public FP32
    state contract.
    """

    if not mx.metal.is_available():
        return None

    source = r"""
        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);
        auto lane = thread_index_in_simdgroup;
        auto row_group = thread_position_in_grid.y;
        auto dv_base = row_group * RowsPerSimd;
        constexpr int n_per_t = Dk / 32;

        // q, k: [B, T, Hk, Dk]
        auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
        auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;

        // v, y: [B, T, Hv, Dv]
        auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
        y += b_idx * T * Hv * Dv + hv_idx * Dv;

        // state_in, state_out: [B, Hv, Dv, Dk]
        auto i_state_head = state_in + n * Dv * Dk;
        auto o_state_head = state_out + n * Dv * Dk;

        float state[RowsPerSimd][n_per_t];
        for (int row = 0; row < RowsPerSimd; ++row) {
          auto dv_idx = dv_base + row;
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * lane + i;
            state[row][i] =
                dv_idx < Dv
                    ? static_cast<float>(
                          i_state_head[dv_idx * Dk + s_idx])
                    : 0.0f;
          }
        }

        // Kimi K3 uses a vector gate: [B, T, Hv, Dk].
        auto g_ = g + (b_idx * T * Hv + hv_idx) * Dk;
        auto beta_ = beta + b_idx * T * Hv;

        for (int t = 0; t < T; ++t) {
          float q_reg[n_per_t];
          float k_reg[n_per_t];
          float g_reg[n_per_t];
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * lane + i;
            q_reg[i] = static_cast<float>(q_[s_idx]);
            k_reg[i] = static_cast<float>(k_[s_idx]);
            g_reg[i] = static_cast<float>(g_[s_idx]);
          }
          auto beta_value = static_cast<float>(beta_[hv_idx]);

          for (int row = 0; row < RowsPerSimd; ++row) {
            auto dv_idx = dv_base + row;
            if (dv_idx < Dv) {
              float kv_mem = 0.0f;
              for (int i = 0; i < n_per_t; ++i) {
                state[row][i] = state[row][i] * g_reg[i];
                kv_mem += state[row][i] * k_reg[i];
              }
              kv_mem = simd_sum(kv_mem);

              auto delta =
                  (static_cast<float>(v_[dv_idx]) - kv_mem) *
                  beta_value;

              float out = 0.0f;
              for (int i = 0; i < n_per_t; ++i) {
                state[row][i] =
                    state[row][i] + k_reg[i] * delta;
                out += state[row][i] * q_reg[i];
              }
              out = simd_sum(out);
              if (lane == 0) {
                y[dv_idx] = static_cast<InT>(out);
              }
            }
          }

          q_ += Hk * Dk;
          k_ += Hk * Dk;
          v_ += Hv * Dv;
          y += Hv * Dv;
          g_ += Hv * Dk;
          beta_ += Hv;
        }

        for (int row = 0; row < RowsPerSimd; ++row) {
          auto dv_idx = dv_base + row;
          if (dv_idx < Dv) {
            for (int i = 0; i < n_per_t; ++i) {
              auto s_idx = n_per_t * lane + i;
              o_state_head[dv_idx * Dk + s_idx] =
                  static_cast<StT>(state[row][i]);
            }
          }
        }
    """

    return mx.fast.metal_kernel(
        name="gated_delta_kda_row4_prefill",
        input_names=["q", "k", "v", "g", "beta", "state_in", "T"],
        output_names=["y", "state_out"],
        source=source,
    )


_experimental_kda_row_prefill_kernel = _make_experimental_kda_row_prefill_kernel()


def experimental_kda_row_prefill_enabled() -> bool:
    """Return whether the default-off row-tiled prefill path was requested."""

    value = os.environ.get(_EXPERIMENTAL_KDA_ROW_PREFILL_ENV, "")
    return value.lower() in {"1", "true", "yes", "on"}


def experimental_kda_row_prefill_eligible(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    state: mx.array,
    mask: Optional[mx.array] = None,
    return_state_history: bool = False,
) -> bool:
    """Check the deliberately narrow Kimi K3 prefill contract."""

    return (
        experimental_kda_row_prefill_enabled()
        and _experimental_kda_row_prefill_kernel is not None
        and mx.default_device() == mx.gpu
        and mx.metal.is_available()
        and mask is None
        and not return_state_history
        and g.ndim == 4
        and q.ndim == 4
        and k.ndim == 4
        and v.ndim == 4
        and q.shape[1] >= _EXPERIMENTAL_KDA_ROW_PREFILL_MIN_TOKENS
        and q.shape[-1] == 128
        and k.shape[-1] == 128
        and v.shape[-1] == 128
        and v.shape[2] % k.shape[2] == 0
        and state.dtype == mx.float32
    )


def experimental_kda_row_prefill_kernel(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: mx.array,
    *,
    rows_per_simd: int = _EXPERIMENTAL_KDA_ROWS_PER_SIMD,
) -> Tuple[mx.array, mx.array]:
    """Run the exact, scratch-free row-tiled KDA prefill specialization."""

    B, T, Hk, Dk = k.shape
    Hv, Dv = v.shape[2:]
    if _experimental_kda_row_prefill_kernel is None:
        raise RuntimeError("The experimental KDA prefill kernel requires Metal")
    if (
        g.ndim != 4
        or Dk != 128
        or Dv != 128
        or Hv % Hk != 0
        or state.dtype != mx.float32
    ):
        raise ValueError(
            "The experimental KDA prefill kernel requires vector gates, "
            "Dk=Dv=128, aligned heads, and FP32 state"
        )
    if rows_per_simd not in (1, 2, 4, 8):
        raise ValueError("rows_per_simd must be one of 1, 2, 4, or 8")

    row_groups = (Dv + rows_per_simd - 1) // rows_per_simd
    return _experimental_kda_row_prefill_kernel(
        inputs=[q, k, v, g, beta, state, T],
        template=[
            ("InT", q.dtype),
            ("StT", state.dtype),
            ("Dk", Dk),
            ("Dv", Dv),
            ("Hk", Hk),
            ("Hv", Hv),
            ("RowsPerSimd", rows_per_simd),
        ],
        grid=(32, row_groups, B * Hv),
        threadgroup=(32, 4, 1),
        output_shapes=[(B, T, Hv, Dv), state.shape],
        output_dtypes=[q.dtype, state.dtype],
    )


@mx.compile
def _gated_delta_step_ops(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: mx.array,
    mask: Optional[mx.array] = None,
) -> Tuple[mx.array, mx.array]:
    """
    Ops-based reference implementation for a single recurrent step.

    Shapes:
      - q, k: [B, H, Dk]
      - v: [B, H, Dv]
      - g: [B, H] or [B, H, Dk]
      - beta: [B, H]
      - state: [B, H, Dv, Dk]
    Returns:
      - y: [B, H, Dv]
      - new_state: [B, H, Dv, Dk]
    """

    # Decay
    old_state = state
    if g.ndim == 2:
        decay = g[..., None, None]
    elif g.ndim == 3:
        decay = g[..., None, :]
    else:
        raise ValueError(f"Unsupported gating shape {g.shape}")
    state = state * decay
    kv_mem = (state * k[..., None, :]).sum(axis=-1)  # [B, H, Dv]
    delta = (v - kv_mem) * beta[..., None]  # [B, H, Dv]
    state = state + k[..., None, :] * delta[..., None]
    # Output projection along key dim with q
    y = (state * q[..., None, :]).sum(axis=-1)  # [B, H, Dv]

    if mask is not None:
        mask = mx.expand_dims(mask, axis=(1, 2, 3))
        state = mx.where(mask, state, old_state)
    return y.astype(q.dtype), state


def gated_delta_kernel(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: mx.array,
    mask: Optional[mx.array] = None,
    return_state_history: bool = False,
) -> Tuple[mx.array, ...]:
    B, T, Hk, Dk = k.shape
    Hv, Dv = v.shape[2:]
    input_type = q.dtype
    state_type = state.dtype
    if g.ndim == 4:
        kernel = (
            _gated_delta_kernel_vec_history
            if return_state_history
            else _gated_delta_kernel_vec
        )
        inputs = [q, k, v, g, beta, state, T]
        if mask is not None:
            kernel = (
                _gated_delta_kernel_vec_masked_history
                if return_state_history
                else _gated_delta_kernel_vec_masked
            )
            inputs.append(mask)
    else:
        kernel = (
            _gated_delta_kernel_history if return_state_history else _gated_delta_kernel
        )
        inputs = [q, k, v, g, beta, state, T]
        if mask is not None:
            kernel = (
                _gated_delta_kernel_masked_history
                if return_state_history
                else _gated_delta_kernel_masked
            )
            inputs.append(mask)

    output_shapes = [(B, T, Hv, Dv), state.shape]
    output_dtypes = [input_type, state_type]
    if return_state_history:
        output_shapes.append((B, Hv, T, Dv, Dk))
        output_dtypes.append(state_type)
    return kernel(
        inputs=inputs,
        template=[
            ("InT", input_type),
            ("StT", state_type),
            ("Dk", Dk),
            ("Dv", Dv),
            ("Hk", Hk),
            ("Hv", Hv),
        ],
        grid=(32, Dv, B * Hv),
        threadgroup=(32, 4, 1),
        output_shapes=output_shapes,
        output_dtypes=output_dtypes,
    )


def gated_delta_ops(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: Optional[mx.array] = None,
    mask: Optional[mx.array] = None,
    return_state_history: bool = False,
) -> Tuple[mx.array, ...]:
    """
    Ops-based reference implementation for prompt prefill (sequential loop).
    Supports both scalar and vectorized gating.

    Shapes:
      - q, k: [B, T, Hk, Dk]
      - v: [B, T, Hv, Dv]
      - g: [B, T, Hv] (scalar) or [B, T, Hv, Dk] (vectorized)
      - beta: [B, T, Hv]
      - state: [B, Hv, Dv, Dk]
    Returns:
      - y: [B, T, Hv, Dv]
      - state: [B, Hv, Dv, Dk]
    """
    B, T, Hk, Dk = q.shape
    Hv, Dv = v.shape[-2:]
    if state is None:
        state = mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32)

    if (repeat_factor := Hv // Hk) > 1:
        q = mx.repeat(q, repeat_factor, -2)
        k = mx.repeat(k, repeat_factor, -2)

    ys = []
    state_history = []
    for t in range(T):
        y, state = _gated_delta_step_ops(
            q[:, t],
            k[:, t],
            v[:, t],
            g[:, t],
            beta[:, t],
            state,
            None if mask is None else mask[:, t],
        )
        ys.append(y)
        if return_state_history:
            state_history.append(state)
    y = mx.stack(ys, axis=1)
    if return_state_history:
        return y, state, mx.stack(state_history, axis=2)
    return y, state


def gated_delta_update(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    a: mx.array,
    b: mx.array,
    A_log: mx.array,
    dt_bias: mx.array,
    state: Optional[mx.array] = None,
    mask: Optional[mx.array] = None,
    use_kernel: bool = True,
    lower_bound: Optional[float] = None,
    beta_scale: float = 1.0,
    return_state_history: bool = False,
) -> Tuple[mx.array, ...]:
    beta = mx.sigmoid(b)
    if beta_scale != 1.0:
        beta = beta * beta_scale
    if lower_bound is None:
        g = compute_g(A_log, a, dt_bias)
    else:
        g = compute_g_safe(A_log, a, dt_bias, lower_bound)
    if state is None:
        B, _, Hk, Dk = q.shape
        Hv, Dv = v.shape[-2:]
        state = mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32)

    if use_kernel and experimental_kda_row_prefill_eligible(
        q,
        k,
        v,
        g,
        state,
        mask,
        return_state_history,
    ):
        return experimental_kda_row_prefill_kernel(
            q,
            k,
            v,
            g,
            beta,
            state,
        )

    if (
        not use_kernel
        or mx.default_device() != mx.gpu
        or not mx.metal.is_available()
        or k.shape[-1] < 32
        or k.shape[-1] % 32 != 0
    ):
        return gated_delta_ops(
            q,
            k,
            v,
            g,
            beta,
            state,
            mask,
            return_state_history=return_state_history,
        )
    return gated_delta_kernel(
        q,
        k,
        v,
        g,
        beta,
        state,
        mask,
        return_state_history=return_state_history,
    )
