"""Lifecycle tests for pipeline-safe token generation."""

from __future__ import annotations

import mlx.core as mx

from mlx_lm.generate import generate_step


class _CountingModel:
    def __init__(self, forced_token: int = 3, vocab_size: int = 8) -> None:
        self.calls = 0
        self.forced_token = forced_token
        self.vocab_size = vocab_size

    def __call__(
        self, inputs: mx.array, cache: list[object] | None = None
    ) -> mx.array:
        del cache
        self.calls += 1
        row = mx.array(
            [
                10.0 if token == self.forced_token else 0.0
                for token in range(self.vocab_size)
            ]
        )
        return mx.broadcast_to(row, (*inputs.shape, self.vocab_size))


def _steps(
    model: _CountingModel, *, max_tokens: int, async_lookahead: bool
):
    return generate_step(
        prompt=mx.array([1], dtype=mx.uint32),
        model=model,
        max_tokens=max_tokens,
        async_lookahead=async_lookahead,
        prompt_cache=[],
    )


def test_pipeline_consumer_close_does_not_start_an_extra_forward() -> None:
    model = _CountingModel()
    steps = _steps(model, max_tokens=50, async_lookahead=False)

    token, _ = next(steps)
    assert token == model.forced_token
    assert model.calls == 1

    # EOS, stop sequences, and cancellation all close or abandon the iterator
    # while it is suspended at this yield.
    steps.close()
    assert model.calls == 1


def test_pipeline_consumer_exception_does_not_start_an_extra_forward() -> None:
    class Cancelled(Exception):
        pass

    model = _CountingModel()
    steps = _steps(model, max_tokens=-1, async_lookahead=False)
    next(steps)
    assert model.calls == 1

    try:
        steps.throw(Cancelled())
    except Cancelled:
        pass
    else:
        raise AssertionError("the injected cancellation should propagate")
    assert model.calls == 1


def test_no_lookahead_preserves_token_and_forward_counts() -> None:
    model = _CountingModel()
    outputs = list(_steps(model, max_tokens=3, async_lookahead=False))

    assert [token for token, _ in outputs] == [model.forced_token] * 3
    assert model.calls == 3


def test_default_path_still_looks_ahead_before_yield() -> None:
    model = _CountingModel()
    steps = _steps(model, max_tokens=50, async_lookahead=True)

    next(steps)
    assert model.calls == 2
    steps.close()
