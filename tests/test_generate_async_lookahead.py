"""Lifecycle tests for pipeline-safe token generation."""

from __future__ import annotations

import unittest

import mlx.core as mx

from mlx_lm.generate import generate_step


class _CountingModel:
    def __init__(self, forced_token: int = 3, vocab_size: int = 8) -> None:
        self.calls = 0
        self.forced_token = forced_token
        self.vocab_size = vocab_size

    def __call__(self, inputs: mx.array, cache: list[object] | None = None) -> mx.array:
        del cache
        self.calls += 1
        row = mx.array(
            [
                10.0 if token == self.forced_token else 0.0
                for token in range(self.vocab_size)
            ]
        )
        return mx.broadcast_to(row, (*inputs.shape, self.vocab_size))


def _steps(model: _CountingModel, *, max_tokens: int, async_lookahead: bool):
    return generate_step(
        prompt=mx.array([1], dtype=mx.uint32),
        model=model,
        max_tokens=max_tokens,
        async_lookahead=async_lookahead,
        prompt_cache=[],
    )


class AsyncLookaheadLifecycleTest(unittest.TestCase):
    def test_pipeline_consumer_close_does_not_start_an_extra_forward(self) -> None:
        model = _CountingModel()
        steps = _steps(model, max_tokens=50, async_lookahead=False)

        token, _ = next(steps)
        self.assertEqual(token, model.forced_token)
        self.assertEqual(model.calls, 1)

        # EOS, stop sequences, and cancellation all close or abandon the
        # iterator while it is suspended at this yield.
        steps.close()
        self.assertEqual(model.calls, 1)

    def test_pipeline_consumer_exception_does_not_start_an_extra_forward(
        self,
    ) -> None:
        class Cancelled(Exception):
            pass

        model = _CountingModel()
        steps = _steps(model, max_tokens=-1, async_lookahead=False)
        next(steps)
        self.assertEqual(model.calls, 1)

        with self.assertRaises(Cancelled):
            steps.throw(Cancelled())
        self.assertEqual(model.calls, 1)

    def test_no_lookahead_preserves_token_and_forward_counts(self) -> None:
        model = _CountingModel()
        outputs = list(_steps(model, max_tokens=3, async_lookahead=False))

        self.assertEqual(
            [token for token, _ in outputs],
            [model.forced_token] * 3,
        )
        self.assertEqual(model.calls, 3)

    def test_default_path_still_looks_ahead_before_yield(self) -> None:
        model = _CountingModel()
        steps = _steps(model, max_tokens=50, async_lookahead=True)

        next(steps)
        self.assertEqual(model.calls, 2)
        steps.close()
