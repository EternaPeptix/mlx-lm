import unittest

import mlx.core as mx

from mlx_lm.generate import generate_step
from mlx_lm.sample_utils import make_sampler


class _FakeVocabParallelModel:
    def __init__(self, *, supported: bool = True):
        self.supported = supported
        self.full_calls = 0
        self.compact_calls = 0

    def supports_vocab_parallel_greedy(self) -> bool:
        return self.supported

    def vocab_parallel_greedy(self, inputs, cache=None):
        self.compact_calls += 1
        return mx.full((inputs.shape[0],), 3, dtype=mx.uint32)

    def __call__(self, inputs, cache=None):
        self.full_calls += 1
        row = mx.array([0.0, 1.0, 4.0, 2.0])
        return mx.broadcast_to(row, (*inputs.shape, row.shape[0]))


class TestGreedyVocabParallel(unittest.TestCase):
    def _one_step(self, model, **kwargs):
        return next(
            generate_step(
                mx.array([1], dtype=mx.uint32),
                model,
                max_tokens=1,
                async_lookahead=False,
                prompt_cache=[],
                **kwargs,
            )
        )

    def test_compact_path_returns_token_without_logprobs(self):
        model = _FakeVocabParallelModel()
        token, logprobs = self._one_step(
            model,
            sampler=make_sampler(temp=0.0),
            greedy_vocab_parallel_no_logprobs=True,
        )

        self.assertEqual(token, 3)
        self.assertEqual(logprobs.size, 0)
        self.assertEqual(model.compact_calls, 1)
        self.assertEqual(model.full_calls, 0)

    def test_logits_processor_falls_back_to_full_logits(self):
        model = _FakeVocabParallelModel()
        token, logprobs = self._one_step(
            model,
            sampler=make_sampler(temp=0.0),
            logits_processors=[lambda _tokens, logits: logits],
            greedy_vocab_parallel_no_logprobs=True,
        )

        self.assertEqual(token, 2)
        self.assertEqual(logprobs.size, 4)
        self.assertEqual(model.compact_calls, 0)
        self.assertEqual(model.full_calls, 1)

    def test_unmarked_sampler_falls_back_to_full_logits(self):
        model = _FakeVocabParallelModel()
        token, logprobs = self._one_step(
            model,
            sampler=lambda _logprobs: mx.array([1], dtype=mx.uint32),
            greedy_vocab_parallel_no_logprobs=True,
        )

        self.assertEqual(token, 1)
        self.assertEqual(logprobs.size, 4)
        self.assertEqual(model.compact_calls, 0)
        self.assertEqual(model.full_calls, 1)

    def test_unsupported_model_falls_back_to_full_logits(self):
        model = _FakeVocabParallelModel(supported=False)
        token, logprobs = self._one_step(
            model,
            sampler=make_sampler(temp=0.0),
            greedy_vocab_parallel_no_logprobs=True,
        )

        self.assertEqual(token, 2)
        self.assertEqual(logprobs.size, 4)
        self.assertEqual(model.compact_calls, 0)
        self.assertEqual(model.full_calls, 1)
