from __future__ import annotations

import unittest

import mlx.core as mx

from mlx_lm.generate import speculative_generate_step


class _FakeCache:
    def __init__(self, *, trimmable: bool):
        self.offset = 0
        self._trimmable = trimmable

    @property
    def state(self):
        return mx.array([self.offset], dtype=mx.int32)

    def is_trimmable(self):
        return self._trimmable

    def trim(self, amount):
        if not self._trimmable:
            return 0
        amount = min(self.offset, amount)
        self.offset -= amount
        return amount


class _FakeTransaction:
    def __init__(self, cache, width):
        self.cache = cache
        self.width = width
        self.initial_offset = cache.offset
        self.active = True


class _FakeModel:
    def __init__(
        self,
        *,
        transactional: bool,
        fail_wide: bool = False,
        forced_token: int = 3,
        vocab_size: int = 8,
    ):
        self.layers = [object()]
        self.transactional = transactional
        self.fail_wide = fail_wide
        self.forced_token = forced_token
        self.vocab_size = vocab_size
        self.resolve_calls = []
        self.cancel_calls = 0

        if not transactional:
            # Hook discovery is deliberately attribute-based.
            self.begin_speculative_cache = None
            self.resolve_speculative_cache = None
            self.cancel_speculative_cache = None

    def __call__(self, inputs, cache=None):
        if cache is not None:
            cache[0].offset += int(inputs.shape[-1])
        if self.fail_wide and inputs.shape[-1] > 1:
            raise RuntimeError("injected target forward failure")
        row = mx.array(
            [
                10.0 if token == self.forced_token else 0.0
                for token in range(self.vocab_size)
            ]
        )
        return mx.broadcast_to(row, (*inputs.shape, self.vocab_size))

    def begin_speculative_cache(self, cache, width):
        transaction = _FakeTransaction(cache[0], width)
        return transaction

    def resolve_speculative_cache(self, transaction, consumed):
        self.assert_transaction_advanced(transaction)
        transaction.cache.offset = transaction.initial_offset + consumed
        transaction.active = False
        self.resolve_calls.append(consumed)

    def cancel_speculative_cache(self, transaction):
        transaction.cache.offset = transaction.initial_offset
        transaction.active = False
        self.cancel_calls += 1

    @staticmethod
    def assert_transaction_advanced(transaction):
        if transaction.cache.offset != transaction.initial_offset + transaction.width:
            raise AssertionError(
                "target cache did not advance by the verification width"
            )


def _generator(
    target,
    draft,
    target_cache,
    draft_cache,
    *,
    max_tokens=5,
    stats=None,
):
    return speculative_generate_step(
        prompt=mx.array([1, 2], dtype=mx.uint32),
        model=target,
        draft_model=draft,
        num_draft_tokens=2,
        max_tokens=max_tokens,
        prompt_cache=[target_cache, draft_cache],
        speculative_round_callback=None if stats is None else stats.append,
    )


class SpeculativeTransactionLifecycleTest(unittest.TestCase):
    def test_consumer_close_resolves_only_emitted_draft_tokens(self):
        target = _FakeModel(transactional=True)
        draft = _FakeModel(transactional=False)
        target_cache = _FakeCache(trimmable=False)
        draft_cache = _FakeCache(trimmable=True)
        generator = _generator(target, draft, target_cache, draft_cache)

        token, _, from_draft = next(generator)
        self.assertEqual(token, target.forced_token)
        self.assertTrue(from_draft)
        generator.close()

        # One prompt token was prefetched. The wide target then consumed the
        # remaining prompt token and exactly one emitted accepted draft.
        self.assertEqual(target_cache.offset, 3)
        self.assertEqual(target.resolve_calls, [2])
        self.assertEqual(target.cancel_calls, 0)

    def test_wide_forward_failure_restores_the_pre_forward_cache(self):
        target = _FakeModel(transactional=True, fail_wide=True)
        draft = _FakeModel(transactional=False)
        target_cache = _FakeCache(trimmable=False)
        draft_cache = _FakeCache(trimmable=True)
        generator = _generator(target, draft, target_cache, draft_cache)

        with self.assertRaisesRegex(RuntimeError, "injected"):
            next(generator)

        self.assertEqual(target_cache.offset, 1)
        self.assertEqual(target.resolve_calls, [])
        self.assertEqual(target.cancel_calls, 1)

    def test_normal_completion_releases_every_transaction(self):
        target = _FakeModel(transactional=True)
        draft = _FakeModel(transactional=False)
        target_cache = _FakeCache(trimmable=False)
        draft_cache = _FakeCache(trimmable=True)
        stats = []

        outputs = list(
            _generator(
                target,
                draft,
                target_cache,
                draft_cache,
                max_tokens=5,
                stats=stats,
            )
        )

        self.assertEqual(len(outputs), 5)
        self.assertTrue(target.resolve_calls)
        self.assertEqual(target.cancel_calls, 0)
        self.assertTrue(all(1 <= consumed <= 3 for consumed in target.resolve_calls))
        self.assertTrue(stats)
        self.assertTrue(
            all(round_stats.source == "draft_model" for round_stats in stats)
        )
        self.assertEqual(sum(round_stats.committed_tokens for round_stats in stats), 5)


if __name__ == "__main__":
    unittest.main()
