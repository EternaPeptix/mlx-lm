from __future__ import annotations

import unittest

import mlx.core as mx

from mlx_lm.generate import (
    PromptLookupDrafter,
    setup_arg_parser,
    speculative_generate_step,
    stream_generate,
)
from mlx_lm.tokenizer_utils import TokenizerWrapper


class _FakeCache:
    def __init__(self):
        self.offset = 0

    @property
    def state(self):
        return mx.array([self.offset], dtype=mx.int32)

    def is_trimmable(self):
        return False


class _FakeTransaction:
    def __init__(self, cache, width):
        self.cache = cache
        self.width = width
        self.initial_offset = cache.offset
        self.active = True


class _PatternModel:
    def __init__(self, *, fail_wide=False, vocab_size=8):
        self.layers = [object()]
        self.fail_wide = fail_wide
        self.vocab_size = vocab_size
        self.resolve_calls = []
        self.cancel_calls = 0
        self.transaction_active = False
        self.next_token = {1: 2, 2: 3, 3: 4, 4: 1}

    def __call__(self, inputs, cache=None):
        if cache is not None:
            cache[0].offset += int(inputs.shape[-1])
        if self.fail_wide and self.transaction_active and inputs.shape[-1] > 1:
            raise RuntimeError("injected target forward failure")

        rows = []
        for batch in inputs.tolist():
            batch_rows = []
            for token in batch:
                predicted = self.next_token.get(int(token), 0)
                batch_rows.append(
                    [
                        10.0 if candidate == predicted else 0.0
                        for candidate in range(self.vocab_size)
                    ]
                )
            rows.append(batch_rows)
        return mx.array(rows, dtype=mx.float32)

    def begin_speculative_cache(self, cache, width):
        self.transaction_active = True
        return _FakeTransaction(cache[0], width)

    def resolve_speculative_cache(self, transaction, consumed):
        if transaction.cache.offset != transaction.initial_offset + transaction.width:
            raise AssertionError("target cache did not advance by the draft width")
        transaction.cache.offset = transaction.initial_offset + consumed
        transaction.active = False
        self.transaction_active = False
        self.resolve_calls.append(consumed)

    def cancel_speculative_cache(self, transaction):
        transaction.cache.offset = transaction.initial_offset
        transaction.active = False
        self.transaction_active = False
        self.cancel_calls += 1


class _Detokenizer:
    def __init__(self, tokenizer):
        self.reset()

    def reset(self):
        self.text = ""
        self.tokens = []
        self._last_segment = ""

    def add_token(self, token):
        self.tokens.append(token)
        self._last_segment = str(token)
        self.text += self._last_segment

    @property
    def last_segment(self):
        segment = self._last_segment
        self._last_segment = ""
        return segment

    def finalize(self):
        pass


def _tokenizer():
    tokenizer = object.__new__(TokenizerWrapper)
    tokenizer._detokenizer_class = _Detokenizer
    tokenizer._eos_token_ids = set()
    return tokenizer


def _lookup_generator(
    target,
    target_cache,
    *,
    prompt=(1, 2, 3, 4, 1, 2),
    max_tokens=5,
    prompt_lookup_num_tokens=3,
    prompt_lookup_history=None,
    stats=None,
):
    return speculative_generate_step(
        prompt=mx.array(prompt, dtype=mx.uint32),
        model=target,
        prompt_lookup_num_tokens=prompt_lookup_num_tokens,
        prompt_lookup_history=prompt_lookup_history,
        max_tokens=max_tokens,
        prompt_cache=[target_cache],
        speculative_round_callback=None if stats is None else stats.append,
    )


class PromptLookupDrafterTest(unittest.TestCase):
    def test_cli_flags_enable_prompt_lookup_and_round_stats(self):
        args = setup_arg_parser().parse_args(
            [
                "--prompt-lookup-num-tokens",
                "6",
                "--prompt-lookup-max-ngram-size",
                "5",
                "--speculative-round-stats",
            ]
        )

        self.assertEqual(args.prompt_lookup_num_tokens, 6)
        self.assertEqual(args.prompt_lookup_max_ngram_size, 5)
        self.assertTrue(args.speculative_round_stats)

    def test_longest_suffix_match_returns_historical_continuation(self):
        drafter = PromptLookupDrafter([1, 2, 3, 4, 1, 2])

        self.assertEqual(drafter.draft(3), [3, 4, 1])

    def test_no_match_returns_no_draft(self):
        drafter = PromptLookupDrafter([1, 2, 3])

        self.assertEqual(drafter.draft(4), [])


class PromptLookupGenerationTest(unittest.TestCase):
    def test_matching_drafts_are_accepted_and_reported(self):
        target = _PatternModel()
        target_cache = _FakeCache()
        stats = []

        outputs = list(
            _lookup_generator(
                target,
                target_cache,
                max_tokens=3,
                stats=stats,
            )
        )

        self.assertEqual([token for token, _, _ in outputs], [3, 4, 1])
        self.assertTrue(all(from_draft for _, _, from_draft in outputs))
        self.assertEqual(target.resolve_calls, [4])
        self.assertEqual(len(stats), 1)
        self.assertEqual(stats[0].source, "prompt_lookup")
        self.assertEqual(stats[0].drafted_tokens, 3)
        self.assertEqual(stats[0].accepted_tokens, 3)
        self.assertEqual(stats[0].committed_tokens, 3)
        self.assertEqual(stats[0].target_cache_tokens, 4)
        self.assertFalse(stats[0].cancelled)

    def test_no_match_falls_back_to_one_target_token(self):
        target = _PatternModel()
        target_cache = _FakeCache()
        stats = []

        outputs = list(
            _lookup_generator(
                target,
                target_cache,
                prompt=(1, 2, 3),
                max_tokens=1,
                stats=stats,
            )
        )

        self.assertEqual([token for token, _, _ in outputs], [4])
        self.assertFalse(outputs[0][2])
        self.assertEqual(target.resolve_calls, [])
        self.assertEqual(stats[0].drafted_tokens, 0)
        self.assertEqual(stats[0].accepted_tokens, 0)
        self.assertEqual(stats[0].committed_tokens, 1)
        self.assertEqual(stats[0].target_cache_tokens, 1)

    def test_consumer_close_commits_only_the_emitted_match(self):
        target = _PatternModel()
        target_cache = _FakeCache()
        stats = []
        generator = _lookup_generator(target, target_cache, stats=stats)

        token, _, from_draft = next(generator)
        self.assertEqual(token, 3)
        self.assertTrue(from_draft)
        generator.close()

        self.assertEqual(target.resolve_calls, [2])
        self.assertEqual(target.cancel_calls, 0)
        self.assertEqual(target_cache.offset, 7)
        self.assertEqual(stats[0].accepted_tokens, 1)
        self.assertEqual(stats[0].committed_tokens, 1)
        self.assertEqual(stats[0].target_cache_tokens, 2)
        self.assertFalse(stats[0].cancelled)

    def test_target_error_cancels_the_transaction(self):
        target = _PatternModel(fail_wide=True)
        target_cache = _FakeCache()
        stats = []
        generator = _lookup_generator(target, target_cache, stats=stats)

        with self.assertRaisesRegex(RuntimeError, "injected"):
            next(generator)

        self.assertEqual(target.resolve_calls, [])
        self.assertEqual(target.cancel_calls, 1)
        self.assertEqual(target_cache.offset, 5)
        self.assertEqual(stats[0].drafted_tokens, 3)
        self.assertEqual(stats[0].committed_tokens, 0)
        self.assertEqual(stats[0].target_cache_tokens, 0)
        self.assertTrue(stats[0].cancelled)

    def test_max_tokens_bounds_drafts_and_output(self):
        target = _PatternModel()
        target_cache = _FakeCache()
        stats = []

        outputs = list(
            _lookup_generator(
                target,
                target_cache,
                max_tokens=2,
                prompt_lookup_num_tokens=7,
                stats=stats,
            )
        )

        self.assertEqual(len(outputs), 2)
        self.assertEqual(stats[0].drafted_tokens, 2)
        self.assertEqual(stats[0].accepted_tokens, 2)
        self.assertEqual(stats[0].committed_tokens, 2)

        empty_stats = []
        empty = list(
            _lookup_generator(
                _PatternModel(),
                _FakeCache(),
                max_tokens=0,
                stats=empty_stats,
            )
        )
        self.assertEqual(empty, [])
        self.assertEqual(empty_stats, [])

    def test_stream_generate_reaches_prompt_lookup_without_a_draft_model(self):
        target = _PatternModel()
        stats = []

        responses = list(
            stream_generate(
                target,
                _tokenizer(),
                mx.array([1, 2, 3, 4, 1, 2], dtype=mx.uint32),
                max_tokens=2,
                prompt_lookup_num_tokens=2,
                speculative_round_callback=stats.append,
                prompt_cache=[_FakeCache()],
            )
        )

        self.assertEqual(responses[-1].generation_tokens, 2)
        self.assertTrue(stats)
        self.assertEqual(stats[0].source, "prompt_lookup")
        self.assertEqual(stats[0].drafted_tokens, 2)

    def test_lookup_history_seeds_cached_prefix_without_refill(self):
        target = _PatternModel()
        target_cache = _FakeCache()
        target_cache.offset = 4
        stats = []

        outputs = list(
            _lookup_generator(
                target,
                target_cache,
                prompt=(1, 2),
                prompt_lookup_history=(1, 2, 3, 4, 1, 2),
                max_tokens=2,
                prompt_lookup_num_tokens=2,
                stats=stats,
            )
        )

        self.assertEqual([token for token, _, _ in outputs], [3, 4])
        self.assertTrue(all(from_draft for _, _, from_draft in outputs))
        self.assertEqual(stats[0].drafted_tokens, 2)
        self.assertEqual(stats[0].accepted_tokens, 2)

    def test_lookup_history_fails_closed_on_inconsistent_context(self):
        with self.assertRaisesRegex(ValueError, "must end with"):
            next(
                _lookup_generator(
                    _PatternModel(),
                    _FakeCache(),
                    prompt=(1, 2),
                    prompt_lookup_history=(1, 2, 3, 4),
                )
            )

        with self.assertRaisesRegex(ValueError, "precomputed prompt_cache"):
            next(
                speculative_generate_step(
                    prompt=mx.array([1, 2], dtype=mx.uint32),
                    model=_PatternModel(),
                    prompt_lookup_num_tokens=2,
                    prompt_lookup_history=(1, 2, 3, 4, 1, 2),
                )
            )

        with self.assertRaisesRegex(ValueError, "requires prompt-lookup"):
            next(
                stream_generate(
                    _PatternModel(),
                    _tokenizer(),
                    mx.array([1, 2], dtype=mx.uint32),
                    max_tokens=1,
                    prompt_lookup_history=(1, 2),
                    prompt_cache=[_FakeCache()],
                )
            )


if __name__ == "__main__":
    unittest.main()
