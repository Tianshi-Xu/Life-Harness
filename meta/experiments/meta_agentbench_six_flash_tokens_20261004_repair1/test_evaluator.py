"""Regression for preserving original per-request retries, not whole-episode retries."""
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import requests


class RetryTests(unittest.TestCase):
    def make_session(self):
        import evaluator
        session = evaluator.Session.__new__(evaluator.Session)
        session.history = [{'role': 'user', 'content': 'public task'}]
        session.turn, session.calls, session.tools = 0, [], []
        session.harness = SimpleNamespace(transform_request=lambda x: x,
                                         transform_response=lambda x, h: x)
        return evaluator, session

    def responses(self):
        bad = requests.Response()
        bad.status_code, bad._content = 500, b'Internal Server Error'
        good = requests.Response()
        good.status_code = 200
        good._content = json.dumps({'choices': [{'message': {'role': 'assistant', 'content': 'OK'}}],
                                   'usage': {'total_tokens': 4}}).encode()
        return bad, good

    def test_transient_500_retried_and_audited_in_same_episode(self):
        evaluator, session = self.make_session()
        bad, good = self.responses()
        tokenizer = SimpleNamespace(apply_chat_template=lambda *a, **k: [1])
        with patch.object(evaluator, 'TOKENIZER', tokenizer), patch.object(evaluator.time, 'sleep'), \
             patch('requests.post', side_effect=[bad, good]) as post:
            session.sync_action()
        self.assertEqual(post.call_count, 2)
        self.assertEqual(len(session.calls), 1)
        self.assertEqual(session.calls[0]['transport_statuses'][0]['status'], 500)
        self.assertEqual(session.calls[0]['response']['choices'][0]['message']['content'], 'OK')
        self.assertEqual(session.calls[0]['request']['max_tokens'], 4096)

    def test_at_most_original_three_requests_and_error_is_not_hidden(self):
        evaluator, session = self.make_session()
        bad, _ = self.responses()
        tokenizer = SimpleNamespace(apply_chat_template=lambda *a, **k: [1])
        with patch.object(evaluator, 'TOKENIZER', tokenizer), patch.object(evaluator.time, 'sleep'), \
             patch('requests.post', return_value=bad) as post:
            with self.assertRaises(requests.HTTPError):
                session.sync_action()
        self.assertEqual(post.call_count, 3)
        self.assertEqual(len(session.calls[0]['transport_errors']), 3)
        self.assertNotIn('response', session.calls[0])


if __name__ == '__main__':
    unittest.main()
