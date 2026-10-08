"""Regression for the comment-only false rejection and preserved runtime bans."""
import tempfile
import unittest
from pathlib import Path

from common import DOMAINS, HERE
from driver import candidate_restriction


class ValidationTests(unittest.TestCase):
    def test_comments_and_documentation_are_not_operations(self):
        source = '''"""Transform requests without socket or filesystem I/O."""
# requests, open(...), subprocess are prohibited at runtime.
class AgentHarness:
    def __init__(self, public_context=None):
        """Public requests may contain text."""
        pass
    def transform_request(self, request):
        return request
    def transform_response(self, response, history):
        return response
'''
        self.assertIsNone(candidate_restriction(source))

    def test_executable_imports_names_and_hidden_data_strings_still_blocked(self):
        for source in ('import requests\n', 'from socket import socket\n',
                       'import subprocess\n', 'x = "tasks.json"\n',
                       'x = "published_h"\n', 'requests.get("url")\n',
                       'x = open("a")\n', 'x = os.environ\n'):
            with self.subTest(source=source):
                self.assertIsNotNone(candidate_restriction(source))

    def test_historical_ast_restrictions_are_unchanged(self):
        from evaluator import core
        for statement in ('import requests', 'import os', 'eval("1")',
                          'getattr.__globals__', '__import__("socket")'):
            with self.subTest(statement=statement), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'candidate.py'
                path.write_text(statement + '\n')
                with self.assertRaises(ValueError):
                    core.validate_plugin(path)

    def test_existing_candidates_in_all_domains_pass_without_byte_changes(self):
        from evaluator import core
        for domain in DOMAINS:
            for path in (HERE / domain / 'workspace/agents').glob('*.py'):
                if path.name == '__init__.py':
                    continue
                with self.subTest(domain=domain, candidate=path.name):
                    before = path.read_bytes()
                    self.assertIsNone(candidate_restriction(before.decode()))
                    core.validate_plugin(path)
                    self.assertEqual(path.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
