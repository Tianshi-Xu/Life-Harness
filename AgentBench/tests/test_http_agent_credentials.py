import json
import os
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from src.client.agents.http_agent import HTTPAgent
from src.configs import ConfigLoader


class HTTPAgentCredentialsTest(unittest.TestCase):
    reference = "Bearer ${env.MR_API_KEY}"
    fake_key = "synthetic-test-credential"

    def agent(self, headers=None):
        return HTTPAgent(
            url="https://example.invalid/v1/chat/completions",
            headers=headers if headers is not None else {"Authorization": self.reference},
            return_format="{response[choices][0][message][content]}",
        )

    def test_resolves_only_request_headers(self):
        original = {"Authorization": self.reference, "Content-Type": "application/json"}
        agent = self.agent(original)
        with patch.dict(os.environ, {"MR_API_KEY": self.fake_key}, clear=True):
            resolved = agent._request_headers()
        self.assertEqual(resolved["Authorization"], "Bearer " + self.fake_key)
        self.assertEqual(resolved["Content-Type"], "application/json")
        self.assertEqual(agent.headers, original)
        self.assertEqual(original["Authorization"], self.reference)
        self.assertNotIn(self.fake_key, json.dumps(agent.headers))

    def test_resolves_multiple_environment_references(self):
        agent = self.agent({"X-Test": "${env.FIRST}:${env.SECOND}"})
        with patch.dict(os.environ, {"FIRST": "one", "SECOND": "two"}, clear=True):
            self.assertEqual(agent._request_headers(), {"X-Test": "one:two"})

    def test_picks_up_environment_changes_without_mutation(self):
        agent = self.agent()
        for value in ("first-synthetic-value", "second-synthetic-value"):
            with self.subTest(value=value):
                with patch.dict(os.environ, {"MR_API_KEY": value}, clear=True):
                    self.assertEqual(agent._request_headers()["Authorization"], "Bearer " + value)
                self.assertEqual(agent.headers["Authorization"], self.reference)

    def test_missing_or_empty_credentials_fail_before_network(self):
        for environment in ({}, {"MR_API_KEY": ""}):
            for method in ("inference", "inference_openai"):
                with self.subTest(environment=environment, method=method):
                    with patch.dict(os.environ, environment, clear=True):
                        with patch("src.client.agents.http_agent.requests.post") as post:
                            with self.assertRaisesRegex(ValueError, "MR_API_KEY"):
                                getattr(self.agent(), method)([])
                            post.assert_not_called()

    def test_both_inference_paths_send_resolved_headers(self):
        payload = {
            "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            "usage": {"total_tokens": 1},
        }
        for method in ("inference", "inference_openai"):
            with self.subTest(method=method):
                response = Mock(status_code=200)
                response.json.return_value = payload
                agent = self.agent()
                with patch.dict(os.environ, {"MR_API_KEY": self.fake_key}, clear=True):
                    with patch("src.client.agents.http_agent.requests.post", return_value=response) as post:
                        result = getattr(agent, method)([{"role": "user", "content": "hello"}])
                self.assertEqual(post.call_args.kwargs["headers"]["Authorization"], "Bearer " + self.fake_key)
                self.assertEqual(agent.headers["Authorization"], self.reference)
                if method == "inference":
                    self.assertEqual(result, "ok")
                else:
                    self.assertEqual(result, (payload["choices"][0]["message"], payload["usage"]))

    def test_literal_headers_remain_compatible(self):
        original = {"Authorization": "Bearer EMPTY", "Content-Type": "application/json"}
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(self.agent(original)._request_headers(), original)

    def test_repository_config_remains_safe_to_serialize(self):
        path = Path(__file__).resolve().parents[1] / "configs" / "agents" / "api_agents.yaml"
        with patch.dict(os.environ, {"MR_API_KEY": self.fake_key}, clear=True):
            loader = ConfigLoader()
            config = loader.load_from(str(path))
            for name in ("claude-opus-4-8", "gemini-3.1-pro-preview", "gpt-5.5"):
                with self.subTest(agent=name):
                    agent = HTTPAgent(**config[name]["parameters"])
                    self.assertEqual(agent.headers["Authorization"], self.reference)
                    self.assertEqual(agent._request_headers()["Authorization"], "Bearer " + self.fake_key)
            local = HTTPAgent(**config["qwen3-4b-instruct"]["parameters"])
            self.assertEqual(local._request_headers()["Authorization"], "Bearer EMPTY")
            self.assertNotIn(self.fake_key, json.dumps(config))
            self.assertNotIn(self.fake_key, json.dumps(loader.loaded))


if __name__ == "__main__":
    unittest.main()
