"""Offline tests for the Azure stage 1 run (ai, work) and the Azure OpenAI client."""

import json
import unittest
from unittest.mock import MagicMock, patch

from amc.classification import ai_work
from amc.classification.azure import AzureClient, AzureError

WORK = {"ai": "no", "work": "yes"}


def response(status, payload=None, text="", headers=None):
    mock = MagicMock(status_code=status, text=text, headers=headers or {})
    mock.json.return_value = payload
    return mock


def answer(labels, finish="stop"):
    content = json.dumps(labels)
    return response(200, {"choices": [{"finish_reason": finish, "message": {"content": content}}]})


class ValidateTests(unittest.TestCase):
    def test_two_labels(self):
        self.assertEqual(ai_work.validate(WORK), WORK)
        for labels in ({**WORK, "work": "none"}, {"ai": "yes"}, {**WORK, "sector": "it"}):
            with self.assertRaises(ValueError, msg=labels):
                ai_work.validate(labels)


class AzureClientTests(unittest.TestCase):
    def client(self, **options):
        return AzureClient("https://r.openai.azure.com/", "key", sleeper=lambda _: None, **options)

    def test_missing_settings(self):
        with self.assertRaises(AzureError):
            AzureClient("", "key")

    @patch("amc.classification.azure.requests.post")
    def test_v1_request_and_retry(self, post):
        post.side_effect = [response(429, headers={"retry-after": "1"}), answer(WORK)]
        labels = ai_work.AzureClassifier(self.client()).classify("tekst", "gpt-6-luna")
        self.assertEqual(labels, WORK)
        url = post.call_args.args[0]
        body = post.call_args.kwargs["json"]
        self.assertEqual(url, "https://r.openai.azure.com/openai/v1/chat/completions")
        self.assertEqual(body["model"], "gpt-6-luna")
        self.assertNotIn("temperature", body)
        post.side_effect = [answer(WORK)]
        self.client(temperature=0).complete("gpt-4o", "i", "t", {})
        self.assertEqual(post.call_args.kwargs["json"]["temperature"], 0)
        self.assertTrue(body["response_format"]["json_schema"]["strict"])
        self.assertEqual(body["messages"][1]["content"], "tekst")
        self.assertEqual(post.call_args.kwargs["headers"], {"api-key": "key"})

    @patch("amc.classification.azure.requests.post")
    def test_deployment_route_and_effort(self, post):
        post.return_value = answer(WORK)
        self.client(api_version="2025-04-01-preview", reasoning_effort="low").complete(
            "luna", "i", "t", {}
        )
        self.assertEqual(
            post.call_args.args[0],
            "https://r.openai.azure.com/openai/deployments/luna/chat/completions"
            "?api-version=2025-04-01-preview",
        )
        self.assertEqual(post.call_args.kwargs["json"]["reasoning_effort"], "low")

    @patch("amc.classification.azure.requests.post")
    def test_filtered_or_cut_answers_are_invalid(self, post):
        classifier = ai_work.AzureClassifier(self.client())
        for reply in (
            response(400, text='{"error":{"code":"content_filter"}}'),
            answer(WORK, finish="length"),
            response(200, {"choices": [{"finish_reason": "stop", "message": {"content": "x"}}]}),
        ):
            post.return_value = reply
            with self.assertRaises(ValueError):
                classifier.classify("t", "m")
        post.return_value = response(401)
        with self.assertRaises(AzureError):
            classifier.classify("t", "m")


if __name__ == "__main__":
    unittest.main()
