"""Models deployed on Azure OpenAI (Azure AI Foundry), through the chat completions API.

The client only sends an instruction and a text with a JSON schema and returns the answer;
each labelling stage parses and validates it (for example `ai_work.AzureClassifier`).
Requires AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY. Without AZURE_OPENAI_API_VERSION
the version-free v1 API is used, where `model` is the deployment name; with it, the older
`/openai/deployments/{model}` route.
"""

import random
import time
from collections.abc import Callable

import requests


class AzureError(Exception):
    """A safe, human-readable failure suitable for console output."""


class AzureClient:
    def __init__(
        self,
        endpoint: str,
        api_key: str,
        *,
        api_version: str = "",
        max_tokens: int = 4000,
        reasoning_effort: str = "",
        temperature: float | None = None,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if not endpoint.strip() or not api_key.strip():
            raise AzureError(
                "Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY in .env before classifying."
            )
        self._endpoint = endpoint.strip().rstrip("/")
        self._api_version = api_version.strip()
        # No shared session: requests are sent from several threads.
        self._headers = {"api-key": api_key.strip()}
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.temperature = temperature
        self._sleeper = sleeper

    def _url(self, model: str) -> str:
        if self._api_version:
            return (
                f"{self._endpoint}/openai/deployments/{model}/chat/completions"
                f"?api-version={self._api_version}"
            )
        base = (
            self._endpoint
            if self._endpoint.endswith("/openai/v1")
            else (self._endpoint + "/openai/v1")
        )
        return base + "/chat/completions"

    def complete(self, model: str, instruction: str, text: str, schema: dict) -> str:
        """Raw JSON answer constrained by `schema`.

        Retries network failures, 429 and 5xx with growing waits. A text blocked by Azure's
        content filter or an answer cut at the token limit raises ValueError, so the run
        counts it as invalid and retries it next time.
        """
        body = {
            "model": model,
            # Reasoning models accept only the default temperature, so it is sent only when
            # set (0 for models such as gpt-4o, for reproducible labels).
            "max_completion_tokens": self.max_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "labels", "strict": True, "schema": schema},
            },
            "messages": [
                {"role": "system", "content": instruction},
                {"role": "user", "content": text},
            ],
        }
        if self.reasoning_effort:
            body["reasoning_effort"] = self.reasoning_effort
        if self.temperature is not None:
            body["temperature"] = self.temperature
        for attempt in range(8):
            try:
                response = requests.post(
                    self._url(model), json=body, headers=self._headers, timeout=180
                )
            except requests.RequestException:
                self._sleeper(min(5 * 2**attempt, 300))
                continue
            if response.status_code == 200:
                choice = response.json()["choices"][0]
                if choice.get("finish_reason") in ("length", "content_filter"):
                    raise ValueError(f"Answer stopped: {choice['finish_reason']}.")
                return choice["message"].get("content") or ""
            if response.status_code in (401, 403):
                raise AzureError(f"Azure OpenAI HTTP {response.status_code}: key rejected.")
            if response.status_code == 400 and "content_filter" in response.text:
                raise ValueError("The text was blocked by the Azure content filter.")
            if response.status_code == 404:
                raise AzureError(
                    f"Azure OpenAI HTTP 404: no deployment {model!r} at this endpoint."
                )
            if response.status_code != 429 and response.status_code < 500:
                raise AzureError(f"Azure OpenAI HTTP {response.status_code}: {response.text[:300]}")
            wait = response.headers.get("retry-after", "")
            delay = float(wait) if wait.replace(".", "", 1).isdigit() else 5 * 2**attempt
            self._sleeper(min(delay, 300) + random.uniform(0, 1))
        raise AzureError("Azure OpenAI failed repeatedly; try again later.")
