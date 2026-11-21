"""Answer-model clients. Both talk only to the agentgate model proxy, with the run's token
where a provider key would go; the proxy decides, meters and adds the real key.

- Anthropic (Claude): a strict `submit_answer` tool with tool_choice auto. Claude Opus 5.5
  rejects forced tool choice, and agentgate's admission denies `output_config` (structured
  outputs, effort), so the tool is the way to get the JSON shape.
- OpenAI-compatible (DeepSeek): JSON mode (`response_format: json_object`).

Each request carries the question's audit event id as Panoptes' correlation id, so the
proxied exchange joins the question in the audit log.
"""

from typing import Protocol

import anthropic
import openai

from brain.answer.compose import SCHEMA, Prompt
from brain.gate.session import ModelSession
from brain.tenancy.config import Provider

CORRELATION_HEADER = "X-Panoptes-Correlation-Id"
TOOL_NAME = "submit_answer"
TIMEOUT_S = 180.0


class ModelError(RuntimeError):
    """The proxy or the provider refused or failed the call. Holds a status and a reason
    code, never a key or a prompt."""


class AnswerModel(Protocol):
    def generate(self, prompt: Prompt, correlation_id: str) -> str | dict[str, object] | None:
        """The raw reply (JSON text or the tool's input), or None if the model gave none."""
        ...


class AnthropicAnswerModel:
    def __init__(self, session: ModelSession) -> None:
        self._session = session
        # max_retries=0: a retry would reserve spend again for a call the proxy refused.
        self._client = anthropic.Anthropic(
            api_key=session.token, base_url=session.url, max_retries=0, timeout=TIMEOUT_S
        )

    def generate(self, prompt: Prompt, correlation_id: str) -> str | dict[str, object] | None:
        model = self._session.model
        try:
            response = self._client.messages.create(
                model=model.name,
                max_tokens=model.max_output_tokens,
                system=prompt.system + f"\n\nSubmit your reply by calling {TOOL_NAME}.",
                messages=[{"role": "user", "content": prompt.user}],
                tools=[
                    {
                        "name": TOOL_NAME,
                        "description": "Submit the cited answer to the question.",
                        "input_schema": SCHEMA,
                        "strict": True,
                    }
                ],
                tool_choice={"type": "auto"},
                extra_headers={CORRELATION_HEADER: correlation_id},
            )
        except anthropic.RateLimitError as exc:
            raise ModelError(f"model call refused (429 {_reason(exc.body)})") from None
        except anthropic.APIStatusError as exc:
            raise ModelError(f"model call failed ({exc.status_code} {_reason(exc.body)})") from None
        except anthropic.APIConnectionError:
            raise ModelError("could not reach the model proxy") from None
        if response.stop_reason in ("refusal", "max_tokens"):
            return None
        for block in response.content:
            if block.type == "tool_use" and block.name == TOOL_NAME:
                return dict(block.input) if isinstance(block.input, dict) else None
        texts = [block.text for block in response.content if block.type == "text"]
        return "".join(texts) or None


class OpenAICompatibleAnswerModel:
    def __init__(self, session: ModelSession) -> None:
        self._session = session
        self._client = openai.OpenAI(
            api_key=session.token, base_url=session.url + "/v1", max_retries=0, timeout=TIMEOUT_S
        )

    def generate(self, prompt: Prompt, correlation_id: str) -> str | dict[str, object] | None:
        model = self._session.model
        try:
            response = self._client.chat.completions.create(
                model=model.name,
                max_tokens=model.max_output_tokens,
                messages=[
                    {"role": "system", "content": prompt.system},
                    {"role": "user", "content": prompt.user},
                ],
                response_format={"type": "json_object"},
                extra_headers={CORRELATION_HEADER: correlation_id},
            )
        except openai.RateLimitError as exc:
            raise ModelError(f"model call refused (429 {_reason(exc.body)})") from None
        except openai.APIStatusError as exc:
            raise ModelError(f"model call failed ({exc.status_code} {_reason(exc.body)})") from None
        except openai.APIConnectionError:
            raise ModelError("could not reach the model proxy") from None
        if not response.choices:
            return None
        choice = response.choices[0]
        content = (choice.message.content or "").strip()
        # DeepSeek's JSON mode returns blank content when it could not produce JSON.
        if choice.finish_reason != "stop" or not content:
            return None
        return content


def answer_model(session: ModelSession) -> AnswerModel:
    if session.model.provider is Provider.ANTHROPIC:
        return AnthropicAnswerModel(session)
    return OpenAICompatibleAnswerModel(session)


def _reason(body: object) -> str:
    """The proxy's reason code from an error body, or the provider's error type."""
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, str):
            return error
        if isinstance(error, dict) and isinstance(error.get("type"), str):
            return str(error["type"])
    return "error"
