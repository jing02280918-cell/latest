"""OpenAI-compatible model client for V-DriftBench v3.15 models in CoP."""

from __future__ import annotations

import os
import time
from typing import Dict, List

import openai

from language_models_sonnet import LanguageModel


class VDBOpenAI(LanguageModel):
    """Routes to vLLM or DeepSeek via OpenAI-compatible API."""

    API_RETRY_SLEEP = 10
    API_ERROR_OUTPUT = "$ERROR$"
    API_QUERY_SLEEP = 0.5
    API_MAX_RETRY = 5

    def __init__(
        self,
        model_name: str,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        extra_body: dict | None = None,
        timeout: float = 60.0,
        seed: int | None = None,
    ):
        self.model_name = model_name
        self.base_url = base_url
        self.api_key = api_key or "EMPTY"
        self.extra_body = extra_body
        self.timeout = timeout
        self.seed = seed

    def generate(
        self,
        conv: List[Dict],
        max_n_tokens: int,
        temperature: float,
        top_p: float,
    ) -> str:
        output = self.API_ERROR_OUTPUT
        client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
        )
        for _ in range(self.API_MAX_RETRY):
            try:
                kwargs = {
                    "model": self.model_name,
                    "messages": conv,
                    "max_tokens": max_n_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                }
                if self.seed is not None:
                    kwargs["seed"] = self.seed
                if self.extra_body:
                    kwargs["extra_body"] = self.extra_body
                response = client.chat.completions.create(**kwargs)
                output = response.choices[0].message.content or ""
                break
            except openai.OpenAIError as e:
                print(type(e), e)
                err = str(e).lower()
                if "maximum context length" in err or "input_tokens" in err:
                    break
                time.sleep(self.API_RETRY_SLEEP)
            time.sleep(self.API_QUERY_SLEEP)
        return output

    def batched_generate(
        self,
        convs_list: List,
        max_n_tokens: int,
        temperature: float,
        top_p: float = 1.0,
    ):
        return [
            self.generate(conv, max_n_tokens, temperature, top_p) for conv in convs_list
        ]


def build_draft_model(seed: int | None = None) -> VDBOpenAI:
    from vdb_config import VDB_DRAFT_API_KEY, VDB_DRAFT_BASE_URL, VDB_DRAFT_MODEL, VDB_LLM_SEED

    if seed is None:
        seed = VDB_LLM_SEED

    return VDBOpenAI(
        VDB_DRAFT_MODEL,
        base_url=VDB_DRAFT_BASE_URL,
        api_key=VDB_DRAFT_API_KEY,
        seed=seed,
    )


def _target_api_model(model: str) -> str:
    if model.lower() == "glm-5.3-flash":
        return "ZHIPU/GLM-5.3-Flash"
    return model


def build_target_model() -> VDBOpenAI:
    from vdb_config import VDB_TARGET_BASE_URL, VDB_TARGET_MODEL

    model = VDB_TARGET_MODEL
    api_model = _target_api_model(model)
    model_lc = api_model.lower()
    extra_body = None
    timeout = 60.0
    if model_lc.startswith("qwen"):
        api_key = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("VDB_API_KEY", "EMPTY")
        base_url = VDB_TARGET_BASE_URL or "https://dashscope.aliyuncs.com/compatible-mode/v1"
    elif api_model.startswith("ZHIPU/") or model_lc.startswith("zhipu/"):
        api_key = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("VDB_API_KEY", "EMPTY")
        base_url = VDB_TARGET_BASE_URL or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        effort = os.environ.get("VDB_GLM_REASONING_EFFORT", "low")
        extra_body = {"enable_thinking": True, "reasoning_effort": effort}
        timeout = float(os.environ.get("VDB_TARGET_TIMEOUT_S", "300"))
    elif model_lc.startswith("deepseek"):
        api_key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("VDB_API_KEY", "EMPTY")
        base_url = VDB_TARGET_BASE_URL or "https://api.deepseek.com"
        extra_body = {"thinking": {"type": "disabled"}}
    else:
        api_key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("VDB_API_KEY", "EMPTY")
        base_url = VDB_TARGET_BASE_URL

    return VDBOpenAI(
        api_model,
        base_url=base_url,
        api_key=api_key,
        extra_body=extra_body,
        timeout=timeout,
    )
