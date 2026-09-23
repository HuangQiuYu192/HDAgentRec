from __future__ import annotations

import time
import json
import torch
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, TypeVar

from pydantic import BaseModel

from .cache import SQLiteCache
from .memory import apply_memory_update
from .schemas import MemoryUpdate, RerankResponse, UserState

T = TypeVar("T", bound=BaseModel)


@dataclass
class LLMUsage:
    calls: int = 0; tokens: int = 0; latency_seconds: float = 0.0


class LLMClient(ABC):
    def __init__(self, model_name: str, cache: Optional[SQLiteCache] = None): self.model_name, self.cache, self.usage = model_name, cache, LLMUsage()
    @abstractmethod
    def _generate(self, prompt: str) -> tuple[str, int]: ...
    def generate_json(self, prompt: str, schema: type[T], prompt_version: str) -> T:
        key = self.cache.key(self.model_name, prompt_version, prompt) if self.cache else None
        if key and (cached := self.cache.get(key)) is not None: return schema.model_validate_json(cached)
        started = time.perf_counter(); raw, tokens = self._generate(prompt)
        self.usage.calls += 1; self.usage.tokens += tokens; self.usage.latency_seconds += time.perf_counter() - started
        raw = _first_json_object(raw)
        if schema is RerankResponse:
            # Qwen can repeat a supplied ID despite the prompt. Keep the first
            # occurrence deterministically, then let Pydantic validate types.
            payload = json.loads(raw)
            if isinstance(payload.get("ranking"), list):
                payload["ranking"] = list(dict.fromkeys(payload["ranking"]))
            raw = json.dumps(payload)
        result = schema.model_validate_json(raw)
        if key: self.cache.set(key, result.model_dump_json())
        return result


def _first_json_object(raw: str) -> str:
    """Accept harmless model labels/code fences while rejecting non-object replies."""
    start = raw.find("{")
    if start < 0:
        raise ValueError("LLM response contains no JSON object")
    try:
        _value, end = json.JSONDecoder().raw_decode(raw[start:])
    except json.JSONDecodeError as error:
        raise ValueError("LLM response contains invalid JSON") from error
    return raw[start : start + end]


class TransformersLLMClient(LLMClient):
    def __init__(self, model_name: str, cache: Optional[SQLiteCache] = None, device_map="auto"):
        super().__init__(model_name, cache); self.device_map, self._model, self._tokenizer = device_map, None, None
    def _generate(self, prompt: str) -> tuple[str, int]:
        if self._model is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, local_files_only=True)
            self._model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype="auto", device_map=self.device_map, local_files_only=True)
        messages = [{"role": "user", "content": prompt}]
        rendered = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = self._tokenizer([rendered], return_tensors="pt").to(self._model.device)
        with torch.inference_mode():
            output = self._model.generate(**inputs, max_new_tokens=256, do_sample=False)
        generated = output[:, inputs.input_ids.shape[1]:]
        text = self._tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
        return text, int(generated.shape[1])


class DynamicUserAgent:
    def __init__(self, client: LLMClient): self.client, self.invalid_reranks = client, 0
    def update_state(self, state: UserState, item_id: int, step: int, metadata: str, prompt: str) -> UserState:
        return apply_memory_update(state, self.client.generate_json(prompt, MemoryUpdate, "dynamic_state_v1"), item_id, step, metadata)
    def rerank(self, candidate_ids: list[int], prompt: str) -> list[int]:
        ranking = self.client.generate_json(prompt, RerankResponse, "reranker_v1").ranking
        valid = list(dict.fromkeys(item for item in ranking if item in candidate_ids))
        if len(valid) != len(ranking):
            self.invalid_reranks += 1
        return valid + [candidate for candidate in candidate_ids if candidate not in valid]
    @property
    def cost_metrics(self):
        u = self.client.usage
        return {"llm_calls": u.calls, "tokens": u.tokens, "avg_latency": u.latency_seconds / max(u.calls, 1), "invalid_reranks": self.invalid_reranks}
