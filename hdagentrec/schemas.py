from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel as PydanticBaseModel, Field, validator


class BaseModel(PydanticBaseModel):
    """Small Pydantic v1/v2 compatibility shim for shared lab environments."""
    def model_copy(self, *, deep=False):
        return super().model_copy(deep=deep) if hasattr(super(), "model_copy") else self.copy(deep=deep)

    def model_dump(self):
        return super().model_dump() if hasattr(super(), "model_dump") else self.dict()

    def model_dump_json(self):
        return super().model_dump_json() if hasattr(super(), "model_dump_json") else self.json()

    @classmethod
    def model_validate_json(cls, raw):
        return super().model_validate_json(raw) if hasattr(super(), "model_validate_json") else cls.parse_raw(raw)


class Preference(BaseModel):
    concept: str
    strength: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    first_seen: int = Field(ge=0)
    last_seen: int = Field(ge=0)
    evidence_items: list[int] = Field(default_factory=list)


class Intent(BaseModel):
    description: str
    confidence: float = Field(ge=0.0, le=1.0)
    expected_duration: str = "unknown"
    evidence_items: list[int] = Field(default_factory=list)


class InteractionEvidence(BaseModel):
    item_id: int
    step: int = Field(ge=0)
    metadata: str = ""


class UserState(BaseModel):
    long_term_preferences: list[Preference] = Field(default_factory=list)
    short_term_preferences: list[Preference] = Field(default_factory=list)
    current_intent: Optional[Intent] = None
    rejected_preferences: list[Preference] = Field(default_factory=list)
    uncertain_preferences: list[Preference] = Field(default_factory=list)
    evidence: list[InteractionEvidence] = Field(default_factory=list)
    last_update_step: int = 0


class InteractionType(str, Enum):
    stable_preference = "stable_preference"
    temporary_intent = "temporary_intent"
    preference_shift = "preference_shift"
    exploration = "exploration"
    noise = "noise"
    uncertain = "uncertain"


class MemoryUpdate(BaseModel):
    interaction_type: InteractionType
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str
    concept: str = ""
    update_long_term: bool = False
    update_short_term: bool = False
    update_intent: bool = False
    intent_description: Optional[str] = None
    expected_duration: str = "unknown"


class RerankResponse(BaseModel):
    ranking: list[int]
    reasoning_summary: dict[str, str] = Field(default_factory=dict)

    @validator("ranking")
    @classmethod
    def unique_ids(cls, ranking: list[int]) -> list[int]:
        if len(ranking) != len(set(ranking)):
            raise ValueError("candidate ranking must not contain duplicate IDs")
        return ranking
