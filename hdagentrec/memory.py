from __future__ import annotations

from .schemas import Intent, InteractionEvidence, InteractionType, MemoryUpdate, Preference, UserState


def _upsert(values: list[Preference], concept: str, item_id: int, step: int, confidence: float) -> None:
    for preference in values:
        if preference.concept == concept:
            preference.strength = min(1.0, preference.strength + confidence * .2)
            preference.confidence = max(preference.confidence, confidence)
            preference.last_seen, preference.evidence_items = step, (preference.evidence_items + [item_id])[-20:]
            return
    values.append(Preference(concept=concept, strength=confidence, confidence=confidence, first_seen=step, last_seen=step, evidence_items=[item_id]))


def apply_memory_update(state: UserState, update: MemoryUpdate, item_id: int, step: int, metadata="") -> UserState:
    """Apply a proposed operation deterministically; an LLM never mutates state directly."""
    output = state.model_copy(deep=True)
    output.evidence = (output.evidence + [InteractionEvidence(item_id=item_id, step=step, metadata=metadata)])[-100:]
    concept = update.concept or metadata or f"item:{item_id}"
    if update.interaction_type == InteractionType.preference_shift:
        for preference in output.long_term_preferences:
            if preference.concept != concept: preference.strength *= .9
    if update.update_long_term and update.interaction_type != InteractionType.noise: _upsert(output.long_term_preferences, concept, item_id, step, update.confidence)
    if update.update_short_term and update.interaction_type != InteractionType.noise:
        _upsert(output.short_term_preferences, concept, item_id, step, update.confidence)
        output.short_term_preferences.sort(key=lambda x: x.last_seen, reverse=True); output.short_term_preferences = output.short_term_preferences[:10]
    if update.interaction_type == InteractionType.uncertain: _upsert(output.uncertain_preferences, concept, item_id, step, update.confidence)
    if update.update_intent and update.intent_description:
        output.current_intent = Intent(description=update.intent_description, confidence=update.confidence, expected_duration=update.expected_duration, evidence_items=[item_id])
    output.last_update_step = step
    return output
