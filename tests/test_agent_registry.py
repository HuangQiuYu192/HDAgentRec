from hdagentrec.agentverse import DynamicUserRegistry
from hdagentrec.schemas import UserState


def test_registry_has_isolated_states_and_reset():
    registry = DynamicUserRegistry()
    registry.get(1).last_update_step = 3
    assert registry.get(2).last_update_step == 0
    registry.reset()
    assert registry.get(1) == UserState()
