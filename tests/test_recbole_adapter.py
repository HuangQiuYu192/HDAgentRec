import pytest


def test_recbole_adapter_is_importable_without_recbole():
    from hdagentrec.recbole_model import HDAgentSASRec
    # The fallback remains importable in a light unit-test environment.
    assert HDAgentSASRec is not None


@pytest.mark.skipif(__import__("importlib").util.find_spec("recbole") is None, reason="RecBole installed in experiment environment")
def test_recbole_adapter_subclasses_sasrec():
    from recbole.model.sequential_recommender import SASRec
    from hdagentrec.recbole_model import HDAgentSASRec
    assert issubclass(HDAgentSASRec, SASRec)
