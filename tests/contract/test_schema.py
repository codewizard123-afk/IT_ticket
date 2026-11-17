"""The JSON Schema, the Pydantic models and the golden fixtures must agree (README §2, §8.1)."""

from __future__ import annotations

import copy

import pytest
from jsonschema import Draft202012Validator

from shared.constants import CONTRACT_KEYS
from shared.schemas import AnalysisContract, AnalysisResponse

DEMO_IDS = [f"DEMO-00{i}" for i in range(1, 8)]


@pytest.fixture(scope="module")
def validator(contract_schema: dict) -> Draft202012Validator:
    Draft202012Validator.check_schema(contract_schema)
    return Draft202012Validator(contract_schema)


def _errors(validator: Draft202012Validator, doc: dict) -> list[str]:
    return [f"{list(e.path)}: {e.message}" for e in validator.iter_errors(doc)]


def test_all_seven_demos_have_golden_fixtures(golden: dict) -> None:
    assert sorted(golden) == DEMO_IDS


@pytest.mark.parametrize("demo", DEMO_IDS)
def test_golden_full_view_validates(
    validator: Draft202012Validator, golden: dict, demo: str
) -> None:
    assert _errors(validator, golden[demo]) == []
    AnalysisResponse.model_validate(golden[demo])


@pytest.mark.parametrize("demo", DEMO_IDS)
def test_contract_view_has_exactly_the_8_keys(
    validator: Draft202012Validator, golden: dict, demo: str
) -> None:
    contract = AnalysisResponse.model_validate(golden[demo]).to_contract().model_dump()
    assert tuple(contract) == CONTRACT_KEYS
    assert _errors(validator, contract) == []


def test_schema_and_pydantic_agree_on_the_contract_keys(contract_schema: dict) -> None:
    assert tuple(contract_schema["required"]) == CONTRACT_KEYS
    assert tuple(AnalysisContract.model_fields) == CONTRACT_KEYS
    assert set(contract_schema["properties"]) == {*CONTRACT_KEYS, "details"}


# --- negative cases: the schema must reject contract violations ------------------------------
def _mutate(doc: dict, path: list, value: object = None, delete: bool = False) -> dict:
    out = copy.deepcopy(doc)
    node = out
    for key in path[:-1]:
        node = node[key]
    if delete:
        del node[path[-1]]
    else:
        node[path[-1]] = value
    return out


@pytest.mark.parametrize(
    ("path", "value", "delete"),
    [
        (["extra_top_level_key"], 1, False),
        (["classification", "extra"], "x", False),
        (["classification", "confidence"], 1.5, False),
        (["priority", "level"], "P2", False),
        (["kedb_match", "error_code"], "VPN_001", False),
        (["kedb_match", "found"], False, False),  # found=false but error_code still set
        (["recommended_solution", "source"], "LLM", False),
        (["recommended_solution", "source"], "None", False),  # "None" but steps present
        (["similar_incidents", 0, "similarity_score"], -0.1, False),
        (["similar_incidents", 0, "resolution"], None, True),
        (["escalation_required"], "false", False),
        (["human_approval_required"], None, True),
        (["details", "decision"], "maybe", False),
        (["details", "steps", 0, "citations"], [], False),
    ],
)
def test_schema_rejects_violations(
    validator: Draft202012Validator, golden: dict, path: list, value: object, delete: bool
) -> None:
    bad = _mutate(golden["DEMO-001"], path, value, delete)
    assert _errors(validator, bad), f"schema accepted a violation at {path}"


def test_schema_accepts_found_false_with_nulls(
    validator: Draft202012Validator, golden: dict
) -> None:
    doc = copy.deepcopy(golden["DEMO-001"])
    doc["kedb_match"] = {"found": False, "error_code": None, "solution": None}
    doc["recommended_solution"] = {"steps": [], "source": "None"}
    assert _errors(validator, doc) == []
