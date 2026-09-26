"""Tests for OmniParserGrounder (OmniParser detection + VLM element selection)."""

from __future__ import annotations

import base64
import io
import json
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("openadapt_grounding")

from openadapt_grounding.types import Element
from PIL import Image

from openadapt_evals.adapters.base import BenchmarkObservation, BenchmarkTask
from openadapt_evals.agents.omniparser_grounder import OmniParserGrounder


def _png_bytes(size=(100, 50)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, "white").save(buf, format="PNG")
    return buf.getvalue()


ELEMENTS = [
    Element(bounds=(0.0, 0.0, 0.2, 0.1), text="File", element_type="text"),
    Element(bounds=(0.5, 0.5, 0.2, 0.2), text="Save", element_type="icon"),
]
SOM_B64 = base64.b64encode(_png_bytes()).decode()


@pytest.fixture
def client():
    with patch(
        "openadapt_grounding.parsers.omniparser.OmniParserClient"
    ) as client_cls:
        instance = MagicMock()
        instance.parse_with_metadata.return_value = {
            "elements": ELEMENTS,
            "som_image_base64": SOM_B64,
        }
        client_cls.return_value = instance
        yield instance


@pytest.fixture
def obs():
    return BenchmarkObservation(screenshot=_png_bytes(), viewport=(100, 50))


def _task(instruction="Click the Save button"):
    return BenchmarkTask(task_id="t1", instruction=instruction, domain="desktop")


def test_clicks_center_of_selected_element(client, obs):
    grounder = OmniParserGrounder("http://localhost:8000")
    with patch("openadapt_evals.vlm.vlm_call", return_value="1") as vlm:
        action = grounder.act(obs, _task())

    assert action.type == "click"
    assert (action.x, action.y) == pytest.approx((0.6, 0.6))
    assert action.raw_action["element_index"] == 1
    assert action.raw_action["element_text"] == "Save"
    # The annotated SOM image is sent to the selector, not the raw screenshot.
    assert vlm.call_args.kwargs["images"] == [base64.b64decode(SOM_B64)]


def test_retries_once_on_invalid_reply(client, obs):
    grounder = OmniParserGrounder("http://localhost:8000")
    with patch("openadapt_evals.vlm.vlm_call", side_effect=["the save icon", "0"]) as vlm:
        action = grounder.act(obs, _task())

    assert vlm.call_count == 2
    assert "was not a valid index" in vlm.call_args.args[0]
    assert action.raw_action["element_index"] == 0


@pytest.mark.parametrize("replies", [["-1"], ["7", "9"], ["no", "idea"]])
def test_no_match_returns_wait(client, obs, replies):
    grounder = OmniParserGrounder("http://localhost:8000")
    with patch("openadapt_evals.vlm.vlm_call", side_effect=replies):
        action = grounder.act(obs, _task())

    assert action.type == "wait"
    assert action.raw_action["grounding_failed"] == "no_matching_element"


def test_omniparser_error_returns_wait(client, obs):
    client.parse_with_metadata.side_effect = ConnectionError("down")
    grounder = OmniParserGrounder("http://localhost:8000")
    action = grounder.act(obs, _task())

    assert action.type == "wait"
    assert action.raw_action["grounding_failed"] == "omniparser_error"


def test_no_screenshot_returns_wait(client):
    grounder = OmniParserGrounder("http://localhost:8000")
    action = grounder.act(BenchmarkObservation(screenshot=None), _task())

    assert action.type == "wait"
    assert action.raw_action["grounding_failed"] == "no_screenshot"


def test_no_elements_returns_wait(client, obs):
    client.parse_with_metadata.return_value = {"elements": [], "som_image_base64": None}
    grounder = OmniParserGrounder("http://localhost:8000")
    action = grounder.act(obs, _task())

    assert action.raw_action["grounding_failed"] == "no_elements"


def test_cache_dir_saves_som_and_elements(client, obs, tmp_path):
    grounder = OmniParserGrounder("http://localhost:8000", cache_dir=str(tmp_path))
    with patch("openadapt_evals.vlm.vlm_call", return_value="1"):
        grounder.act(obs, _task())

    pngs = list(tmp_path.glob("*.png"))
    jsons = list(tmp_path.glob("*.json"))
    assert len(pngs) == 1 and len(jsons) == 1
    records = json.loads(jsons[0].read_text())
    assert [r["text"] for r in records] == ["File", "Save"]
