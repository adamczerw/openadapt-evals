"""Tests for the Anthropic backend of openadapt_evals.vlm."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from openadapt_evals import vlm


def _create_kwargs(model: str) -> dict:
    """Call _vlm_call_anthropic with a mocked client, return messages.create kwargs."""
    client = MagicMock()
    client.messages.create.return_value = MagicMock(
        content=[MagicMock(text="ok")],
        usage=MagicMock(input_tokens=1, output_tokens=1),
    )
    with patch("anthropic.Anthropic", return_value=client), patch.object(
        vlm, "_track_response_cost"
    ):
        assert vlm._vlm_call_anthropic("hi", model=model, temperature=0.3) == "ok"
    return client.messages.create.call_args.kwargs


@pytest.mark.parametrize(
    "model",
    [
        "claude-opus-4-7",
        "claude-opus-4-8",
        "claude-opus-5",
        "claude-opus-5-5",
        "claude-sonnet-5",
        "claude-fable-5",
        "claude-fable-5-1",
        "claude-mythos-5-1",
    ],
)
def test_temperature_omitted_for_models_without_sampling_params(model):
    assert "temperature" not in _create_kwargs(model)


@pytest.mark.parametrize(
    "model",
    [
        "claude-opus-4-6",
        "claude-opus-4-5",
        "claude-sonnet-4-6",
        "claude-haiku-4-5-20251001",
    ],
)
def test_temperature_sent_for_older_models(model):
    assert _create_kwargs(model)["temperature"] == 0.3
