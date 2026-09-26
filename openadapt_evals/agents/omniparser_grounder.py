"""OmniParser-based grounder for PlannerGrounderAgent.

Uses OmniParser to detect all UI elements, then asks a VLM to select
the one matching the planner's instruction.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import re
import time
from pathlib import Path

from openadapt_evals.adapters.base import (
    BenchmarkAction,
    BenchmarkObservation,
    BenchmarkTask,
)
from openadapt_evals.agents.base import BenchmarkAgent

logger = logging.getLogger(__name__)


class OmniParserGrounder(BenchmarkAgent):
    """Grounder that uses OmniParser for element detection + VLM for selection.

    Compatible with PlannerGrounderAgent as the ``grounder`` argument.

    Flow:
        1. Parse screenshot with OmniParser → get all elements + SOM image
        2. Ask a VLM to pick the element index matching the planner's instruction
        3. Return a click action at the normalized center of the matched element

    Args:
        omniparser_url: URL of the OmniParser server (e.g. "http://localhost:8000"
            when using an SSH tunnel).
        selector_model: VLM model name used for element selection.
        selector_provider: Provider for the selector VLM ("anthropic" or "openai").
        timeout: OmniParser request timeout in seconds.
        cache_dir: If set, each parse result is saved here as
                   ``{timestamp}.png`` (annotated SOM image) and
                   ``{timestamp}.json`` (element descriptions).

    Example:
        grounder = OmniParserGrounder("http://localhost:8000")
        agent = PlannerGrounderAgent(
            planner="claude-sonnet-4-6",
            planner_provider="anthropic",
            grounder=grounder,
        )
    """

    def __init__(
        self,
        omniparser_url: str,
        selector_model: str = "claude-haiku-4-5-20251001",
        selector_provider: str = "anthropic",
        timeout: float = 180.0,
        cache_dir: str | None = None,
    ):
        from openadapt_grounding.parsers.omniparser import OmniParserClient

        self._client = OmniParserClient(omniparser_url, timeout=timeout)
        self._selector_model = selector_model
        self._selector_provider = selector_provider
        self._cache_dir = Path(cache_dir) if cache_dir else None
        if self._cache_dir:
            self._cache_dir.mkdir(parents=True, exist_ok=True)

    def act(
        self,
        observation: BenchmarkObservation,
        task: BenchmarkTask,
        history=None,
    ) -> BenchmarkAction:
        """Ground the planner's instruction to a click coordinate.

        ``task.instruction`` is the planner's ``target_description``.
        """
        if not observation.screenshot:
            logger.warning("No screenshot in observation")
            return self._grounding_failed("no_screenshot", task.instruction)

        from PIL import Image

        image = Image.open(io.BytesIO(observation.screenshot))

        try:
            result = self._client.parse_with_metadata(image)
        except Exception as exc:  # noqa: BLE001 - any parser failure -> no-op action
            logger.error("OmniParser call failed: %s", exc)
            return self._grounding_failed("omniparser_error", task.instruction)

        elements = result["elements"]
        som_b64 = result.get("som_image_base64")

        if self._cache_dir and som_b64:
            self._save_parse_result(som_b64, elements)

        if not elements:
            logger.warning("OmniParser returned no elements")
            return self._grounding_failed("no_elements", task.instruction)

        idx = self._select_element(task.instruction, elements, som_b64, observation.screenshot)
        if idx is None:
            logger.warning("Selector could not pick an element for %r", task.instruction)
            return self._grounding_failed("no_matching_element", task.instruction)

        element = elements[idx]
        cx, cy = element.center
        logger.info(
            "OmniParserGrounder: %r → element[%d] text=%r center=(%.3f, %.3f)",
            task.instruction, idx, element.text, cx, cy,
        )
        return BenchmarkAction(
            type="click",
            x=cx,
            y=cy,
            raw_action={
                "element_index": idx,
                "element_text": element.text,
                "element_type": element.element_type,
                "element_bounds": list(element.bounds),
            },
        )

    @staticmethod
    def _grounding_failed(reason: str, instruction: str) -> BenchmarkAction:
        """Return a no-op action so a grounding failure doesn't end the episode.

        ``wait`` is a no-op in adapters. Returning it (instead of ``done``,
        which runners treat as task completion) lets the planner take a fresh
        look at the screen on the next step.
        """
        return BenchmarkAction(
            type="wait",
            raw_action={"grounding_failed": reason, "instruction": instruction},
        )

    def _save_parse_result(self, som_b64: str, elements: list) -> None:
        """Save SOM image and element descriptions to cache_dir."""
        now = time.time()
        ts = time.strftime("%Y%m%d_%H%M%S", time.localtime(now)) + f"_{int(now * 1000) % 1000:03d}"
        stem = self._cache_dir / ts

        stem.with_suffix(".png").write_bytes(base64.b64decode(som_b64))

        records = [
            {
                "index": i,
                "text": el.text,
                "type": el.element_type,
                "bounds": list(el.bounds),
                "confidence": el.confidence,
            }
            for i, el in enumerate(elements)
        ]
        stem.with_suffix(".json").write_text(json.dumps(records, indent=2))

        logger.debug("Saved parse result to %s.{png,json}", stem)

    def _select_element(
        self,
        instruction: str,
        elements: list,
        som_b64: str | None,
        screenshot_bytes: bytes,
    ) -> int | None:
        """Ask a VLM to pick the element index matching the instruction."""
        from openadapt_evals.vlm import vlm_call

        lines = []
        for i, el in enumerate(elements):
            label = el.text or f"[{el.element_type}]"
            lines.append(f"{i}: {label}")
        element_list = "\n".join(lines)

        prompt = (
            f"Instruction: {instruction}\n\n"
            f"Detected UI elements (index: label):\n{element_list}\n\n"
            "The screenshot is annotated with numbered bounding boxes matching the list above. "
            "Which element index should be clicked to follow the instruction? "
            f"Reply with only one integer between 0 and {len(elements) - 1}, nothing else. "
            "If none of the elements matches the instruction, reply with exactly -1."
        )

        # Prefer the SOM image (already annotated with numbers) over raw screenshot
        img_bytes = base64.b64decode(som_b64) if som_b64 else screenshot_bytes

        for attempt in range(2):
            raw = vlm_call(
                prompt,
                images=[img_bytes],
                model=self._selector_model,
                provider=self._selector_provider,
                max_tokens=16,
                cost_label="omniparser_grounder",
            )

            match = re.search(r"-?\d+", raw.strip())
            if match:
                idx = int(match.group())
                if idx == -1:
                    logger.info(
                        "Selector reports no matching element for %r", instruction,
                    )
                    return None
                if 0 <= idx < len(elements):
                    return idx
                logger.warning(
                    "Index %d out of range (len=%d), attempt %d",
                    idx, len(elements), attempt + 1,
                )
            else:
                logger.warning(
                    "Could not parse element index from: %r, attempt %d",
                    raw, attempt + 1,
                )

            # Retry once with an explicit correction appended.
            prompt += (
                f"\n\nYour previous reply ({raw.strip()!r}) was not a valid index. "
                f"Reply with ONLY a single integer from 0 to {len(elements) - 1}, "
                "or -1 if no element matches."
            )

        return None
