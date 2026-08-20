#!/usr/bin/env python3
"""CPU-only regression tests for the DFlash2 lab/report tools."""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import dflash2_lab
import dflash2_report


def sample(value: str) -> dict[str, str]:
    return {"text_sha256": hashlib.sha256(value.encode("utf-8")).hexdigest()}


def counted_sample(value: str, tokens: int) -> dict[str, object]:
    return {**sample(value), "usage": {"completion_tokens": tokens}}


class DFlash2ToolsTests(unittest.TestCase):
    def test_server_command_pins_draft_device(self) -> None:
        args = SimpleNamespace(
            server_bin=Path("llama-server"),
            model=Path("target.gguf"),
            draft_model=Path("dflash.gguf"),
            spec_n_max=3,
            devices="Vulkan1,Vulkan0",
            draft_devices="Vulkan0",
            split_mode="layer",
            ctx_size=4096,
            gpu_layers=99,
            parallel=1,
            server_extra="--flash-attn on",
        )

        command = dflash2_lab.build_server_command(args, 8089)

        self.assertEqual("Vulkan0", command[command.index("-devd") + 1])
        self.assertEqual("3", command[command.index("--spec-draft-n-max") + 1])
        self.assertEqual("1", command[command.index("-np") + 1])
        self.assertEqual(["--flash-attn", "on"], command[-2:])

    def test_stable_exact_matrix(self) -> None:
        prompts = ["alpha", "beta"]

        def client(prompt: str, _n_max: int) -> dict[str, str]:
            return sample(prompt)

        matrix = dflash2_lab.run_matrix(
            client,
            prompts,
            parallel=2,
            spec_n_max=7,
            serial_repeats=2,
            target_waves=2,
            spec_waves=2,
            identical_waves=2,
        )
        analysis = dflash2_report.analyze_payload({"prompts": prompts, **matrix})
        self.assertEqual([1, 1], analysis["summary"]["serial_spec_distinct"])
        self.assertEqual([True, True], analysis["summary"]["serial_spec_matches_target"])
        self.assertEqual("STABLE_AND_BIT_EXACT", analysis["findings"][0]["code"])

    def test_heterogeneous_batch_shape_classification(self) -> None:
        prompts = ["alpha", "beta"]
        payload = {
            "prompts": prompts,
            "phases": {
                "serial_target": {"waves": [[sample("a"), sample("b")]]},
                "serial_spec": {"waves": [[sample("a"), sample("b")]]},
                "heterogeneous_target": {
                    "waves": [[sample("a"), sample("b")], [sample("a"), sample("b")]]
                },
                "heterogeneous_spec": {
                    "waves": [[sample("a"), sample("b1")], [sample("a"), sample("b2")]]
                },
                "identical_target": {"waves": [[sample("a"), sample("a")]]},
                "identical_spec": {"waves": [[sample("a"), sample("a")]]},
            },
        }
        analysis = dflash2_report.analyze_payload(payload)
        codes = {finding["code"] for finding in analysis["findings"]}
        self.assertIn("HETEROGENEOUS_BATCH_SHAPE_SENSITIVITY", codes)
        self.assertNotIn("IDENTICAL_SPEC_MULTISLOT_INSTABILITY", codes)

    def test_identical_spec_multislot_classification(self) -> None:
        prompts = ["alpha", "beta"]
        payload = {
            "prompts": prompts,
            "phases": {
                "serial_target": {"waves": [[sample("a"), sample("b")]]},
                "serial_spec": {"waves": [[sample("a"), sample("b")]]},
                "heterogeneous_target": {"waves": [[sample("a"), sample("b")]]},
                "heterogeneous_spec": {"waves": [[sample("a"), sample("b")]]},
                "identical_target": {"waves": [[sample("a"), sample("a")]]},
                "identical_spec": {"waves": [[sample("a"), sample("a2")]]},
            },
        }
        analysis = dflash2_report.analyze_payload(payload)
        codes = {finding["code"] for finding in analysis["findings"]}
        self.assertIn("IDENTICAL_SPEC_MULTISLOT_INSTABILITY", codes)
        self.assertEqual(2, analysis["summary"]["identical_spec_distinct"])

    def test_max_token_boundary_classification(self) -> None:
        prompts = ["alpha"]
        payload = {
            "prompts": prompts,
            "phases": {
                "serial_target": {"waves": [[sample("a")]]},
                "serial_spec": {"waves": [[sample("a")]]},
                "heterogeneous_target": {"waves": [[sample("a")]]},
                "heterogeneous_spec": {"waves": [[sample("a")]]},
                "identical_target": {"waves": [[sample("a")]]},
                "identical_spec": {"waves": [[sample("a")]]},
                "max_token_boundaries": {
                    "cases": [
                        {"max_tokens": 7, "target": counted_sample("x", 7), "spec": counted_sample("x", 7)},
                        {"max_tokens": 8, "target": counted_sample("y", 8), "spec": counted_sample("z", 8)},
                    ]
                },
            },
        }
        analysis = dflash2_report.analyze_payload(payload)
        codes = {finding["code"] for finding in analysis["findings"]}
        self.assertIn("MAX_TOKEN_BOUNDARY_MISMATCH", codes)
        self.assertEqual([True, False], [item["passed"] for item in analysis["summary"]["max_token_boundaries"]])

    def test_server_log_parser(self) -> None:
        log = """
prompt eval time = 100.0 ms / 10 tokens (10.0 ms per token, 100.0 tokens per second)
       eval time = 200.0 ms / 20 tokens (10.0 ms per token, 100.0 tokens per second)
    eval time = 0.0 ms / 1 tokens (0.0 ms per token, 1000000.0 tokens per second)
draft acceptance rate = 0.75000 (15 accepted / 20 generated)
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "server.log"
            path.write_text(log, encoding="utf-8")
            parsed = dflash2_report.parse_server_log(path)
        self.assertEqual(2, parsed["request_count"])
        self.assertEqual(1, parsed["decode_sample_count"])
        self.assertEqual(1, parsed["decode_degenerate_count"])
        self.assertEqual(100.0, parsed["prompt_tps_mean"])
        self.assertEqual(100.0, parsed["decode_tps_mean"])
        self.assertEqual(0.75, parsed["acceptance_mean"])


if __name__ == "__main__":
    unittest.main()