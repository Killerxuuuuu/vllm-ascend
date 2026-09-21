# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests; unittest avoids importing the vLLM/NPU pytest conftest."""

import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[3] / "benchmarks/scripts/collect_dsv4_indexer_client.py"
SPEC = importlib.util.spec_from_file_location("collect_dsv4_indexer_client", SCRIPT)
client = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(client)


class TestClient(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config = {
            "enable": 0,
            "acl_task_time": 3,
            "torch_prof_step_num": 8,
            "profiler_step_num": 512,
            "prof_dir": str(self.root / "raw"),
        }
        self.config_path = self.root / "config.json"
        self.config_path.write_text(json.dumps(self.config))

    def test_validate_short_capture(self):
        self.assertEqual(client.validate_config(self.config, 3, 16), 8)

    def test_reject_128_step_window_with_48_token_budget(self):
        self.config["torch_prof_step_num"] = 128
        with self.assertRaises(ValueError):
            client.validate_config(self.config, 3, 16)

    def test_reject_active_or_wrong_mode(self):
        for key, value in (("enable", 1), ("acl_task_time", 1), ("torch_prof_step_num", 0)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                client.validate_config({**self.config, key: value}, 3, 16)

    def test_enable_preserves_other_fields_and_detects_external_changes(self):
        client.set_enabled(self.config_path, self.config, 1)
        self.assertEqual(client.read_config(self.config_path), {**self.config, "enable": 1})
        client.set_enabled(self.config_path, self.config, 0)
        modified = {**self.config, "prof_dir": "another-capture"}
        self.config_path.write_text(json.dumps(modified))
        with self.assertRaises(RuntimeError):
            client.set_enabled(self.config_path, self.config, 1)
        self.assertEqual(client.read_config(self.config_path), modified)

    def test_lock_rejects_competing_client_and_releases_after_error(self):
        with self.assertRaisesRegex(ValueError, "injected"), client.config_lock(self.config_path):
            with self.assertRaises(RuntimeError), client.config_lock(self.config_path):
                self.fail("Should not acquire a second lock")
            raise ValueError("injected")
        self.assertFalse(self.config_path.with_name("config.json.client.lock").exists())

    def test_old_and_partial_metadata_not_counted(self):
        raw = self.root / "raw"
        raw.mkdir()
        old = raw / "profiler_info_old.json"
        old.write_text('{"complete": true}')
        baseline = client.info_files(raw)
        partial = raw / "profiler_info_new.json"
        partial.write_text('{"partial":')
        self.assertEqual(client.new_info_files(raw, baseline), {})
        partial.write_text('{"complete": true}')
        self.assertEqual(set(client.new_info_files(raw, baseline)), {partial})

    def test_frequency_trace_not_completion_evidence(self):
        raw = self.root / "raw"
        raw.mkdir()
        (raw / "trace_view.json").write_text('{"traceEvents": [{"ph": "C", "name": "AI Core Freq"}]}')
        self.assertEqual(client.info_files(raw), {})

    def test_listener_filters_port_address_and_state(self):
        net = self.root / "net"
        net.mkdir()
        (net / "tcp").write_text(
            "header\n"
            "0: 0100007F:1F40 00000000:0000 0A 0 0 0 1000 0 123\n"
            "1: 0100007F:1F41 00000000:0000 0A 0 0 0 1000 0 456\n"
            "2: 0100007F:1F40 00000000:0000 01 0 0 0 1000 0 789\n"
        )
        self.assertEqual(client.listener_inodes(self.root, 8000), {"123"})

    def test_process_stamp_handles_spaces_and_parentheses(self):
        (self.root / "stat").write_text("42 (worker (test)) " + " ".join(["S"] + ["0"] * 18 + ["98765"]))
        self.assertEqual(client.process_stamp(self.root), "98765")

    def test_model_must_be_served(self):
        models = {"data": [{"id": "current"}]}
        self.assertEqual(client.choose_model(models, None), "current")
        with self.assertRaises(ValueError):
            client.choose_model(models, "stale")

    def test_stale_client_directory_rejected_before_writes(self):
        args = client.parse_args(["--run-dir", str(self.root / "stale")])
        with (
            patch.object(client, "discover_server", return_value={"config_path": str(self.config_path)}),
            self.assertRaises(ValueError),
        ):
            client.collect(args)
        self.assertEqual(client.read_config(self.config_path), self.config)
        self.assertEqual(list(self.root.glob("client-*")), [])

    def test_discover_uses_server_environment(self):
        proc = self.root / "proc"
        process = proc / "42"
        descriptors = process / "fd"
        descriptors.mkdir(parents=True)
        (descriptors / "3").touch()
        (process / "environ").write_bytes((client.CONFIG_ENV + "=" + str(self.config_path)).encode() + bytes([0]))
        (process / "stat").write_text("42 (server) " + " ".join(["S"] + ["0"] * 18 + ["321"]))
        with (
            patch.object(client, "listener_inodes", return_value={"777"}),
            patch.object(client.os, "readlink", return_value="socket:[777]"),
            patch.object(client.os, "getuid", return_value=process.stat().st_uid, create=True),
            patch.dict(os.environ, {client.CONFIG_ENV: str(self.root / "stale.json")}),
        ):
            server = client.discover_server(8000, proc)
        self.assertEqual(server["pid"], 42)
        self.assertEqual(server["config_path"], str(self.config_path.resolve()))

    def test_replaced_server_is_rejected(self):
        with patch.object(client, "process_stamp", return_value="new"), self.assertRaises(RuntimeError):
            client.verify_server({"pid": 42, "starttime": "old"})

    def run_mock_collection(self, fail=False, produce_info=True):
        args = client.parse_args(["--flush-timeout", "0.02"])
        server = {"pid": 42, "starttime": "1", "cwd": str(self.root), "config_path": str(self.config_path)}
        phases = []

        def request(opener, base, actual_server, model, args, label, output, short=False):
            phases.append((label, client.read_config(self.config_path)["enable"]))
            if label == "collect-1":
                if fail:
                    raise RuntimeError("injected HTTP failure")
                if produce_info:
                    raw = self.root / "raw"
                    raw.mkdir()
                    (raw / "profiler_info_0.json").write_text('{"config": {"done": true}}')
            return {"label": label}

        with (
            patch.object(client, "discover_server", return_value=server),
            patch.object(client, "verify_server"),
            patch.object(client, "get_json", return_value={"data": [{"id": "current"}]}),
            patch.object(client, "send_request", side_effect=request),
            patch.object(client.time, "sleep"),
        ):
            status = client.collect(args)
        self.assertEqual(client.read_config(self.config_path)["enable"], 0)
        self.assertEqual(phases[0], ("warmup-1", 0))
        self.assertIn(("collect-1", 1), phases)
        self.assertEqual(phases[-1], ("stop-check", 0))
        summary = json.loads(next(self.root.glob("client-*/summary.json")).read_text(encoding="utf-8"))
        return status, summary

    def test_full_client_sequence_without_npu(self):
        status, summary = self.run_mock_collection()
        self.assertEqual(status, 0)
        self.assertEqual(summary["status"], "capture_exported_not_analyzed")

    def test_request_failure_turns_capture_off_and_writes_report(self):
        status, summary = self.run_mock_collection(fail=True)
        self.assertEqual(status, 1)
        self.assertEqual(summary["status"], "failed")
        self.assertIn("injected HTTP failure", summary["errors"][0])

    def test_no_export_does_not_report_success(self):
        status, summary = self.run_mock_collection(produce_info=False)
        self.assertEqual(status, 1)
        self.assertEqual(summary["status"], "failed")
        self.assertEqual(summary["profiler_info_files"], [])


if __name__ == "__main__":
    unittest.main()
