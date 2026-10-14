from __future__ import annotations

import json
import logging
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _ctx(root: Path, *, audit_cfg: dict | None = None, dry_run: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        active_stages=["audit"],
        args=SimpleNamespace(dry_run=dry_run),
        audit_cfg=audit_cfg or {"enabled": True},
        env={},
        executor=None,
        force_stages=set(),
        logger=logging.getLogger("test"),
        output_root=root / "out",
        repo_root=REPO_ROOT,
        state_path=root / "state" / "pipeline_state.json",
        tokenize_cfg={"backend": "megatron_gpt"},
        tokenizer_cfg={},
        tokenized_dir=root / "tokenized",
    )


def _write_megatron_manifest(path: Path, prefix: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "megatron_gpt",
        "prefixes": [str(prefix)],
        "languages": ["eng_Latn"],
        "documents_per_language": {"eng_Latn": 2},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


class AuditPipelineTest(unittest.TestCase):
    def test_audit_config_defaults_to_output_root_audit_report(self) -> None:
        from multilingual_megatron.pipeline.stages.audit.config import resolve_audit_config

        with tempfile.TemporaryDirectory() as tmp:
            ctx = _ctx(Path(tmp))
            cfg = resolve_audit_config(ctx)

        self.assertTrue(cfg.enabled)
        self.assertEqual(cfg.backend, "megatron_gpt")
        self.assertTrue(str(cfg.output_path).endswith("out/audit/tokenization_audit.json"))
        self.assertTrue(cfg.require_prefix_files)

    def test_single_megatron_manifest_passes_when_prefix_files_exist(self) -> None:
        from multilingual_megatron.pipeline.stages.audit.audit_stage import stage_audit

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ctx = _ctx(root)
            prefix = root / "tokenized" / "train" / "eng_text_document"
            prefix.parent.mkdir(parents=True, exist_ok=True)
            Path(str(prefix) + ".bin").write_bytes(b"bin")
            Path(str(prefix) + ".idx").write_bytes(b"idx")
            _write_megatron_manifest(root / "tokenized" / "megatron_data_manifest.json", prefix)

            stage_audit(ctx)

            report = json.loads((root / "out" / "audit" / "tokenization_audit.json").read_text())
            self.assertTrue(report["pass"])
            self.assertEqual(report["num_prefixes"], 1)
            self.assertEqual(report["missing_files"], [])

    def test_single_megatron_manifest_fails_when_prefix_file_missing(self) -> None:
        from multilingual_megatron.pipeline.stages.audit.audit_stage import stage_audit

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ctx = _ctx(root)
            prefix = root / "tokenized" / "train" / "eng_text_document"
            _write_megatron_manifest(root / "tokenized" / "megatron_data_manifest.json", prefix)

            with self.assertRaises(SystemExit):
                stage_audit(ctx)

    def test_multi_tokenizer_manifest_writes_aggregate_report(self) -> None:
        from multilingual_megatron.pipeline.stages.audit.audit_stage import stage_audit

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ctx = _ctx(root)
            tokenized_dir = root / "tokenized"
            tokenizers = {}
            for tokenizer_id in ("tok_a", "tok_b"):
                prefix = tokenized_dir / tokenizer_id / "train" / "eng_text_document"
                prefix.parent.mkdir(parents=True, exist_ok=True)
                Path(str(prefix) + ".bin").write_bytes(b"bin")
                Path(str(prefix) + ".idx").write_bytes(b"idx")
                data_manifest = tokenized_dir / tokenizer_id / "megatron_data_manifest.json"
                _write_megatron_manifest(data_manifest, prefix)
                tokenizers[tokenizer_id] = {
                    "tokenizer_id": tokenizer_id,
                    "tokenizer_model": f"{tokenizer_id}.model",
                    "vocab_size": 32000,
                    "data_manifest": str(data_manifest),
                }
            tokenized_dir.mkdir(parents=True, exist_ok=True)
            (tokenized_dir / "tokenized_manifests.json").write_text(
                json.dumps({"schema_version": 1, "tokenizers": tokenizers}),
                encoding="utf-8",
            )

            stage_audit(ctx)

            report = json.loads((root / "out" / "audit" / "tokenization_audit.json").read_text())
            self.assertTrue(report["pass"])
            self.assertEqual(report["tokenizer_count"], 2)
            self.assertEqual(sorted(report["tokenizers"].keys()), ["tok_a", "tok_b"])


if __name__ == "__main__":
    unittest.main()
