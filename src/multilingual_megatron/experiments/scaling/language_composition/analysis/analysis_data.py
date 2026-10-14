"""Read-only extraction of sealed study evidence into portable analysis inputs."""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from multilingual_megatron.experiments.scaling.stage1_fit.m6_composition import composition_sets
from multilingual_megatron.experiments.scaling.language_composition.study_contracts import EXPERTS, HORIZONS, SEEDS


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


class Reader:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.bindings = {}

    def path(self, path):
        path = Path(path)
        return path.resolve() if path.is_absolute() else (self.root / path).resolve()

    def bind(self, path, expected=None):
        path = self.path(path)
        key = str(path)
        if key not in self.bindings:
            self.bindings[key] = sha256(path)
        if expected is not None and self.bindings[key] != expected:
            raise ValueError(f"input hash mismatch: {path}")
        return path

    def read(self, path, expected=None):
        return json.loads(self.bind(path, expected).read_text())


def reference_jobs(manifest, reader):
    records = {}
    for ref in manifest["reference_reuse"]:
        result_path = reader.bind(ref["result"]["path"], ref["result"]["sha256"])
        evaluation_id = ref["source_job_id"].replace("_branch_", "_eval_")
        evaluation_path = result_path.parent.parent / evaluation_id / "evaluation_manifest.json"
        records[(ref["seed"], ref["experts"], ref["horizon_tokens"])] = evaluation_path
    if len(records) != 54:
        raise ValueError("reference checkpoint mapping is incomplete")
    return records


def extract_losses(manifest, summary, reader, spec):
    from multilingual_megatron.experiments.scaling.ab_v2.artifacts import _verified_receipt_evidence
    from multilingual_megatron.experiments.scaling.ab_v2.variant import variant_scope
    from multilingual_megatron.experiments.scaling.language_composition.training.training_workflow import VARIANT

    rows, parity = [], []
    root = reader.path(manifest["output_root"])
    references = reference_jobs(manifest, reader)
    jobs = {j["job_id"]: j for j in manifest["jobs"]}
    expected_languages = None
    for number, job_id in enumerate(sorted(summary["evaluation_evidence"])):
        with variant_scope(VARIANT):
            _verified_receipt_evidence(manifest, job_id)
        reader.bind(root / "control/receipts" / f"{job_id}.json")
        args = jobs[job_id]["arguments"]
        arm, seed, expert, horizon = (args[k] for k in ("composition_arm", "model_seed", "experts", "D"))
        path = root / "jobs" / job_id / "evaluation_manifest.json"
        value = reader.read(path)
        languages = set(value["languages"])
        if len(languages) != 196 or (expected_languages is not None and languages != expected_languages):
            raise ValueError("inconsistent K196 validation membership")
        expected_languages = languages
        old_path = references[(seed, expert, horizon)] if arm == "reference" else None
        old = reader.read(old_path) if old_path else None
        if old is not None and old["checkpoint"] != value["checkpoint"]:
            raise ValueError("reference reevaluation changed its checkpoint")
        for language, item in sorted(value["languages"].items()):
            artifact = Path(item["path"])
            artifact = artifact if artifact.is_absolute() else path.parent / artifact
            row = reader.read(artifact, item["sha256"])
            count = int(row["token_count"])
            if count <= 0 or not np.isclose(row["nll_sum"] / count, row["raw_nll"], rtol=0, atol=1e-12):
                raise ValueError("invalid scored-token count or raw loss")
            rows.append({"job_id": job_id, "arm": arm, "seed": seed, "E": expert, "horizon": horizon,
                "language": language, "raw_nll": row["raw_nll"],
                "adjusted_loss": row["atlas_tao_unigram_adjusted_loss"],
                "unigram_nll": -row["unigram_log_probability_sum"] / count,
                "token_count": count, "target_hash": row["target_token_ids_sha256"],
                "unigram_hash": row["unigram_counts_sha256"],
                "vocab_size": row["vocab_size"], "special_tokens": json.dumps(row["special_token_ids"])})
            if old is not None and language in spec["shared_languages"]:
                desc = old["languages"][language]
                source = Path(desc["path"])
                source = source if source.is_absolute() else old_path.parent / source
                previous = reader.read(source, desc["sha256"])
                if previous["target_token_ids_sha256"] != row["target_token_ids_sha256"] or previous["token_count"] != count:
                    raise ValueError("reference validation tokens changed")
                error = abs(previous["raw_nll"] - row["raw_nll"])
                if error > spec["reference_raw_loss_tolerance"]:
                    raise ValueError("reference raw-loss parity failed")
                parity.append({"seed": seed, "E": expert, "horizon": horizon, "language": language,
                    "raw_loss_difference": error,
                    "old_unigram_nll": -previous["unigram_log_probability_sum"] / count,
                    "new_unigram_nll": rows[-1]["unigram_nll"]})
        if (number + 1) % 27 == 0:
            print(f"Verified {number + 1}/162 evaluations", flush=True)
    if len(parity) != 378:
        raise ValueError("incomplete shared-language reference parity")
    return pd.DataFrame(rows), pd.DataFrame(parity)


def check_prefix(reference, candidate, count):
    """Require a common cumulative stream across ramp and cooldown boundaries."""
    if count <= 0 or len(reference) < count or len(candidate) < count:
        raise ValueError("invalid cumulative sample cutoff")
    for start in range(0, count, 1_000_000):
        stop = min(count, start + 1_000_000)
        if not np.array_equal(reference[start:stop], candidate[start:stop]):
            raise ValueError("training segments do not share a cumulative language stream")


def extract_exposures(manifest, reader, metadata):
    from multilingual_megatron.experiments.scaling.ab_v2.artifacts import _verified_receipt_evidence
    from multilingual_megatron.experiments.scaling.ab_v2.variant import variant_scope
    from multilingual_megatron.experiments.scaling.stage1_release.exposure import _exact_exposures
    from multilingual_megatron.experiments.scaling.language_composition.training.training_workflow import VARIANT

    root = reader.path(manifest["output_root"])
    output, support, checks = [], [], []
    for arm, seed, expert in itertools.product(("compact", "diverse"), SEEDS, EXPERTS):
        trunk = f"lc_trunk_{arm}_b_e{expert}_s{seed}"
        ids = [trunk] + [f"lc_branch_{arm}_b_e{expert}_d{d}_s{seed}" for d in HORIZONS]
        results = {}
        for job_id in ids:
            with variant_scope(VARIANT):
                _verified_receipt_evidence(manifest, job_id)
            reader.bind(root / "control/receipts" / f"{job_id}.json")
            result = reader.read(root / "jobs" / job_id / "result.json")
            if result["status"] != "complete" or result["training_health"]["errors"]:
                raise ValueError("unhealthy training receipt")
            results[job_id] = result
        terminal = results[ids[-1]]
        tier_path = reader.bind(terminal["data_manifest"], terminal["data_manifest_sha256"])
        languages = sorted(reader.read(tier_path)["prefixes_by_language"])
        members = composition_sets(metadata, languages=languages)
        if seed == SEEDS[0] and expert == EXPERTS[0]:
            for language, groups in members.items():
                support.append({"arm": arm, "language": language, **metadata[language],
                    "family_donors": list(groups["family"]), "script_donors": list(groups["script"]),
                    "other_donors": sorted(set(languages) - {language} - set(groups["family"]) - set(groups["script"]))})
        final_plan = reader.read(terminal["segment_plan"], terminal["segment_plan_sha256"])
        canonical_cache = reader.path(final_plan["segments"][-1]["output_dir"]) / "dataset_cache"
        def cache_files(directory):
            indices = list(directory.glob("*-BlendedDataset-train-dataset_index.npy"))
            descriptions = list(directory.glob("*-BlendedDataset-train-description.txt"))
            if len(indices) != 1 or len(descriptions) != 1:
                raise ValueError(f"missing or ambiguous retained blend cache: {directory}")
            return reader.bind(indices[0]), reader.bind(descriptions[0])
        canonical_index, canonical_description = cache_files(canonical_cache)
        reference = np.load(canonical_index, mmap_mode="r")
        canonical_paths = [v["dataset_path"] for v in reader.read(canonical_description)["datasets"]]
        for job_id, result in results.items():
            if result["data_manifest_sha256"] != terminal["data_manifest_sha256"]:
                raise ValueError("trajectory changed its training mixture")
            plan = reader.read(result["segment_plan"], result["segment_plan_sha256"])
            previous_end = None
            for segment in plan["segments"]:
                start, end = segment["actual_start_samples"], segment["actual_end_samples"]
                if previous_end is not None and start != previous_end:
                    raise ValueError("noncontiguous segment sample intervals")
                previous_end = end
                directory = reader.path(segment["output_dir"])
                audit = reader.read(directory / "checkpoint_state_audit.json")
                if audit.get("errors") or audit.get("consumed_train_samples") != end:
                    raise ValueError("checkpoint consumption disagrees with segment plan")
                index, description = cache_files(directory / "dataset_cache")
                if [v["dataset_path"] for v in reader.read(description)["datasets"]] != canonical_paths:
                    raise ValueError("dataset ordinal mapping changed within trajectory")
                check_prefix(reference, np.load(index, mmap_mode="r"), end)
                if segment["seq_length"] != 2048 or end * 2048 != segment["actual_end_tokens"]:
                    raise ValueError("segment exposure convention changed")
                checks.append({"job_id": job_id, "segment_index": segment["segment_index"],
                    "start_samples": start, "end_samples": end, "prefix_equal": True,
                    "index_sha256": reader.bindings[str(index)]})
            if job_id == trunk:
                if plan["segments"][0]["actual_start_samples"] != 0:
                    raise ValueError("trunk does not start at sample zero")
                continue
            branch_start = plan["segments"][0]["actual_start_samples"]
            parent_plan = reader.read(results[trunk]["segment_plan"])
            if branch_start not in {v["actual_end_samples"] for v in parent_plan["segments"]}:
                raise ValueError("branch does not resume from a recorded trunk boundary")
            count = int(result["expected_consumed_samples"])
            if previous_end != count or count * 2048 != result["actual_end_tokens"]:
                raise ValueError("endpoint consumption mismatch")
            exposure = _exact_exposures(index_path=canonical_index,
                description_path=canonical_description, tier_manifest_path=tier_path,
                end_samples=count, sequence_length=2048)
            for language in languages:
                group = members[language]
                target = exposure[language]
                family = sum(exposure[l] for l in group["family"])
                script = sum(exposure[l] for l in group["script"])
                other = result["actual_end_tokens"] - target - family - script
                if min(target, family, script, other) < 0 or target == 0:
                    raise ValueError("invalid exposure partition")
                output.append({"arm": arm, "seed": seed, "E": expert,
                    "horizon": result["arguments"]["D"], "language": language,
                    "D_total": result["actual_end_tokens"], "D_t": target,
                    "D_family": family, "D_script": script, "D_other": other,
                    "K": 12, "dataset_seed": result["arguments"]["dataset_random_seed"]})
        print(f"Verified cumulative training stream: {arm}/E{expert}/seed{seed}", flush=True)
    return pd.DataFrame(output), support, checks
