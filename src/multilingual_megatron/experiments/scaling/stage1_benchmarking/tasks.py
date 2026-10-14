"""Pinned lm-eval task definitions for the Stage-1 benchmark suite."""

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .contracts import (
    BELEBELE_DATASET, BELEBELE_REVISION, BenchmarkProfile, COPA_DATASET,
    COPA_REVISION, GLOBAL_MMLU_DATASET, GLOBAL_MMLU_REVISION, HELLASWAG_DATASET,
    HELLASWAG_REVISION, OKAPI_DATASET, OKAPI_REVISION, PINNED_BENCHMARK_REVISIONS,
    UHURA_ARC_EASY_DATASET, UHURA_ARC_EASY_REVISION, XCOPA_DATASET, XCOPA_REVISION,
    XNLI_DATASET, XNLI_REVISION, XSTORYCLOZE_DATASET, XSTORYCLOZE_REVISION,
)

def _preprocess_hellaswag_text(text: str) -> str:
    text = text.strip().replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    return text.replace("  ", " ")


def _localized(value: Any, language: str) -> Any:
    if isinstance(value, dict):
        if language in value:
            return value[language]
        if "en" in value:
            return value["en"]
        raise ValueError(f"Okapi localized value has no {language!r} or 'en' key")
    if isinstance(value, list):
        return [_localized(item, language) for item in value]
    return value


def load_pinned_okapi_dataset(
    *,
    language: str,
    revision: str = OKAPI_REVISION,
    **_kwargs: Any,
) -> Any:
    """Load one pinned Okapi JSON file and repair localized dict-valued cells."""

    import datasets
    from huggingface_hub import hf_hub_download

    path = Path(
        hf_hub_download(
            repo_id=OKAPI_DATASET,
            repo_type="dataset",
            filename=f"data/{language}_validation.json",
            revision=revision,
        )
    )
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"invalid Okapi JSON payload: {path}")
    normalized: list[dict[str, Any]] = []
    for raw in rows:
        if not isinstance(raw, dict):
            raise ValueError(f"non-object Okapi row in {path}")
        row = dict(raw)
        for field in ("ctx_a", "ctx_b", "ctx", "activity_label", "endings"):
            row[field] = _localized(row[field], language)
        if not isinstance(row["endings"], list) or any(
            not isinstance(item, str) for item in row["endings"]
        ):
            raise ValueError(f"invalid normalized Okapi endings in {path}")
        normalized.append(row)
    return datasets.DatasetDict(
        {"validation": datasets.Dataset.from_list(normalized)}
    )


def process_okapi_docs(dataset: Any) -> Any:
    def _process(row: Mapping[str, Any]) -> dict[str, Any]:
        context = row["ctx_a"] + " " + row["ctx_b"].capitalize()
        return {
            "query": _preprocess_hellaswag_text(
                row["activity_label"] + ": " + context
            ),
            "choices": [
                _preprocess_hellaswag_text(item) for item in row["endings"]
            ],
            "gold": int(row["label"]),
        }

    return dataset.map(_process)


def okapi_task_config(language: str) -> dict[str, Any]:
    return {
        "task": f"stage1_okapi_hellaswag_{language}",
        "dataset_path": OKAPI_DATASET,
        "dataset_name": language,
        "custom_dataset": load_pinned_okapi_dataset,
        "dataset_kwargs": {"language": language, "revision": OKAPI_REVISION},
        "output_type": "multiple_choice",
        "training_split": None,
        "validation_split": "validation",
        "test_split": None,
        "process_docs": process_okapi_docs,
        "doc_to_text": "query",
        "doc_to_target": "{{label.lstrip()}}",
        "doc_to_choice": "choices",
        "metric_list": [
            {"metric": "acc", "aggregation": "mean", "higher_is_better": True},
            {"metric": "acc_norm", "aggregation": "mean", "higher_is_better": True},
        ],
        "metadata": {"version": 1.0},
    }


def hellaswag_english_task_config(task_manager: Any) -> dict[str, Any]:
    """Pinned original-English HellaSwag task for English-only checkpoints."""

    config = _task_yaml_config(task_manager, "hellaswag")
    config.update(
        {
            "task": "stage1_hellaswag_en",
            "dataset_path": HELLASWAG_DATASET,
            "dataset_kwargs": {"revision": HELLASWAG_REVISION},
            "metric_list": _accuracy_metrics(),
        }
    )
    return config


def copa_english_task_config(task_manager: Any) -> dict[str, Any]:
    """Pinned original-English COPA task for English-only checkpoints."""

    config = _task_yaml_config(task_manager, "copa")
    config.update(
        {
            "task": "stage1_copa_en",
            "dataset_path": COPA_DATASET,
            "dataset_name": "copa",
            "dataset_kwargs": {"revision": COPA_REVISION},
            "metric_list": _accuracy_metrics(),
        }
    )
    return config


def load_pinned_xnli_validation_dataset(
    *,
    language: str,
    revision: str = XNLI_REVISION,
    **_kwargs: Any,
) -> Any:
    """Load only the pinned XNLI validation parquet for one language."""

    import datasets
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=XNLI_DATASET,
        repo_type="dataset",
        filename=f"{language}/validation-00000-of-00001.parquet",
        revision=revision,
    )
    cache_dir = os.environ.get("HF_DATASETS_CACHE")
    return datasets.DatasetDict(
        {
            "validation": datasets.Dataset.from_parquet(
                str(path), cache_dir=cache_dir
            )
        }
    )


def xnli_task_config(task_manager: Any, task_name: str, language: str) -> dict[str, Any]:
    from lm_eval import utils

    entry = task_manager.task_index.get(task_name)
    if not isinstance(entry, dict) or entry.get("type") != "task":
        raise ValueError(f"lm-eval XNLI task not found: {task_name}")
    config = utils.load_yaml_config(entry["yaml_path"], mode="full")
    config.update(
        {
            "dataset_path": XNLI_DATASET,
            "dataset_name": language,
            "custom_dataset": load_pinned_xnli_validation_dataset,
            "dataset_kwargs": {"language": language, "revision": XNLI_REVISION},
            "training_split": None,
        }
    )
    return config


def _accuracy_metrics() -> list[dict[str, Any]]:
    """Return raw and continuation-length-normalized multiple-choice metrics."""

    return [
        {"metric": "acc", "aggregation": "mean", "higher_is_better": True},
        {"metric": "acc_norm", "aggregation": "mean", "higher_is_better": True},
    ]


def _task_yaml_config(task_manager: Any, task_name: str) -> dict[str, Any]:
    from lm_eval import utils

    entry = task_manager.task_index.get(task_name)
    if not isinstance(entry, dict) or entry.get("type") != "task":
        raise ValueError(f"lm-eval task not found: {task_name}")
    return utils.load_yaml_config(entry["yaml_path"], mode="full")


def belebele_cloze_task_config(task_manager: Any, task_name: str) -> dict[str, Any]:
    """Score complete answer strings instead of the original A/B/C/D tokens."""

    config = _task_yaml_config(task_manager, task_name)
    config.update(
        {
            "dataset_path": BELEBELE_DATASET,
            "dataset_kwargs": {"revision": BELEBELE_REVISION},
            "doc_to_text": "P: {{flores_passage}}\nQ: {{question.strip()}}\nAnswer:",
            "doc_to_choice": "{{[mc_answer1, mc_answer2, mc_answer3, mc_answer4]}}",
            "metric_list": _accuracy_metrics(),
        }
    )
    return config


def xstorycloze_task_config(task_manager: Any, task_name: str) -> dict[str, Any]:
    config = _task_yaml_config(task_manager, task_name)
    config.update(
        {
            "dataset_path": XSTORYCLOZE_DATASET,
            "dataset_kwargs": {"revision": XSTORYCLOZE_REVISION},
            "metric_list": _accuracy_metrics(),
        }
    )
    return config


def xcopa_task_config(task_manager: Any, task_name: str) -> dict[str, Any]:
    config = _task_yaml_config(task_manager, task_name)
    config.update(
        {
            "dataset_path": XCOPA_DATASET,
            "dataset_kwargs": {"revision": XCOPA_REVISION},
            "metric_list": _accuracy_metrics(),
        }
    )
    return config


def _uhura_choices(doc: Mapping[str, Any]) -> list[str]:
    return [str(value) for value in doc["choices"]["text"]]


def _uhura_target(doc: Mapping[str, Any]) -> int:
    labels = [str(value) for value in doc["choices"]["label"]]
    return labels.index(str(doc["answerKey"]))


def uhura_arc_easy_task_config(language: str, dataset_name: str) -> dict[str, Any]:
    return {
        "task": f"stage1_uhura_arc_easy_{language}",
        "dataset_path": UHURA_ARC_EASY_DATASET,
        "dataset_name": dataset_name,
        "dataset_kwargs": {"revision": UHURA_ARC_EASY_REVISION},
        "output_type": "multiple_choice",
        "training_split": "train",
        "validation_split": "validation" if language != "zu" else None,
        "test_split": "test",
        "doc_to_text": "Question: {{question.strip()}}\nAnswer:",
        "doc_to_target": _uhura_target,
        "doc_to_choice": _uhura_choices,
        "metric_list": _accuracy_metrics(),
        "metadata": {"version": 1.0},
    }


def global_mmlu_cloze_task_config(task_manager: Any, task_name: str) -> dict[str, Any]:
    """Preserve Global-MMLU domain filtering but score full option strings."""

    config = _task_yaml_config(task_manager, task_name)
    config.update(
        {
            "dataset_path": GLOBAL_MMLU_DATASET,
            "dataset_kwargs": {"revision": GLOBAL_MMLU_REVISION},
            "doc_to_text": "Question: {{question.strip()}}\nAnswer:",
            "doc_to_target": _global_mmlu_target,
            "doc_to_choice": "{{[option_a, option_b, option_c, option_d]}}",
            "metric_list": _accuracy_metrics(),
        }
    )
    return config


def _global_mmlu_target(doc: Mapping[str, Any]) -> int:
    answer = doc.get("answer")
    if isinstance(answer, int) and not isinstance(answer, bool):
        if 0 <= answer < 4:
            return answer
        raise ValueError(f"Global MMLU target index is out of range: {answer}")
    label = str(answer).strip().upper()
    try:
        return ("A", "B", "C", "D").index(label)
    except ValueError as exc:
        raise ValueError(f"invalid Global MMLU answer label: {answer!r}") from exc


@dataclass(frozen=True, slots=True)
class TaskDefinition:
    benchmark: str
    category: str
    language: str
    task_name: str
    task_input: str | Mapping[str, Any]
    stratum: str


def _group_tasks(task_manager: Any, group_name: str) -> list[str]:
    from lm_eval import utils

    entry = task_manager.task_index.get(group_name)
    if not isinstance(entry, dict) or entry.get("type") != "group":
        raise ValueError(f"lm-eval group not found: {group_name}")
    tasks = entry.get("task")
    if tasks == -1:
        config = utils.load_yaml_config(entry["yaml_path"], mode="simple")
        tasks = config.get("task")
    if not isinstance(tasks, list) or any(not isinstance(item, str) for item in tasks):
        raise ValueError(f"invalid lm-eval group task list: {group_name}")
    return list(tasks)


def build_task_definitions(profile: BenchmarkProfile, task_manager: Any) -> list[TaskDefinition]:
    definitions: list[TaskDefinition] = []
    supported = {
        "belebele",
        "belebele_cloze",
        "global_mmlu_cloze",
        "global_mmlu_lite",
        "include_base44",
        "okapi_hellaswag",
        "uhura_arc_easy",
        "xcopa",
        "xnli",
        "xstorycloze",
    }
    benchmark_ids = [str(item["id"]) for item in profile.benchmarks]
    if len(set(benchmark_ids)) != len(benchmark_ids):
        raise ValueError("benchmark IDs must be unique")
    unexpected = sorted(set(benchmark_ids) - supported)
    if unexpected:
        raise ValueError(f"unsupported benchmark IDs: {unexpected}")

    for entry in profile.benchmarks:
        benchmark = str(entry["id"])
        category = str(entry["category"])
        if benchmark in {"belebele", "belebele_cloze"}:
            requested_languages = entry.get("languages")
            allowed_languages = (
                {str(language) for language in requested_languages}
                if isinstance(requested_languages, list)
                else None
            )
            for task_name in _group_tasks(task_manager, str(entry["selector"])):
                language = task_name.removeprefix("belebele_")
                if allowed_languages is not None and language not in allowed_languages:
                    continue
                task_input: str | Mapping[str, Any] = task_name
                if benchmark == "belebele_cloze":
                    task_input = belebele_cloze_task_config(task_manager, task_name)
                definitions.append(
                    TaskDefinition(
                        benchmark,
                        category,
                        language,
                        task_name,
                        task_input,
                        "parallel",
                    )
                )
        elif benchmark == "xnli":
            for task_name in _group_tasks(task_manager, str(entry["selector"])):
                language = task_name.removeprefix("xnli_")
                definitions.append(
                    TaskDefinition(
                        benchmark,
                        category,
                        language,
                        task_name,
                        xnli_task_config(task_manager, task_name, language),
                        "parallel",
                    )
                )
        elif benchmark in {"global_mmlu_lite", "global_mmlu_cloze"}:
            for language in entry["languages"]:
                group = f"global_mmlu_{language}"
                for task_name in _group_tasks(task_manager, group):
                    task_input = task_name
                    if benchmark == "global_mmlu_cloze":
                        task_input = global_mmlu_cloze_task_config(
                            task_manager, task_name
                        )
                    definitions.append(
                        TaskDefinition(
                            benchmark,
                            category,
                            str(language),
                            task_name,
                            task_input,
                            task_name.removeprefix(group + "_"),
                        )
                    )
        elif benchmark == "include_base44":
            for language in entry["languages"]:
                group = f"include_base_44_{language}"
                for task_name in _group_tasks(task_manager, group):
                    definitions.append(
                        TaskDefinition(
                            benchmark,
                            category,
                            str(language),
                            task_name,
                            task_name,
                            task_name.removeprefix(group + "_"),
                        )
                    )
        elif benchmark == "okapi_hellaswag":
            if entry.get("variant") == "original_english":
                configs = [("en", hellaswag_english_task_config(task_manager))]
            else:
                configs = [
                    (str(language), okapi_task_config(str(language)))
                    for language in entry["languages"]
                ]
            for language, config in configs:
                definitions.append(
                    TaskDefinition(
                        benchmark,
                        category,
                        language,
                        config["task"],
                        config,
                        "parallel",
                    )
                )
        elif benchmark in {"xstorycloze", "xcopa"}:
            if benchmark == "xcopa" and entry.get("variant") == "original_english":
                config = copa_english_task_config(task_manager)
                definitions.append(
                    TaskDefinition(
                        benchmark,
                        category,
                        "en",
                        config["task"],
                        config,
                        "parallel",
                    )
                )
                continue
            prefix = f"{benchmark}_"
            factory = (
                xstorycloze_task_config
                if benchmark == "xstorycloze"
                else xcopa_task_config
            )
            requested_languages = entry.get("languages")
            allowed_languages = (
                {str(language) for language in requested_languages}
                if isinstance(requested_languages, list)
                else None
            )
            for task_name in _group_tasks(task_manager, str(entry["selector"])):
                language = task_name.removeprefix(prefix)
                if allowed_languages is not None and language not in allowed_languages:
                    continue
                definitions.append(
                    TaskDefinition(
                        benchmark,
                        category,
                        language,
                        task_name,
                        factory(task_manager, task_name),
                        "parallel",
                    )
                )
        elif benchmark == "uhura_arc_easy":
            configs = entry.get("dataset_configs")
            if not isinstance(configs, dict) or not configs:
                raise ValueError("uhura_arc_easy.dataset_configs must be an object")
            for language, dataset_name in configs.items():
                config = uhura_arc_easy_task_config(
                    str(language), str(dataset_name)
                )
                definitions.append(
                    TaskDefinition(
                        benchmark,
                        category,
                        str(language),
                        config["task"],
                        config,
                        "parallel",
                    )
                )

    expected_counts = {
        item["id"]: int(item["languages_expected"]) for item in profile.benchmarks
    }
    actual_counts = {
        benchmark: len({item.language for item in definitions if item.benchmark == benchmark})
        for benchmark in expected_counts
    }
    if actual_counts != expected_counts:
        raise ValueError(
            f"benchmark language coverage mismatch: expected {expected_counts}, got {actual_counts}"
        )
    return definitions
