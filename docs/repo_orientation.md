# Repository orientation

This repository contains experiment configurations, training orchestration,
scaling-law fitting, and numerical analysis for the multilingual MoE study.

## Layout

| Directory | Contents |
|---|---|
| [`configs/`](../configs/) | Corpus, language mixtures, training, benchmark and fitting settings |
| [`src/multilingual_megatron/pipeline/`](../src/multilingual_megatron/pipeline/) | Sampling, tokenization, training and evaluation pipeline |
| [`src/multilingual_megatron/experiments/scaling/`](../src/multilingual_megatron/experiments/scaling/) | Study workflows, release construction and fitting implementations |
| [`result_analysis/`](../result_analysis/) | Saved observations, coefficients, predictions and analysis modules |
| [`envs/`](../envs/) | CPU and training environments |
| [`scripts/`](../scripts/) | Environment setup and execution helpers |
| [`tests/`](../tests/) | Numerical replay and experiment-contract checks |

## Included numerical data

- [`stage1_data_v1/`](../result_analysis/stage1_data_v1/): compact observations and supplementary fitted results; table definitions and checksums are in `manifest.json`.
- [`stage1_selected_law_v1/`](../result_analysis/stage1_selected_law_v1/): selected RN12 law, coefficients, predictions, scoring inputs and composition checks.
- [`stage1_jointmoe_token_contrast_v1/`](../result_analysis/stage1_jointmoe_token_contrast_v1/): JointMoE fits using total tokens and training tokens per language.

The README's CPU command verifies the delivered coefficients and recomputes their
prediction scores. RN12 is the selected 12-shared-coefficient law; historical
RN14 results retain their original identity. Fitted allocation uses the
composition-aware M6-ND law.

## Experiment reproduction

Training and supporting studies use [`configs/scaling_law/`](../configs/scaling_law/),
with the larger-model run in [`configs/stage1/`](../configs/stage1/).
Benchmark definitions are in [`configs/benchmarks/`](../configs/benchmarks/).
Fitting profiles are in [`configs/stage1_fit/`](../configs/stage1_fit/), including
`final_coverage_ablation_20261011.json` for the selected law.

Full training requires the external dependencies linked in the [README](../README.md),
corpus/tokenizer inputs, and GPU resources. Full refitting requires the original
fitting inputs and a local Git revision. Saved-data replay needs neither training
dependencies nor raw corpora. Write regenerated results to new output directories.
