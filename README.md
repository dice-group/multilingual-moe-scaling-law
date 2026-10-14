# Scaling Multilingual Mixture-of-Experts

Code and released numerical results for *Scaling Multilingual Mixture-of-Experts: How Language Coverage Changes Capacity and Compute Allocation*.

The study examines active model size, expert count and language coverage through
controlled pretraining experiments, joint scaling laws and multilingual evaluation.
See the [repository orientation](docs/repo_orientation.md) for the code layout,
analysis modules and included data.

## Setup and reproduce results

Run from the repository root with Python 3.12:

```bash
python3.12 -m venv .venv
. .venv/bin/activate
pip install -r envs/paper-cpu.txt
export PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -m multilingual_megatron.devtools.run_test_suite reviewer_cpu
python -m result_analysis.scaling_law_fit.submission_replay --output outputs/replay
```

Choose a new output directory. The commands verify and replay the included numerical results.

## Training dependencies

**Note:** We use custom forks of Megatron Bridge and LM Evaluation Harness for
training and external evaluation. These dependencies were managed as submodules;
for anonymous review, their source is provided in separate repositories:

- [Megatron Bridge](https://anonymous.4open.science/r/megatron-bridge-mmsl/README.md) → `external/Megatron-Bridge/`
- [LM Evaluation Harness](https://anonymous.4open.science/r/lm-evaluation-harness-mmsl/README.md) → `external/lm-evaluation-harness/`

Include Bridge's pinned Megatron-LM dependency, then run `scripts/env/create_bridge_train.sh`.
Configure local paths and scheduler settings in `configs/`. See
[experiment reproduction](docs/repo_orientation.md#experiment-reproduction) for the required inputs.
