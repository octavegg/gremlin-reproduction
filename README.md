# GREMLIN

**Generating Random Environments for Meta-Learning with Interacting Networks.**
This package reproduces the reported fixed-bank study: sixteen generated games,
six human-designed games, and a learning rule trained on the generated games.
It contains the measured results, exact games and teacher weights, evaluation
code, and numerical meta-learning code. The entry points cover only the reported
fixed-bank study; some frozen internal modules also contain older shared helpers.

| What you want to do | Support |
| --- | --- |
| Rebuild figures and score tables | Ready: `reproduce.py` |
| Train an agent with a frozen teacher | Ready: `evaluate.py` |
| Meta-train the teacher itself | `train.py`: new run from weights, then full-state resume |

## View the results

No training or GPU is needed. This optional command rebuilds the two result
figures and exports the scores as a readable CSV:

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements/figures.txt
.venv/bin/python reproduce.py
```

Open `results/generated-groups.svg`, `results/human-endpoints.svg` and
`results/scores.csv`. Expected records: 160 generated-task learners, 120
human-task learners and 32 fixed-anchor assessments. Figures show final return;
the CSV also retains curve averages and Double DQN. Normalization is per game:
initial-teacher mean = 0, DQN mean = 1, not optimal performance.
The anchor's two seed labels refer to assessment episodes, not trained policies.

## Evaluate a teacher

Inspect the settings first, without installing learning dependencies:

```bash
python evaluate.py --panel human --task acrobot --method gremlin --seed 0
```

Use Linux, Python 3.12 and a CUDA 13-compatible GPU. Install the separate learning
dependencies only if you want to rerun learning:

```bash
python3.12 -m venv .venv-rl
.venv-rl/bin/python -m pip install -r requirements/learning.txt
# Run for up to 60 minutes + five slack minutes; repeat to resume this human learner.
.venv-rl/bin/python evaluate.py --panel human --task acrobot --method gremlin --seed 0 \
  --output results/acrobot-gremlin-0 --minutes 60 --execute
```

| Panel | Tasks | Agent seeds | Interaction budget per agent |
| --- | --- | --- | --- |
| `generated` | `0` through `15` | 0–1 | 282,800 |
| `human` | `acrobot`, `lander` | 0–4 | 1 million |
| `human` | `breakout`, `invaders`, `qbert`, `seaquest` | 0–4 | 5 million |

Methods: `initial`, `gremlin`, `disco103`, `dqn`; generated tasks also accept
`double-dqn`. The time cap does not change the interaction budget. Use a distinct
output folder for each agent; GPU allocation is manual. No SSH setup is needed.
Atari ROMs are not included; benchmark assets retain their applicable terms.

<details>
<summary>Resuming and checking runs</summary>

Human and generated DQN agents resume in the same output folder. A generated
learned-rule agent resumes into a **new** output folder with
`--resume-from previous-output/state-NNNNNNNNN.state`; retain earlier segments
for its complete curve. Interrupted work since the last save may be lost.
Restoring state does not promise bitwise-identical results across GPU models.

```bash
python tools/verify.py --artifacts
```

This verifies the supplied files against their recorded checksums.
Development tests and diagnostic scripts are not included in this package.

</details>

## Meta-train a teacher

**Evaluation above freezes the teacher. It does not improve its weights.**
Meta-training instead updates the teacher through its effect on learning agents.
Using the learning environment installed above:

```bash
# Inspect without training; append --execute when ready.
python train.py --init gremlin --output results/meta --updates 100 --minutes 60
.venv-rl/bin/python train.py --init gremlin --output results/meta \
  --updates 100 --minutes 60 --execute

# Continue that new run, without resetting agents, replay or optimizers.
.venv-rl/bin/python train.py --resume results/meta --updates 100 --minutes 60 --execute
```

`--init gremlin` starts from GREMLIN 1000's weights, with **fresh agents, critics,
replay and optimizers**. It does not resume the historical discovery run.
Use `--init initial` for the supplied original weights, or `--init random
--seed 1` for a new initialization. These use the current leaky activation.
A compatible teacher-only `.state` file is also accepted with explicit
`--activation relu` or `--activation action-input-leaky-relu-0.01`.
Arbitrary architectures and complete checkpoint files are not weight files.

The command uses the **final reported settings**: 32 logical agents, agent
learning rate 0.01, teacher rate 0.005, and sampled 100k/250k/500k/1M-step
lifetimes with weights 8:4:2:1. It keeps the reported replay and per-agent Adam
updates before averaging. It does not replay the historical changes in settings.
For a small portable interface, agents run sequentially on one GPU; this is
slower than the original distributed training and has not had a full-length
GPU validation. Deterministic update/recovery fixtures are tested.

`--updates` bounds **additional** teacher updates per invocation; `--minutes`
adds five slack minutes for finishing, then enforces a hard stop. Every complete
update saves all training state; only the latest two recovery checkpoints remain.
`--resume` uses the same source, games, settings and saved random streams.
Partial work since the last complete update is repeated after interruption.
Progress prints to the terminal; per-agent metrics remain in `audits/`.
`teacher.state` is a convenient weights-only export, refreshed on resume;
only `checkpoints/` is authoritative for recovery.
The default output cap is 16 GiB (`--max-gib`), with a 2 GiB free-space reserve.
Unpublished interrupted checkpoint files are preserved separately, not silently
deleted. This command neither launches SSH workers nor manages GPU allocation.

## Where things are

| Location | Contents |
| --- | --- |
| `data/` | Measured per-agent results, exact settings and checksums |
| `artifacts/` | Sixteen saved games and the initial, trained and Disco103 teacher weights |
| `runtime/` | Frozen numerical code for generated evaluation, human evaluation and discovery |
| `requirements/` | Separate lightweight plotting and GPU-learning dependencies |
| `tools/` | Checksum verifier and internal runtime loader |
| `licenses/` | Content license and preserved third-party notices |

The JSON files are read automatically; there is nothing to fill in or edit.
The three runtime folders preserve the code used for each experiment. Their
historical differences are intentional; they are not three user-facing setups.
`runtime/human/` and `runtime/discovery/` contain their modules directly.
The generated-evaluation snapshot keeps its original layout because its two
unchanged drivers verify and use those paths.
The two scripts in `runtime/generated/scripts/` are called by `evaluate.py`:
`audit_frozen_rule.py` runs learned-rule agents and `run_calibration.py` runs
DQN/Double DQN. Despite their historical names, they are evaluation drivers,
not optional diagnostic scripts.

**Historical controls.** The original
generated initial control uses hard ReLU; the human initial control matches
the trained teacher's leaky activation. These controls must not be interchanged.

**Licenses.** Original code and reproduction materials: [MIT](LICENSE).
Original article and figures: [CC BY 4.0](licenses/LICENSE-CONTENT.md).
[Third-party terms](licenses/THIRD_PARTY.md) remain unchanged. DiscoRL is an
external pinned dependency, not vendored source.
