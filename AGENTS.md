# Repository guidance

- Current update (2026-10-29): `BUGFIXES.md` distinguishes completed fixes from unresolved issues. `README.md` and root `requirements.txt` are current usage/install entry points. Preserve this repository's full experiments when syncing fixes from the separate LOPT package.

## Context and current priority

- Latest launcher preference (2026-10-29): use existing `lio/alg/train_tax.py` with explicit CLI parameters; full commands and TensorBoard are in `../docs/run_commands/escape_room.md`. Do not create tools/train entries or .ps1 launchers. Earlier `tools/train_escape_room.py` was removed after its CLI/runtime checks were integrated into the existing train module.
- Terminology: unqualified "论文" means the user's NeurIPS paper at https://arxiv.org/abs/2305.06227; current configuration checks explicitly use v2, Appendix D.1/D.2 Table 2 and Figure 5. "毕业论文" means `../docs/高尚-51205901009-面向序贯社会困境的信任合作机制与多智能体强化学习算法.pdf`. Do not silently import thesis seed counts or settings into claims about the paper.
- Latest execution status and PowerShell commands are in `../docs/escape-room-20261028/WORKLOG.md` and `RUNNING.md` in that directory. On 2026-10-28, the user authorized a minimal observation fix; N=2 and N=3 each passed 20 CUDA episodes with actual observation equality, finite gradients, real parameter changes, checkpoint restore and TensorBoard output. Earlier CPU checks and initialization snapshots below are historical, not current blockers.
- This is the user's modified fork of Learning to Incentivize Others, used for the Escape Room experiments in LOPT. The upstream-oriented `README.md` does not describe all of the user's additions.
- Read `../HANDOFF.md` for the latest internal author handover and directory-organization task. The user will rename the outer folder paper-LOPT and create a sibling LOPT; the next session organizes the two independent projects without merging/refactoring them. The sibling's AGENTS.md describes Cleanup; its repository-specific instructions do not automatically apply here.
- The user is an author of arXiv:2305.06227, *Learning Optimal "Pigovian Tax" in Sequential Social Dilemmas*. The intended `Felix2048/LOPT` release repository was unpublished at handoff; do not infer missing provenance from that address.
- First establish that the original code can initialize the environment, execute real policies, interact with the environment, and complete a bounded worker and tax-planner training update. Full convergence, published curves, and repeated-seed experiments are later work.
- The eventual release should contain LOPT and its runtime dependencies. Preserve historical implementations until that path has been verified; do not remove components merely because their names contain `baseline`, `policy_gradient`, or `lio`.
- The workspace now contains sibling directories `LOPT/lio/` and `LOPT/sequential_social_dilemma_games/`, each with its own Git history. The workspace root is not itself an initialized release repository. Do not merge histories, move directories, or publish as part of routine initialization.

## Provenance and inspected state

- Remote: `https://github.com/Felix2048/lio.git`; inspected HEAD: `75b2ea0fb4010fd05acbe57e37bb0dc63e31e179` (2023-01-19, `Update configs`).
- On 2026-10-28, after relocation, the worktree had no changes before this guidance file was added. No training source was changed during initialization.
- Preserve upstream attribution, the MIT `LICENSE`, and file-specific notices, including the Apache notice in `lio/utils/configdict.py`.
- Git reports dubious ownership in this checkout. Use a scoped override, e.g. `git -c safe.directory=C:/Users/felix.gao/workspace/LOPT/lio status --short`, with the current absolute path. Do not change global Git configuration as an incidental fix.

## Code map

Paths below are relative to this repository root, which contains another `lio/` directory.

- `lio/env/room_symmetric_tax.py`: LOPT Escape Room environment; worker movement, original rewards, tax-planner observations, bank state, and terminal distribution.
- `lio/env/room_agent.py`: worker positions and observations.
- `lio/alg/tax_agent.py`: LOPT tax/subsidy/budget networks, reward shaping, policy loss, bank penalty, and planner optimization.
- `lio/alg/policy_gradient.py`: ordinary worker policy and optimizer used by LOPT. This is a required component, not merely a comparison training entry point.
- `lio/alg/networks.py`: shared policy network builders, including `actor_mlp` for Escape Room.
- `lio/alg/train_tax.py`: `train_function(config, show_progress=False)`, `run_episode`, `Buffer`, and `TaxPlannerBuffer`; updates all workers and then the planner once per episode.
- `lio/alg/config_room_tax.py`: default configuration loaded by the current single-run and multiprocess tax entry points; currently N=3.
- `lio/alg/er_configs/config_room_tax_n2.py` and `config_room_tax_n3.py`: explicit two- and three-worker experiment settings.
- `lio/alg/evaluate.py`: shared evaluation helpers; LOPT uses `test_room_symmetric_tax_planner`.
- `lio/utils/configdict.py`: dict configuration with attribute access; configs are serialized to `config.json` for each run.
- `lio/alg/train_multiprocess.py`: historical multi-seed launcher, defaulting to five processes and eagerly importing other algorithms. It is not the minimal LOPT smoke entry point.
- `lio/alg/tax_only_agent.py`, `train_tax_only.py`, `config_room_tax_only.py`, and `er_configs/er_tax_only_configs/`: direct incentive/sanction variants corresponding to thesis LISCA-c/d. Do not label these as the main LOPT method based on filenames.
- `lio/alg/train_lio.py`, `lio_agent.py`, `lio_ac.py`, `lio_decentralized.py`, `train_pg.py`, and related configs contain upstream algorithms and comparisons. `lio/env/ssd*.py` and `train_ssd*.py` provide other SSD experiment paths; the sibling repository contains the Cleanup LOPT PPO path.

## Experiment behavior to preserve

- The explicit tax configs use N=2/M=1 or N=3/M=2, at most five steps, bank-enabled tax, and 50,000 training episodes. Defaults include 10 evaluation episodes every 100 training episodes, CPU execution, and seed 12340.
- `EscapeRoom` derives the lever threshold from `n_agents`; changing `config.env.min_at_lever` alone does not change it. Some inherited config fields are unused; inspect consumers before assuming a field controls behavior.
- `run_episode` executes worker policies, calls `env.step`, executes the planner, then calls `env.step_tax_planner` to shape rewards and fill buffers.
- The planner emits per-worker sigmoid tax rates, softmax subsidy proportions, and a sigmoid bank budget ratio scaled by `budget_ratio_scale`. `TaxAgent.act` masks tax on nonpositive worker rewards.
- The planner's reward is the sum of original worker rewards. Workers train on rewards after taxation, subsidies, and terminal bank distribution. Keep these quantities distinct in validation and logs.
- The bank carries revenue between steps; the environment adds the remaining bank equally to worker rewards at termination. Check the terminal behavior together with bank accounting when changing this code.
- The user authorized fixing the omitted action/reward observation inputs. `EscapeRoom.step` now passes actual actions and original rewards to `get_tax_planner_obs`, which uses `is not None` checks for array compatibility. The planner's training reward remains the original collective reward. Preserve the distinction between this input fix and the historical runs with zero-filled fields.
- The non-bank branch of `step_tax_planner` does not assign `shaped_reward_sum` before returning it. The inspected main configs enable the bank; validate other variants explicitly before claiming support.
- The sibling handoff records unresolved differences between thesis settings, launchers, and reported seed counts. Preserve settings during short execution checks and revisit those discrepancies for performance reproduction.

## Environment and execution

- This is a legacy TensorFlow graph/session project. LOPT code imports `tensorflow.compat.v1`; `train_tax.py` disables eager execution before creating placeholders. An independent harness must also select graph mode before constructing the environment or policies.
- README lists upstream Python 3.6, TensorFlow >=1.12, and Gym 0.10.9. `setup.py` only declares package metadata and `find_packages()`; it does not install or lock the runtime dependencies. Do not treat either file as a verified environment specification.
- On 2026-10-28, metadata inspection found `D:/miniconda3/python.exe` (Python 3.9.21), `tensorflow-gpu==2.10.1`, `keras==2.10.0`, `numpy==1.23.0`, `protobuf==3.20.2`, and `scipy==1.13.1`. Gym, Ray, PettingZoo, and pytest were absent from that interpreter's distribution metadata. TensorFlow import, GPU support, environment interaction, and training have not been validated.
- Use an isolated environment for dependency work; preserve the user's existing base environment. Do not upgrade TensorFlow/Gym or port to a new framework as an incidental setup fix.
- The LOPT Escape Room path requires Gym, NumPy, TensorFlow, and SciPy via `evaluate.py`; progress display also uses tqdm. It does not import Ray or the sibling SSD package along this inspected path. Avoid installing all comparison dependencies merely to run this path.
- Make the repository root importable as `lio` (historically through `pip install -e .`). `train_tax.py` also contains a local `from policy_gradient import ...` inside `train_function`, so importing it as a module from the root is insufficient by itself. Handle the algorithm import path explicitly in a short-run harness or make a focused import repair when authorized.
- The historical single-run invocation is `python train_tax.py er` from `lio/alg/`, once package imports and dependencies are available. This starts the default 50,000-episode N=3 run; it is not a bounded validation command.
- Output is resolved from the process working directory as `../results/<exp_name>/<dir_name>`. With the historical working directory, this is `lio/results/` inside this repository. Set a dedicated smoke run name and control the working directory to avoid overwriting runs or writing outside the intended repository.
- `train_function` writes `config.json`, initializes `log.csv`, periodically evaluates/saves, and saves a final checkpoint even for short runs. A run shorter than `config.alg.period` produces no evaluation rows; override the period deliberately if logs are part of validation.
- For repeated runs in one interpreter, use a fresh graph/session or separate process to avoid variable-name collisions. The training function now closes its session and optional TensorBoard writer on successful completion.
- Do not launch `train_multiprocess.py` for a first check. Besides broad imports and multi-seed defaults, it starts processes at module scope without a Windows `__main__` guard.

## Validation and editing

- Initialization consisted of static source inspection, Git checks, and interpreter/package metadata checks. No dependency installation, training, policy forward pass, or environment reset/step has been completed as of this initialization.
- There is no configured unit-test suite or CI in this checkout. `lio/alg/test_er.py` and `test_ssd.py` are historical checkpoint evaluation scripts with config arguments; do not report a bare pytest invocation as the intended validation.
- Start with a dedicated N=2 CPU run with a small explicit episode budget, then repeat N=3. Use real worker/planner networks, environment transitions, and both optimizers; check finite outputs, losses, gradients, and parameters. A smoke check establishes execution, not learning quality or published performance.
- Record exact interpreter/dependency versions, config overrides, command, working directory, output paths, and exceptions. Preserve a rerunnable bounded command and logs once a run succeeds. Check checkpoint saving/restoration after the main execution path works.
- For tax changes, review original vs shaped rewards, planner targets, bank conservation, observation/action shapes, penalty inputs, and terminal payout together. Separate compatibility repairs from changes to algorithm behavior.
- Keep source changes focused and preserve experiment defaults. Do not format or delete unrelated historical code during runtime recovery, and do not fabricate a validated lockfile before successful execution.
