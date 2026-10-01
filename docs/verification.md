# What is verified, and what is not

**The authority for this page is `scripts/verify.sh`, not this document.**
Everything below was established by running it, and the machine-readable record
is `experiments/results/verification_baseline.json`.

## The rule this project verifies by

**Check exit codes, not output text.**

This is not a style preference. It was learned the hard way, twice, and both
instances are recorded below because they are the most honest thing in this
repository.

## Two failures of the verification process itself

### 1. A lint error committed under a message claiming lint passed

For several rounds this project reported "ruff clean". It was not. The checks
were piped through `tail -n 1`, which prints ruff's summary line — "All checks
passed!" — as the *last* line of a run that also printed a `B905` violation
above it. The summary was read as a clean bill.

The error was real (`zip()` without `strict=`), it was visible in the output the
whole time, and it survived being reported as verified across multiple commits.

### 2. Writing the script immediately found a second hole

`scripts/verify.sh` was written to enforce the rule above, and on its **first
run** it reported a failure nobody had seen before:

```
FAIL  ruff format --check   exit=1   40 files would be reformatted
```

`ruff check` (the linter) passes on all of them. `ruff format` is a **different
tool** that had never been run in this project at all. Two independent tooling
holes, one of them invisible to the tool that had been run every time.

### What changed

* `scripts/verify.sh` runs every check unpiped, captures its **exit status**, and
  exits non-zero if any failed. Output is shown for the reader; the status is
  the verdict.
* It writes a JSON record of every check, and the CI workflow asserts that the
  record exists and agrees with the job status — a green run that produced no
  evidence is the failure mode this project exists to catch.
* The `ruff format` deviation is recorded as an **advisory** check: visible on
  every run, not gating, because reformatting 40 files in the same commit as the
  baseline tag would bury the audit trail. It is fixed separately and the count
  is published here.

### A third instance, in the fix for the first two

Checking the tagged baseline in a clean worktree surfaced a third one. The
worktree has no `.venv`, so the script fell back to the system `python3`, every
`pytest` call died with *No module named pytest*, and the script printed
**VERIFICATION FAILED** with exit 1.

That is false. The code was fine; the environment had never been set up. The
script had conflated "a check failed" with "the script could not run" and
reported an environment problem as a defect in the code under test — the same
error class, in the mechanism written to catch that error class.

Fixed by a preflight that distinguishes the three exits:

| Exit | Meaning |
|---|---|
| `0` | every check exited 0 |
| `1` | at least one check genuinely failed |
| `2` | the script could not run — no usable interpreter, or dependencies missing |

The clean-worktree case is now exit 2 with instructions to run
`pip install -e ".[dev]"`, not a claim that the code is broken.

### The generalisation

This is the project's thesis applied to itself. A robotics system that reports
"the pipeline succeeded" has made a claim; the evaluation infrastructure must
independently check that the claim is true, and the checker must itself be
checked. Twice, the checker was reporting on the wrong evidence — once from a
summary line, once from a tool that had not been run.

The same shape appears throughout the codebase: the regression detector grouped
cells by a field that was not the field being varied; the perception filter
reported σ 300× below its actual error; the benchmark suite reported "MET" for
injections that had not been applied. **A mechanism that looks correct and does
less than it appears to is the failure mode this project exists to detect, and it
was present in the verification process before it was present in the robot.**

## Verified locally

macOS, Apple Silicon, Python 3.12, CPU only. Every line below was executed.

| Area | Evidence |
|---|---|
| Package installs and imports | `ruff`, `mypy` clean over 77 source files |
| Lint | `ruff check` exit 0 over `src/robot_ci`, `tests`, `experiments` |
| Types | `mypy src/robot_ci` exit 0, 77 files |
| Unit tests | 315 passing |
| Integration tests | 50 passing |
| End-to-end CLI tests | 33 passing |
| **Total** | **398 tests** |
| Environment | `robot-ci validate` PASSED |
| Import isolation | the engine imports with no MuJoCo, no Isaac, no ROS 2 |
| MuJoCo backend | 60-episode experiment, 72.8 s, real contact dynamics |
| Conditional regression detection | measured; see [`regression.md`](regression.md) |
| Counterexample search | found at 0.75 mm over a 12 mm axis, 15 evaluations |
| Behavioural boundaries | 7 axes swept; 1 boundary located, 6 shown inert |
| Failure taxonomy | 15 categories, multi-hypothesis output |
| Ablation | scored against an injected cause; OFAT correct, correlation wrong |
| Benchmark suite | 20 benchmarks, 0 mismatches, 8 measured expectations |
| **Nominal solvability** | **120/120 per policy, 95% CI [0.969, 1.000]; the old 66.7% figure falsified** |
| **RQ2 opposing effect** | **4x6 noise-by-budget grid, 16 seeds/cell; sign flips in every row, 11 positive vs 10 negative cells** |
| CI gate | `robot-ci test` runs 3 gates and reports coverage |

## Unverified

| Claim | Status | Why |
|---|---|---|
| Isaac Sim adapter works | **Implemented but unverified** | No macOS build, no GPU, never executed. Contract tested against mocks. |
| `IsaacSession` implementation | **Does not exist** | The one missing piece; all unverified work is behind it. |
| Linux CI workflow | **Written, never run** | `.github/workflows/verify.yml` has never executed. |
| ROS 2 packages | **Not started** | `ros2_ws/src/` holds three empty directories — 0 files. |
| MoveIt 2 integration | **Not started** | Same. |
| Notebooks for Colab/Kaggle | **Not started** | `notebooks/` is empty. |
| Cloud execution | **Not started** | No remote host has been used. |
| `ruff format` compliance | **Known deviation** | 40 files would be reformatted; recorded, not gating. |
| Cross-backend agreement | **Not measured** | MuJoCo and Isaac have never been run on the same scenarios. |
| RQ2 adverse channel under a *perceptual* axis | **Untested, structurally unreachable** | Requires a moving object; the schema has no motion field. Latency is inert by mechanism, not by luck. |
| RQ2 aggregate as a population estimate | **Never claimed** | The grid brackets the boundary by design; the mean is reported as a property of that grid only. |

## The remote verification path

The honest limitation of this project is not that it lacks features; it is that
a Mac cannot run ROS 2, MoveIt 2, Isaac Sim, or any GPU workload. Scaffolding
empty packages for those would not address that, so it was not done.

`.github/workflows/verify.yml` is the start of the real answer:

* It runs on `ubuntu-24.04`, where ROS 2 and GPU backends become possible.
* It calls **`scripts/verify.sh`** — the same script run locally — so there is
  exactly one definition of "verified" and CI cannot drift from it.
* The MuJoCo experiments are a **separate, non-blocking** job, so a missing
  optional dependency cannot be mistaken for a broken change.
* It asserts that its own evidence file exists and agrees with the job status.

It has not run. It is the first step, not the destination: it verifies only what
is genuinely portable, and says so rather than implying it verifies Isaac or
MoveIt.

## Reproducing this page

```bash
bash scripts/verify.sh                          # everything, ~12 min
bash scripts/verify.sh --quick                  # skip the experiments, ~1 min
bash scripts/verify.sh --json record.json       # machine-readable record
echo $?
```

Exit codes: `0` all passed, `1` a check failed, `2` the script could not run (no
usable interpreter or missing dependencies — an environment problem, not a
failing check).

The record is written to `experiments/results/verification_baseline.json` and
the transcript to `verification_baseline.txt`.
