# Counterexample discovery

**Evidence: Verified locally.** RQ1, RQ4 and an end-to-end search were all run
on this machine.

## The objective

Find `x` such that the baseline succeeds and the candidate fails, then minimise
`|x|`. A search that reports "found" without saying how small the counterexample
is has hidden half its result.

## Strategies

| Strategy | Evaluations | Wall clock | Counterexample | Minimal? |
|---|---|---|---|---|
| `random` | **1** | 1.4 s | 7.64 mm | no |
| `grid` | 2 | 4.4 s | 4.00 mm | no |
| `binary` | 8 | 17.1 s | **1.17 mm** | yes |
| `adaptive` | 15 | 26.0 s | **1.17 mm** | yes |

Budget 24 evaluations each, identical, seed 3, axis 0–12 mm.

**Tests-to-failure alone ranks `random` best, and that is the wrong answer.**
Random is 15x cheaper than `adaptive` and its counterexample is 6.5x less
minimal: it needs 7.64 mm of noise to reproduce where 1.17 mm suffices. Grid and
random spend their budget finding *something*; binary and adaptive spend it
*minimising*. They are solving different problems, and the comparison is only
fair because both numbers are shown.

**This is one trial per strategy on one axis and one seed.** It shows the
trade-off exists. It cannot show any strategy is better in general.

## The oracle

`EpisodeOracle` runs one real paired episode per point: both arms, same resolved
scenario, same seed. The seed-pairing is a tested property, not a convention —
an oracle that gave the arms different seeds would confound the policy effect
with the environment draw, and every counterexample would be a statement about
seeds.

`CachingOracle` memoises, so a search that revisits a point does not spend
budget re-learning it.

## Reproducing a counterexample

```bash
robot-ci counterexample search --backend mujoco --strategy adaptive \
  --budget 16 --scenario experiments/configs/demo_seed3.yaml \
  --baseline policy_v3 --candidate policy_v1 --axis "depth_noise_mm:0:12:mm"
```

```
 counterexample found
 evaluated  : 15 points
 condition  : depth_noise_mm=0.75
 baseline   : policy_v3 -> success
 candidate  : policy_v1 -> failure
 boundary   : 0.7266 (depth_noise_mm)
 bracket    : [0.0586, 0.0625]
 necessary  : depth_noise_mm
```

0.75 mm out of a 12 mm axis. The artifact is self-contained YAML — full
scenario, perturbation, config hash, git commit — so it replays without this
repository's state.

## The reproduction rate, always

```bash
robot-ci counterexample reproduce <artifact> --backend mujoco --seeds 3,7,11 --strict
```

```
 reproduced    : 1/3 (0.333)
 all agreed    : NO
```

A counterexample that reproduces on a third of seeds is a property of the seed
as much as of the condition. RQ4 measured **4/11 = 36.4%** on an earlier version
and **3/11 = 27.3%** after the IK tolerance change moved the minimal
counterexample from 1.0 mm to 2.0 mm. Both numbers are plausible; the spread
between them at n=11 is itself the argument for reporting a rate rather than a
yes/no.

The replay also surfaced the case a naive check gets wrong: one of the
non-reproducing seeds was the **inverse** of a counterexample — the baseline
fails and the candidate succeeds. The arms disagreed, just not in the direction
the search was looking for. The oracle tests direction.

`--strict` exits non-zero when the rate is below 1.0, so a CI job cannot pass on
a counterexample that does not actually reproduce.

## Arm roles

The search looks for conditions where the **candidate** fails and the
**baseline** succeeds. The research experiments therefore use
`baseline=policy_v3, candidate=policy_v1`: `policy_v1` is the arm that regresses.
Getting this backwards inverts every conclusion, and the oracle will happily
report a null rather than complain.
