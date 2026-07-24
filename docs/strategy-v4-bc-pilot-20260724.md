# StrategyV4 B/C Paired Pilot (2026-07-24)

## Goal

Measure whether the final 32 StrategyV4 combat-event observation fields improve
held-out Physical Fight behavior.

- B: V3 model with the final 32 fields forced to zero.
- C: the same V3 model with live combat-event fields.
- Shared starting checkpoint:
  `trained_models/kof98_strategy_v4a_shared_v3_seed98_final.zip`
- Paired seeds: `98`, `198`, `298`.
- Training: `100k` requested steps per arm (PPO completes whole rollouts).
- Evaluation: four P2 styles, 20 deterministic held-out episodes per style.
- Only the observation gate differs between paired B and C runs.

## Engineering Changes

- Added real reaction, reaction-counter-valid, defense phase, confirm phase, and
  event-density TensorBoard metrics.
- Renamed the old non-zero-vector metric as a wiring diagnostic. A neutral
  one-hot phase makes the vector non-zero even without a physical event.
- Added run-relative checkpoints at `0/25k/50k/75k/100k`.
- Added a deterministic Physical evaluator with raw opportunity/success counts.
- Isolated each P2 style in its own process to avoid FBNeo teardown/reload access
  violations.
- Added a sequential B/C pilot runner and matching VS Code launch config.

## Direction-Filtering Bug Found

Step Events V5 is bidirectional, but the Python defense curriculum originally
accepted `MANUAL_BLOCK_SUCCESS` and `BLOCKSTUN_ENDED` without checking event
direction. P2 blocking P1 could therefore be counted as a P1 manual defense and
could open a P1 counter window.

The final experiment filters P1-defense events to:

```text
source_player == 2
target_player == 1
```

P1 counter attacks and hits are filtered to the opposite direction. A
`wrong-direction` counter-window regression test was added and passes.

The first diagnostic pilot was retained under `v4a`; the valid results below
come from the `v4b_directionfix` runs.

## Safety Results

All six runs completed without:

```text
batch epoch mismatch
chip hit/block conflict
dropped step event
non-finite observation
```

The complete deterministic action/event regression suite also passed.

## Held-Out Aggregate Results

Each row contains 80 episodes.

| Seed | Arm | Win | HP diff | Damage dealt | Damage taken | Max combo | 4+ combo |
| ---: | :-- | --: | ------: | -----------: | -----------: | --------: | -------: |
| 98  | B | 50% | +14.25 | 70.00 | 55.75 | 1.50 | 0% |
| 98  | C | 50% | +27.50 | 73.75 | 46.25 | 1.50 | 0% |
| 198 | B | 50% | +19.75 | 73.25 | 53.50 | 1.25 | 0% |
| 198 | C | 75% | +25.75 | 77.25 | 51.50 | 1.75 | 0% |
| 298 | B | 50% | +30.00 | 76.50 | 46.50 | 1.25 | 0% |
| 298 | C | 25% | -11.50 | 48.25 | 59.75 | 1.25 | 0% |

Paired `C - B` deltas:

| Seed | Win delta | HP diff delta | Max combo delta |
| ---: | --------: | ------------: | --------------: |
| 98  | 0 pp   | +13.25 | 0.00 |
| 198 | +25 pp | +6.00  | +0.50 |
| 298 | -25 pp | -41.50 | 0.00 |

Three-seed means:

| Arm | Win | HP diff | Damage dealt | Damage taken | Max combo | 4+ combo |
| :-- | --: | ------: | -----------: | -----------: | --------: | -------: |
| B | 50% | +21.33 | 73.25 | 51.92 | 1.33 | 0% |
| C | 50% | +13.92 | 66.42 | 52.50 | 1.50 | 0% |

## Conditional Event Results

Across all three seeds:

```text
B: 100 P1 manual blocks, 100 post-block windows, 40 matching counter hits
C:  20 P1 manual blocks,  20 post-block windows,  0 matching counter hits
```

All six runs observed hundreds of starter-hit events, but no held-out run
produced a confirmed hit-follow-up. No run produced a 4+ combo.

The training event source was not sparse:

```text
P1 reaction active: roughly 25% to 32% of reported steps
P2 reaction active: roughly 38% to 48%
Block contacts: roughly 0.5 to 0.9 per 1k emulated fight frames
Clean hits: roughly 3.3 to 4.2 per 1k emulated fight frames
```

However, `confirm_non_neutral_rate` stayed near 79% to 81%. This is suspiciously
high and suggests the generic confirm phase remains latched for much of an
episode. It is likely a weak or misleading observation feature in its current
form.

## P2 Style Means

| Arm | Style | Win | HP diff | Max combo |
| :-- | :---- | --: | ------: | --------: |
| B | Oniyaki | 0% | -16.00 | 3.00 |
| C | Oniyaki | 33.3% | -13.00 | 3.00 |
| B | Guard | 0% | 0.00 | 0.00 |
| C | Guard | 0% | +9.33 | 0.33 |
| B | Jump-in | 100% | +32.00 | 1.33 |
| C | Jump-in | 66.7% | -12.00 | 1.67 |
| B | Poke | 100% | +69.33 | 1.00 |
| C | Poke | 100% | +71.33 | 1.00 |

C trades a possible Oniyaki improvement for a large and inconsistent Jump-in
regression. The aggregate average hides this behavior.

## Decision

The event observation wiring is safe, but the pilot does not show a reliable
policy benefit:

- C is not consistently better across paired seeds.
- Mean held-out win rate is unchanged.
- Mean HP differential is worse.
- Physical 4+ combo and hit-confirm transfer remain zero.
- Post-block response is worse in C.
- Seed variance is larger than the observed average benefit.

Do not continue C with the same setup for a long run.

The next experiment should first fix/shorten the generic confirm phase and then
make Physical reward or curriculum transfer require correct event-conditioned
behavior. More steps with the same reward and phase encoding are unlikely to
make the policy use the timing information.

## Artifacts

- Models:
  `trained_models/kof98_strategy_v4b_directionfix_*`
- Held-out JSON:
  `ai_logs/evaluations/kof98_strategy_v4b_directionfix_*`
- TensorBoard:
  `ai_logs/kof98_strategy_v4b_directionfix_*`
- Runner:
  `tools/run_strategy_v4_pilot.py`
- Evaluator:
  `tools/evaluate_kof98_physical.py`
