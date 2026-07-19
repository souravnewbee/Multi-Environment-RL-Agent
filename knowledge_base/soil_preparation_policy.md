# Soil Preparation Policy
## Domain: Agriculture | Task: soil_preparation

## Overview
The goal of soil preparation is to bring soil conditions into the optimal
range for exotic fruit cultivation (calibrated for dragon fruit / pitaya)
before the planting window closes. The agent has a fixed number of days
to prepare the soil and must plant before the deadline. Planting in
poorly prepared soil wastes the entire season.

## Target Soil Conditions
All three conditions must be met before planting is justified:

| Parameter        | Target Range      | Why It Matters                              |
|------------------|-------------------|---------------------------------------------|
| Soil pH          | 5.5 – 7.0         | Controls nutrient availability and root health |
| Organic Matter   | ≥ 60%             | Provides nutrients, water retention, microbial activity |
| Drainage Quality | ≥ 60%             | Dragon fruit is cactus-family and rots in waterlogged soil |

## Action Rules

### Add Compost
Compost should be applied whenever organic matter is below 60%.
It is the most broadly beneficial action — it raises organic matter,
gently nudges pH toward neutral, and improves soil structure over time.
Compost should not be overused when all three targets are already met,
as it has a small cost and provides diminishing returns beyond the target.

### Adjust pH
pH adjustment should be applied whenever soil pH falls outside the 5.5–7.0
range. If pH is too acidic (below 5.5), lime is applied to raise it.
If pH is too alkaline (above 7.0), sulfur is applied to lower it.
Adjusting pH when it is already in range wastes cost and risks
overshooting — the agent should not apply pH adjustment unnecessarily.

### Improve Drainage
Drainage improvement should be applied whenever drainage quality is below 60%.
Poor drainage is the most critical risk for dragon fruit and other exotic
cactus-family crops — waterlogged roots cause rapid root rot and total
crop loss. Drainage improvement is more expensive than compost and should
be prioritised when drainage is critically low (below 30%).

### Plant Now
Planting should only occur when all three conditions are met:
pH in range, organic matter ≥ 60%, drainage quality ≥ 60%.
Planting with all three conditions met yields maximum reward (+30).
Planting with two conditions met is acceptable but suboptimal (+8).
Planting with only one condition met carries high crop failure risk (-10).
Planting with zero conditions met will likely destroy the crop (-25).

The agent must not delay planting indefinitely to chase perfect conditions.
If the planting window is closing (days_remaining ≤ 5) and at least two
conditions are met, planting is preferable to missing the window entirely,
which incurs a -20 penalty on top of any preparation reward.

## Priority Order When Multiple Actions Are Needed
1. Fix drainage first if drainage quality < 30 (critical risk)
2. Fix pH second if outside range (blocks nutrient uptake)
3. Add compost third if organic matter < 60
4. Plant when all three targets are met or window is closing

## Risk Thresholds
- Drainage < 30%: Critical. Improve drainage immediately.
- pH < 4.5 or pH > 8.0: Severe. pH adjustment is urgent.
- Organic matter < 20%: Poor. Compost is needed urgently.
- days_remaining ≤ 3: Plant regardless of remaining gaps.
