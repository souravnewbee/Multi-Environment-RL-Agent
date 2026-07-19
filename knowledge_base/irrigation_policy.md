# Irrigation Management Policy
## Domain: Agriculture | Task: irrigation

## Overview
The irrigation agent manages water delivery to an exotic fruit crop over
a growing season. The goal is to keep soil moisture within the crop's
optimal range, maintain a healthy water reservoir for later in the season,
and avoid both over-irrigation (root rot, waterlogging) and under-irrigation
(crop stress, yield loss). Rainfall trend is a key signal — the agent must
learn to anticipate rainfall rather than always irrigating reactively.

## Core Principle: Soil Moisture Target
The optimal soil moisture range for the current fruit profile must be
maintained. Moisture that is too low causes crop stress. Moisture that
is too high causes waterlogging and root rot, which is equally damaging.
The agent must manage both extremes.

## Action Rules

### Irrigate Heavy
Heavy irrigation should only be applied when:
- Crop stress is high (above 40) AND
- Water reservoir has at least 15 units remaining AND
- Rainfall trend is neutral or negative (no rain expected)

Heavy irrigation when the crop is not stressed is wasteful and depletes
the reservoir unnecessarily. Heavy irrigation when the reservoir is nearly
empty risks leaving the crop with no water buffer later in the season.
Heavy irrigation during or immediately before heavy rainfall causes
waterlogging.

### Irrigate Light
Light irrigation is the preferred action for routine moisture maintenance.
It should be applied when:
- Crop stress is moderate (20–40) OR
- Soil moisture is approaching the lower bound of the target range AND
- Water reservoir has at least 6 units

Light irrigation is preferred over heavy irrigation in most situations
because it is less costly, less likely to cause waterlogging, and
preserves reservoir capacity for later stress events.

### Skip Irrigation
Skipping irrigation is correct when:
- Rainfall trend is positive (rain is coming) AND/OR
- Crop stress is low (below 20) AND
- Soil moisture is within or above the target range

Skipping irrigation during a dry spell (rainfall_trend ≤ -1) when crop
stress is already elevated is dangerous and increases stress further.
The agent should not skip irrigation to conserve water when the crop
is already stressed — this trades a small water saving for a large
yield loss.

## Rainfall Trend Interpretation
| rainfall_trend | Meaning            | Irrigation Guidance                    |
|----------------|--------------------|----------------------------------------|
| -2             | Severe drought     | Irrigate heavily, conserve reservoir   |
| -1             | Dry spell          | Irrigate lightly, monitor stress       |
|  0             | Neutral            | Irrigate based on crop stress level    |
| +1             | Light rain coming  | Skip or light irrigate only            |
| +2             | Heavy rain coming  | Skip irrigation entirely               |

## Reservoir Management
The water reservoir is a finite resource that does not fully replenish
unless rainfall is heavy. The agent must budget reservoir use across the
full season. Running out of water reservoir with high crop stress and no
rainfall is the worst possible outcome — it results in sustained critical
stress that cannot be relieved.

- Reservoir > 60: Safe to irrigate freely based on crop need
- Reservoir 30–60: Moderate conservation — prefer light over heavy
- Reservoir 10–30: High conservation — only irrigate under high stress
- Reservoir < 10: Critical — skip unless crop stress is critical (≥ 70)

## Critical Stress Rule
If crop stress reaches or exceeds 90, the crop is close to dying.
At this point the agent must irrigate immediately regardless of
reservoir level or rainfall forecast. A dead crop yields nothing.

## Risk Thresholds
- Crop stress ≥ 90: Critical — irrigate immediately
- Reservoir < 10 and rainfall_trend ≤ -1: Emergency — strict conservation
- Rainfall trend ≥ 2: Never irrigate — waterlogging risk
