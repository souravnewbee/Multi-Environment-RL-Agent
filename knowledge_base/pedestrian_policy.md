# Pedestrian Crossing Control Policy
UMORDA Traffic Domain — Pedestrian Crossing Management

## Safety Priority Principle
Pedestrian safety is the highest priority in crossing control. A pedestrian waiting to cross is always at greater personal risk than a vehicle waiting in a queue. When pedestrian wait time is high, the signal must switch to allow crossing regardless of vehicle queue size. Vehicle throughput is secondary to pedestrian safety.

## Pedestrian Wait Time Thresholds
- Wait time 0 to 2 steps: low urgency, vehicles may continue if queue is high.
- Wait time 3 to 4 steps: moderate urgency, begin prioritizing pedestrians.
- Wait time 5 steps: high urgency, switch to pedestrian phase immediately.
- Wait time 6 or more steps: critical safety threshold, system forces pedestrian phase. This is a hard guarantee.

## Vehicle Wait Time Guidelines
- Wait time 0 to 3 steps: low urgency, pedestrians take priority if waiting.
- Wait time 4 to 6 steps: moderate urgency, balance against pedestrian needs.
- Wait time 7 or more steps: high vehicle urgency, but pedestrian safety still takes precedence if ped wait is also high.

## Phase Duration
Each phase should remain active for at least 2 steps before switching. Switching too rapidly causes confusion for both pedestrians and drivers and increases accident risk at the moment of transition.

## Rejection of Vehicle Priority Over Safety
Under no circumstances should vehicle queue size alone justify ignoring pedestrians who have been waiting beyond the safety threshold. Even if 9 vehicles are waiting, a pedestrian who has waited 6 or more steps must be served immediately. Traffic inconvenience does not outweigh personal safety risk.

## Empty Crossing Rule
If no pedestrians are waiting, the pedestrian phase should not be activated unnecessarily. Allowing vehicles to flow freely when the crossing is empty improves overall traffic efficiency.
