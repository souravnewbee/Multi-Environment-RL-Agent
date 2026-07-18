# Intersection Signal Control Policy
UMORDA Traffic Domain — Single Traffic Intersection Control

## Signal Priority Criteria
The traffic signal should always prioritize the direction with the highest cumulative wait time, not just the highest car count. A direction that has been waiting longer has greater urgency regardless of how many cars are present. When both directions have equal wait times, the direction with more cars should be served first.

## Wait Time Thresholds
- Wait time 0 to 2 steps: low urgency, signal may stay as is.
- Wait time 3 to 5 steps: moderate urgency, consider switching if other direction is lower.
- Wait time 6 to 7 steps: high urgency, switch signal immediately.
- Wait time 8 or more steps: critical, safety override forces signal switch regardless of Q-values.

## Minimum Green Duration
A signal must remain green for at least 3 steps before switching. Switching too early causes driver confusion, increases accident risk at the moment of transition, and reduces overall throughput because vehicles need time to react and clear the intersection. The agent should not flip signals rapidly back and forth.

## Traffic Load Guidelines
- Total cars 0 to 3: light traffic, low priority for switching.
- Total cars 4 to 6: moderate traffic, normal signal timing applies.
- Total cars 7 or more: heavy traffic, prioritize clearing the busier direction quickly.

## Safety Override Rule
If any direction has been waiting for 8 or more steps, the system must force the signal to serve that direction immediately. This is a hard guarantee that cannot be overridden by Q-values or any other consideration. No direction should ever be starved indefinitely.

## Switching Cost
Every signal switch carries a cost in terms of lost throughput and driver reaction time. Unnecessary switches should be avoided. A switch is justified only when the waiting direction has clearly higher urgency based on wait time and car count combined.
