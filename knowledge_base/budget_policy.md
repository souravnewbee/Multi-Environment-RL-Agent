# Budget Allocation Policy
## Domain: Finance | Task: budget

## Overview
The budget allocation agent distributes a fixed financial budget across
multiple departments over a planning period. Each step represents one
department submitting a funding request. The agent must decide whether
to fund the request fully, fund it partially, or defer it to a later
period. The agent must prioritise urgent requests, avoid overspending,
and ensure that all departments receive at least partial consideration
before the budget is exhausted.

## Core Principle: Urgent Requests Must Be Funded First
Urgent requests represent time-critical operational needs that cannot
wait. Deferring an urgent request incurs the heaviest possible penalty
(-15 performance, -8 fairness) because it risks operational failure
in the affected department. This mirrors the ER triage principle:
critical needs are always served before routine ones.

## Funding Criteria

### Allocate Full
Full allocation is correct when:
- A URGENT request is present AND sufficient budget remains, OR
- Budget remaining is above 60% of total AND the request is
  within the normal request range

Full allocation of urgent requests yields the highest reward
(+15 performance, +5 fairness). Full allocation of non-urgent
requests when budget is plentiful is also rewarded (+6 performance).

Full allocation must not be attempted when remaining budget is
less than the request amount. The agent must verify that
remaining = total - spent >= request before choosing Full Allocation.
Attempting to overspend incurs a hard penalty (-10).

### Allocate Partial
Partial allocation (40–70% of the request) is the preferred
action when:
- Budget is constrained (remaining 20–60% of total) AND
- The request is non-urgent, OR
- An urgent request is present but full allocation would
  exhaust the budget entirely, leaving zero for remaining departments

Partial allocation of non-urgent requests is smart planning
(+4 performance, +5 cost). Partial allocation of urgent
requests is acceptable but not ideal (-2 fairness) — it
addresses the need without fully resolving it.

### Defer
Deferring a request is acceptable only when:
- The request is non-urgent AND
- Multiple departments remain to be funded AND
- Budget remaining is critically low (below 20% of total)

Deferring an urgent request is never acceptable, even under
severe budget pressure. Partial treatment of an urgent request
is always preferable to deferral.

Deferring the final department's request is penalised (-8)
regardless of urgency, as it indicates the budget was not
managed efficiently across the full planning period.

## Budget Utilisation Thresholds
| Remaining Budget    | Guidance                                          |
|---------------------|---------------------------------------------------|
| > 60% of total      | Allocate Full for all requests including non-urgent |
| 40–60% of total     | Full for urgent, partial for non-urgent           |
| 20–40% of total     | Partial for all; defer only clearly non-urgent    |
| < 20% of total      | Defer non-urgent; partial for urgent only         |
| 0                   | Must defer all remaining requests                 |

## Over-Budget Rule
Total spending must not exceed 105% of the allocated budget.
Going over budget incurs a severe penalty (-20). The agent
must track cumulative spend (amount_spent) against total_budget
and refuse to allocate when doing so would breach this limit.

## Clean Finish Bonus
If all departments have been processed and the agent finishes
with remaining budget and no unresolved urgent requests, a
clean finish bonus is awarded (+8). This rewards the full-season
view of budget management, not just individual step decisions.

## Risk Thresholds
- urgent_requests > 0 and remaining < 30: Escalate -- use partial treatment
- departments_remaining = 1: Final step -- allocate what is needed, no deferral
- amount_spent > 0.9 × total_budget: Warning -- near budget limit
- amount_spent > total_budget: Hard stop -- over budget penalty triggered
