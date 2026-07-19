# Pest Control and Treatment Allocation Policy
## Domain: Agriculture | Task: pest_control

## Overview
The pest control agent manages a limited treatment resource budget across
multiple farm plots over a growing season. Each step represents a decision
about how to allocate treatment resources to a plot facing a pest outbreak.
The agent must prioritise urgent outbreaks, conserve resources for later
plots, and avoid both over-allocation and neglect of critical threats.

## Core Principle: Urgent Outbreaks Take Priority
Urgent outbreaks must always be addressed before non-urgent ones.
An untreated urgent outbreak spreads to adjacent plots, compounds
crop damage exponentially, and incurs a severe fairness penalty because
the affected plot's crop is sacrificed through neglect.
This mirrors the ER triage principle from hospital policy:
critical cases are served first regardless of resource pressure.

## Action Rules

### Full Treatment
Full treatment should be applied when:
- An urgent outbreak is present AND sufficient resources remain, OR
- Resources are plentiful (remaining > 60% of total) and the plot
  needs complete treatment to prevent spread

Full treatment of an urgent outbreak yields the highest performance
reward (+15) and fairness reward (+5). Full treatment of a non-urgent
plot when resources are tight is suboptimal — partial treatment
preserves more budget for remaining plots.

Full treatment must not be applied when remaining resources are
insufficient — attempting to fully treat beyond budget incurs a hard
penalty (-10). The agent must check remaining resources before
choosing Full Treatment.

### Partial Treatment
Partial treatment is the recommended action when:
- Resources are limited (remaining 30–60% of total) AND
- The outbreak is non-urgent, OR
- An urgent outbreak exists but full treatment would exhaust
  resources needed for remaining plots

Partial treatment of non-urgent outbreaks is smart resource management
(+4 performance, +5 cost). Partial treatment of urgent outbreaks is
acceptable but not ideal (-2 fairness penalty) — it reduces the
outbreak without eliminating it entirely.

### Defer
Deferring treatment is only acceptable when:
- The outbreak is non-urgent AND
- Multiple plots remain to be treated AND
- Resources are critically low (remaining < 20% of total)

Deferring an urgent outbreak is the worst possible action (-15
performance, -8 fairness). Even when resources are severely limited,
partial treatment of an urgent outbreak is always preferable to
deferring it entirely.

Deferring the last plot regardless of urgency is also penalised (-8)
as it indicates poor resource planning across the season.

## Resource Management Thresholds
| Remaining Resources | Guidance                                        |
|---------------------|-------------------------------------------------|
| > 60% of total      | Full treat freely, prioritise urgent plots      |
| 30–60% of total     | Prefer partial for non-urgent, full for urgent  |
| 10–30% of total     | Partial treatment only; defer non-urgent        |
| < 10% of total      | Defer non-urgent; partial-treat urgent only     |
| 0                   | Cannot treat — all further plots must be deferred |

## Over-Budget Rule
Spending more than 105% of the total resource budget incurs a severe
penalty (-20). The agent must track cumulative resource usage and
ensure total spend does not exceed the budget. This is a hard constraint.

## Clean Finish Bonus
If the agent reaches the last plot with remaining resources and no
urgent outbreaks pending, it receives a clean finish bonus (+8).
This rewards efficient resource planning across the full season —
not just individual plot decisions.

## Risk Thresholds
- urgent_outbreaks > 3: Resource strain — prioritise urgent plots only
- plots_remaining = 1: Final plot — conserve nothing, use what is needed
- remaining < 10% of total: Critical conservation mode
- urgent_outbreaks > 0 and remaining < partial_cost: Escalate to supervisor
