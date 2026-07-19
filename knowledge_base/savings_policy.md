# Personal Savings Management Policy
## Domain: Finance | Task: savings

## Overview
The savings agent manages a personal financial plan over a 12-month
horizon. Each month it chooses between saving more aggressively,
spending normally, or investing a portion of existing savings.
The goal is to maximise total savings at the end of the period
while avoiding bankruptcy (savings hitting zero) and balancing
the cost of reduced spending quality against long-term financial health.

## Core Principle: Savings Must Never Hit Zero
Running out of savings is the single worst outcome (-15 penalty).
The agent must maintain a positive savings buffer at all times.
Every other objective is secondary to avoiding a zero-savings state.

## Action Rules

### Save More
Save More reduces discretionary expenses to 60% of normal and
deposits the remainder into savings. This is the correct action when:
- Current savings are dangerously low (below 100 units) AND
  income exceeds reduced expenses, OR
- Months remaining are few (≤ 3) and savings target is not met, OR
- Monthly expenses significantly exceed income (net negative cash flow)
  and the agent needs to cut spending to avoid going broke

Save More carries a cost penalty (-2) reflecting reduced quality of
life from aggressive spending cuts. It should not be applied every
month regardless of savings level — when savings are healthy and
income exceeds expenses, Spend Normal or Invest are more efficient.

### Spend Normal
Spend Normal is the balanced default action. It saves 50% of the
net income remaining after normal expenses. This is correct when:
- Income exceeds expenses (positive net cash flow) AND
- Savings are at a comfortable level (above 100 units) AND
- No immediate financial emergency requires aggressive saving

Spend Normal generates moderate performance reward (+4) and a
fairness bonus (+2) reflecting sustainable balanced living.

Spend Normal when expenses exceed income causes the agent to
draw from savings (net negative), incurring a heavy penalty (-8).
The agent must check that income > expenses before choosing
Spend Normal.

### Invest
Investing allocates 30% of current savings into a risky asset.
The outcome is probabilistic — 45% chance of gain, 55% chance
of loss or flat return. Investment is correct when:
- Current savings exceed 200 units (comfortable buffer to risk), AND
- Months remaining are at least 4 (enough time to recover from loss), AND
- Monthly income reliably exceeds monthly expenses

Investing with very low savings (below 10 units available to invest)
is not permitted — the savings are too small to justify the risk.

Investing when savings are low or months are short is dangerous
because a loss with no time to recover may push savings to zero.

## Monthly Cash Flow Assessment
Before selecting any action, the agent should assess net cash flow:
- net = monthly_income - monthly_expenses
- net > 0: Income exceeds expenses — all three actions are viable
- net = 0: Break-even — Save More or Spend Normal only
- net < 0: Deficit — Save More is mandatory to reduce spending

## Savings Level Thresholds
| Savings Level | Guidance                                              |
|---------------|-------------------------------------------------------|
| > 400 units   | Invest or Spend Normal — savings are healthy          |
| 200–400 units | Spend Normal — maintain steady accumulation           |
| 100–200 units | Save More if net is tight; Spend Normal if net is positive |
| < 100 units   | Save More urgently — bankruptcy risk is elevated      |
| 0             | Broke — severe penalty, immediate corrective action   |

## Time Pressure Rules
- months_remaining ≤ 3: Save aggressively regardless of current level
- months_remaining ≤ 1 and savings ≥ 300: Goal achieved (+10 bonus)
- months_remaining = 0: Episode ends — final savings is the terminal score

## Risk Thresholds
- savings < 10: Do not invest — too little to risk
- expenses > income: Do not Spend Normal — deficit spending
- savings = 0: Emergency — Save More immediately
