# Trading Agent Policy
## Domain: Finance | Task: trading

## Overview
The trading agent makes Buy, Sell, or Hold decisions on a single asset
over a 60-step episode using real or synthetic market price data.
The agent observes price trend, shares held, cash available, and total
portfolio value. The goal is to grow portfolio value by buying in
uptrends and selling before or during downtrends, while avoiding
costly mistakes such as buying into crashes or selling during bull runs.

## Core Principle: Trade With the Trend
The fundamental rule of the trading agent is to align decisions with
the current price trend. Buying in an uptrend and selling in a
downtrend maximises returns. Going against the trend is penalised
because it reflects poor market timing and results in losses.

## Price Trend Scale
| trend value | Market Condition | Primary Action       |
|-------------|------------------|----------------------|
| +2 (BULL)   | Strong bull run  | Buy or Hold          |
| +1 (UP)     | Uptrend          | Buy or Hold          |
|  0 (FLAT)   | Sideways market  | Hold or neutral Buy  |
| -1 (DOWN)   | Downtrend        | Sell                 |
| -2 (CRASH)  | Market crash     | Sell immediately     |

## Action Rules

### Buy
Buying is correct when price trend is positive (UP or BULL).
Buying in an uptrend captures price appreciation and generates
performance reward proportional to trend strength.

Buying in a FLAT market is marginally acceptable — the market
may begin trending upward.

Buying in a DOWN or CRASH market is penalised because the agent
is purchasing an asset that is losing value. The steeper the
downtrend, the heavier the penalty.

Buying must not be attempted when cash is insufficient to cover
the current share price. Attempting to buy without funds incurs
a small penalty for the wasted action.

### Sell
Selling is correct when price trend is negative (DOWN or CRASH).
Selling during a downtrend locks in value before further losses
and generates a performance reward proportional to trend severity.
Selling during a CRASH is the highest-reward sell action.

Selling in a FLAT market is neutral — acceptable but not optimal.

Selling during an UP or BULL market is penalised as premature —
the agent is giving up future gains by exiting too early.

Selling must not be attempted when no shares are held.
Attempting to sell with zero shares incurs a small penalty.

### Hold
Holding is correct when shares are held and the market is trending
upward — the agent earns passive gains proportional to the number
of shares held and the trend strength.

Holding during a CRASH or DOWN trend causes losses proportional to
shares held and trend severity. The agent should not hold passively
into a downtrend when it has the option to sell.

Holding with zero shares in any market condition produces no
reward or penalty — it is a wasted step.

## Portfolio Management Rules
- Never hold zero cash and zero shares simultaneously (fully exposed)
- Prefer selling in DOWN before buying in any new position
- Do not buy maximum shares in a single step — preserve cash buffer
- Trend signals are short-term (5-day window) — do not overtrade

## Risk Thresholds
- trend = -2 and shares > 0: Sell immediately regardless of cash level
- trend = +2 and cash available: Buy if not already heavily invested
- cash < current share price: Hold only, cannot buy
- shares = 0 and trend ≤ 0: Hold cash, do not buy into downtrend
