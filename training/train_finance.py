"""
training/train_finance.py
=========================
Trains Q-learning agents for all 3 Finance tasks.
Matches the exact structure of train_hospital.py.

Saves:
    qtables/finance_trading.npy
    qtables/finance_savings.npy
    qtables/finance_budget.npy
    qtables/finance_trading_metadata.json
    qtables/finance_savings_metadata.json
    qtables/finance_budget_metadata.json

Run from repo root:
    python training/train_finance.py
"""

import sys
import os
import json
import time
import numpy as np
import random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from environments.finance_env import FinanceEnv

# ── Hyperparameters (match train_hospital.py exactly) ─────────────────────────
EPISODES      = 50_000
ALPHA         = 0.1
GAMMA         = 0.9
EPS_START     = 1.0
EPS_MIN       = 0.01
EPS_DECAY     = 0.9995
EVAL_EPISODES = 500
LOG_EVERY     = 5_000

# Tickers to rotate during trading training
TICKERS = ["AAPL", "TSLA", "SPY", "MSFT"]


# ── Discretise (task-specific, same as demo_finance.py) ───────────────────────
def discretize(state, task):
    if task == "trading":
        trend  = int(state["price_trend"]) + 2              # 0-4
        shares = min(int(state["shares_held"]), 15) // 4    # 0-3
        cash_buckets = [0, 200, 500, 1000, 1500, 2000]
        cash = next((i for i, b in enumerate(cash_buckets)
                     if state["cash"] <= b), 5)
        return (
            int(np.clip(trend,  0, 4)),
            int(np.clip(shares, 0, 3)),
            int(np.clip(cash,   0, 5)),
        )

    elif task == "savings":
        income  = min(int(state["monthly_income"]),  150) // 30   # 0-5
        savings = min(int(state["current_savings"]), 500) // 125  # 0-4
        months  = min(int(state["months_remaining"]), 12) // 3    # 0-4
        return (
            int(np.clip(income,  0, 5)),
            int(np.clip(savings, 0, 4)),
            int(np.clip(months,  0, 4)),
        )

    elif task == "budget":
        total    = int(state["total_budget"])
        spent    = int(state["amount_spent"])
        used_pct = min(int((spent / max(total, 1)) * 5), 4)
        urgent   = min(int(state["urgent_requests"]), 4)
        depts    = min(int(state["departments_remaining"]), 4)
        return (
            int(np.clip(used_pct, 0, 4)),
            int(np.clip(urgent,   0, 4)),
            int(np.clip(depts,    0, 4)),
        )


def get_q_shape(task):
    if task == "trading":
        return (5, 4, 6, 3)    # trend, shares, cash, actions
    elif task == "savings":
        return (6, 5, 5, 3)    # income, savings, months, actions
    elif task == "budget":
        return (5, 5, 5, 3)    # used_pct, urgent, depts, actions


# ── Smooth helper for plotting ─────────────────────────────────────────────────
def smooth(arr, window=500):
    if len(arr) < window:
        return arr
    return np.convolve(arr, np.ones(window) / window, mode="valid")


# ── Single task training ───────────────────────────────────────────────────────
def train_task(task):
    print(f"\n{'='*60}")
    print(f"  Training Finance — {task.upper()}")
    print(f"  Episodes: {EPISODES:,} | Alpha: {ALPHA} | Gamma: {GAMMA}")
    print(f"{'='*60}")

    Q         = np.zeros(get_q_shape(task))
    eps       = EPS_START
    rewards   = []
    start     = time.time()

    for ep in range(EPISODES):
        # Rotate tickers for trading task every episode
        if task == "trading":
            ticker = TICKERS[ep % len(TICKERS)]
            env    = FinanceEnv(task=task, ticker=ticker, use_real_data=True)
        else:
            if ep == 0:
                env = FinanceEnv(task=task)

        state    = env.reset()
        total_r  = 0.0
        done     = False

        while not done:
            s = discretize(state, task)

            # Epsilon-greedy
            if random.random() < eps:
                action = random.randint(0, env.n_actions - 1)
            else:
                action = int(np.argmax(Q[s]))

            next_state, reward, done, _ = env.step(action)
            s_ = discretize(next_state, task)

            # Bellman update
            Q[s][action] += ALPHA * (
                reward + GAMMA * np.max(Q[s_]) - Q[s][action]
            )

            state    = next_state
            total_r += reward

        eps = max(EPS_MIN, eps * EPS_DECAY)
        rewards.append(total_r)

        if (ep + 1) % LOG_EVERY == 0:
            window_avg = np.mean(rewards[-LOG_EVERY:])
            elapsed    = time.time() - start
            print(f"  Ep {ep+1:>6,} | "
                  f"avg reward (last {LOG_EVERY:,}): {window_avg:+.1f} | "
                  f"eps: {eps:.4f} | "
                  f"elapsed: {elapsed:.0f}s")

    # ── Evaluate trained vs random ────────────────────────────────────────────
    print(f"\n  Evaluating over {EVAL_EPISODES} episodes...")
    trained_scores = []
    random_scores  = []

    for _ in range(EVAL_EPISODES):
        if task == "trading":
            eval_env = FinanceEnv(task=task, ticker="AAPL", use_real_data=True)
        else:
            eval_env = FinanceEnv(task=task)

        # Trained
        state = eval_env.reset()
        total, done = 0.0, False
        while not done:
            s      = discretize(state, task)
            action = int(np.argmax(Q[s]))
            state, r, done, _ = eval_env.step(action)
            total += r
        trained_scores.append(total)

        # Random
        state = eval_env.reset()
        total, done = 0.0, False
        while not done:
            action = random.randint(0, eval_env.n_actions - 1)
            state, r, done, _ = eval_env.step(action)
            total += r
        random_scores.append(total)

    trained_mean = np.mean(trained_scores)
    random_mean  = np.mean(random_scores)
    improvement  = ((trained_mean - random_mean) / max(abs(random_mean), 1)) * 100

    print(f"\n  ✓ Trained agent : {trained_mean:+.2f}")
    print(f"  ✗ Random agent  : {random_mean:+.2f}")
    print(f"  ↑ Improvement   : {improvement:+.1f}%")

    total_time = time.time() - start
    print(f"  ⏱ Total time    : {total_time:.1f}s")

    return Q, rewards, trained_mean, random_mean, improvement, total_time


# ── Save Q-table + metadata JSON ──────────────────────────────────────────────
def save_results(task, Q, rewards, trained, baseline, improvement, elapsed):
    os.makedirs("qtables", exist_ok=True)
    os.makedirs("training/plots", exist_ok=True)

    # Save Q-table
    q_path = f"qtables/finance_{task}.npy"
    np.save(q_path, Q)
    print(f"\n  Saved Q-table  → {q_path}")

    # Save metadata JSON (matches hospital format)
    env       = FinanceEnv(task=task) if task != "trading" else FinanceEnv(task=task, ticker="AAPL")
    env_info  = env.get_info()
    metadata  = {
        "domain":           "finance",
        "task":             task,
        "episodes":         EPISODES,
        "alpha":            ALPHA,
        "gamma":            GAMMA,
        "epsilon_start":    EPS_START,
        "epsilon_min":      EPS_MIN,
        "epsilon_decay":    EPS_DECAY,
        "q_table_shape":    list(Q.shape),
        "state_vars":       env_info["state_vars"],
        "actions":          env_info["actions"],
        "n_actions":        env_info["n_actions"],
        "trained_mean":     round(float(trained), 4),
        "random_mean":      round(float(baseline), 4),
        "improvement_pct":  round(float(improvement), 2),
        "training_time_s":  round(float(elapsed), 1),
        "final_epsilon":    round(float(EPS_MIN), 4),
        "avg_last_5k":      round(float(np.mean(rewards[-5000:])), 4),
        "avg_first_5k":     round(float(np.mean(rewards[:5000])), 4),
        "tickers_used":     TICKERS if task == "trading" else None,
        "data_source":      "yfinance + GBM fallback" if task == "trading" else "simulation",
    }

    json_path = f"qtables/finance_{task}_metadata.json"
    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved metadata → {json_path}")

    # Save reward curve plot
    plot_path = f"training/plots/finance_{task}_reward_curve.png"
    fig, ax   = plt.subplots(figsize=(10, 4))
    ax.plot(rewards, alpha=0.12, color="steelblue")
    ax.plot(smooth(rewards), color="steelblue", linewidth=2, label="Smoothed reward")
    ax.axhline(trained,  color="green", linestyle="--", linewidth=1.5,
               label=f"Trained avg: {trained:+.1f}")
    ax.axhline(baseline, color="red",   linestyle="--", linewidth=1.5,
               label=f"Random avg:  {baseline:+.1f}")
    ax.set_title(f"Finance — {task} | Q-Learning Training Curve",
                 fontweight="bold")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Episode Reward")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"  Saved plot     → {plot_path}")

    return metadata


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "=" * 60)
    print("  UMORDA — Finance Domain Training")
    print(f"  Tasks: trading, savings, budget")
    print(f"  Episodes per task: {EPISODES:,}")
    print("=" * 60)

    all_metadata = {}

    for task in ["trading", "savings", "budget"]:
        Q, rewards, trained, baseline, improvement, elapsed = train_task(task)
        meta = save_results(task, Q, rewards, trained, baseline, improvement, elapsed)
        all_metadata[task] = meta

    # ── Combined summary ──────────────────────────────────────────────────────
    print(f"\n\n{'='*65}")
    print(f"{'FINANCE DOMAIN — FINAL TRAINING SUMMARY':^65}")
    print(f"{'='*65}")
    print(f"{'Task':<12} {'Trained':>10} {'Random':>10} {'Improve%':>10} {'Time(s)':>10}")
    print(f"{'-'*65}")
    for task, meta in all_metadata.items():
        print(f"{task:<12} "
              f"{meta['trained_mean']:>+10.1f} "
              f"{meta['random_mean']:>+10.1f} "
              f"{meta['improvement_pct']:>+9.1f}% "
              f"{meta['training_time_s']:>10.1f}")
    print(f"{'='*65}")

    # Save combined summary JSON
    summary_path = "qtables/finance_training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_metadata, f, indent=2)
    print(f"\n  Combined summary saved → {summary_path}")
    print("\n  All finance Q-tables ready.")
    print("  Run:  python view_qtable_finance.py  to inspect them.")
    print("  Run:  python demo_finance.py  to test interactively.\n")


if __name__ == "__main__":
    main()
