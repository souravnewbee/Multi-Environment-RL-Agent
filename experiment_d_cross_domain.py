"""
Experiment D — Cross-Domain Pattern Validation
================================================
Tests whether a single "triage under scarce resources" decision pattern
generalises across three structurally unrelated domains.

Tasks:
  hospital / er_queue     — emergency vs normal patients
  finance  / budget       — urgent vs non-urgent department requests
  agriculture/pest_control — urgent vs non-urgent outbreaks

Two outputs:

1. Structural comparison table (PNG)
   Side-by-side listing of reward rules showing the shared abstract pattern.

2. Decision boundary chart (PNG)
   For each task: at what % resource remaining does the trained Q-table
   switch from Full → Partial → Defer, for a given urgency level?
   All three overlaid on one normalised x-axis (% resource remaining).

No new training. Uses existing Q-tables only.
"""

import sys, os, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sqlite3

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from agents.hospital_agent     import discretize as hosp_disc
from agents.finance_agent      import FinanceAgent
from agents.agriculture_agent  import discretize as agri_disc
from qtable_store              import load_qtable


# ══════════════════════════════════════════════════════════════════════════════
# Load Q-tables
# ══════════════════════════════════════════════════════════════════════════════

def load_from_db(name):
    db   = os.path.join(ROOT, "qtables", "qtables.db")
    conn = sqlite3.connect(db)
    row  = conn.execute("SELECT shape, data FROM qtables WHERE name=?", (name,)).fetchone()
    conn.close()
    shape = tuple(int(x) for x in row[0].split(","))
    return np.frombuffer(row[1], dtype=np.float64).reshape(shape)


Q_er     = load_from_db("hospital_er_queue")      # shape (6,6,2)
Q_budget = load_from_db("finance_budget")          # shape (5,5,5,3)
Q_pest   = load_qtable("agriculture_pest_control") # shape (5,3,3,3)


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 — Structural Comparison Table
# ══════════════════════════════════════════════════════════════════════════════

print("Building structural comparison table...")

col_headers = ["Dimension", "ER Queue\n(Hospital)", "Budget\n(Finance)", "Pest Control\n(Agriculture)"]

rows = [
    ["State vars",
     "emergency_queue\nnormal_queue",
     "urgent_requests\namount_spent\ntotal_budget\ndepts_remaining",
     "urgent_outbreaks\nresource_used\ntotal_resource\nplots_remaining"],

    ["Actions",
     "Serve Emergency\nServe Normal",
     "Allocate Full\nAllocate Partial\nDefer",
     "Full Treatment\nPartial Treatment\nDefer"],

    ["Urgent-present\n→ serve first\n(reward)",
     "Serve Emerg: +10\nServe Normal: −5",
     "Full urgent: +15 perf\n+5 fair\nDefer urgent: −15 perf\n−8 fair",
     "Full urgent: +15 perf\n+5 fair\nDefer urgent: −15 perf\n−8 fair"],

    ["Resource low\n→ conserve",
     "N/A (no cost\ncomponent)",
     "Partial non-urgent:\n+5 cost\nFull non-urgent: −1 cost",
     "Partial non-urgent:\n+5 cost\nFull non-urgent: −1 cost"],

    ["Defer urgent\n→ heaviest penalty",
     "−5 reward\n(served normal\nwhile emerg waiting)",
     "−15 perf\n−8 fair",
     "−15 perf\n−8 fair"],

    ["Shared abstract\npattern",
     "✓ Serve critical\nfirst, always",
     "✓ Serve critical\nfirst, always",
     "✓ Serve critical\nfirst, always"],
]

fig, ax = plt.subplots(figsize=(14, 7))
ax.axis("off")

tbl = ax.table(
    cellText=rows,
    colLabels=col_headers,
    loc="center",
    cellLoc="center",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(8.5)
tbl.scale(1.0, 2.6)

# Header row styling
for col_idx in range(len(col_headers)):
    tbl[0, col_idx].set_facecolor("#1565C0")
    tbl[0, col_idx].set_text_props(color="white", fontweight="bold")

# Domain column colours
domain_colors = ["#E3F2FD", "#E8F5E9", "#FFF8E1"]
for row_idx in range(1, len(rows) + 1):
    tbl[row_idx, 0].set_facecolor("#F5F5F5")
    tbl[row_idx, 0].set_text_props(fontweight="bold")
    for col_idx, color in enumerate(domain_colors, start=1):
        tbl[row_idx, col_idx].set_facecolor(color)

# Highlight the shared pattern row
for col_idx in range(len(col_headers)):
    tbl[len(rows), col_idx].set_facecolor("#B2DFDB")
    tbl[len(rows), col_idx].set_text_props(fontweight="bold")

ax.set_title(
    "Experiment D — Cross-Domain Structural Comparison\n"
    "All three tasks share the same abstract triage-under-scarcity pattern",
    fontsize=12, fontweight="bold", pad=20
)

tbl_path = os.path.join(ROOT, "experiment_d_structural_table.png")
plt.savefig(tbl_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved → {tbl_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 — Decision Boundary Chart
#
# For each task, sweep % resource remaining (0→100%) at a fixed urgency=1.
# Ask the Q-table: what action does it choose?
# Plot the action as a function of % resource remaining.
# Overlay all three on one normalised x-axis.
# ══════════════════════════════════════════════════════════════════════════════

print("Computing decision boundaries...")

PCT_STEPS = 100   # 1% increments

# ── ER Queue ─────────────────────────────────────────────────────────────────
# State: (emergency_queue_bin, normal_queue_bin)
# emergency_queue: 0-20 → bins 0-5 (step 4); we fix emergency=1 (bin=0)
# "resource" analogue: normal_queue size (when it's large, more normal waiting)
# But ER doesn't have a budget — we use normal_queue fill % as the x-axis proxy
# emergency bin 1 = some emergency present; sweep normal_queue bin 0-5
er_actions  = []  # 0=Serve Emergency, 1=Serve Normal
er_pcts     = []
for normal_bin in range(6):
    emerg_bin = 1   # urgency present
    action = int(np.argmax(Q_er[emerg_bin, normal_bin]))
    # x: normal_queue fill pct (0=empty, 5=full)
    pct = normal_bin / 5.0 * 100
    er_actions.append(action)
    er_pcts.append(pct)

# ── Budget ────────────────────────────────────────────────────────────────────
# State: (used_pct_bin, urgent_bin, depts_bin) → shape (5,5,5,3)
# Fix urgent_bin=1 (urgent present), depts_bin=2 (mid-episode)
# Sweep used_pct_bin 0→4, convert to % remaining = (1 - used/5)*100
budget_actions = []
budget_pcts    = []
agent_budget   = FinanceAgent(task="budget")
for used_bin in range(5):
    urgent_bin = 1
    depts_bin  = 2
    action = int(np.argmax(Q_budget[used_bin, urgent_bin, depts_bin]))
    pct_remaining = (1.0 - used_bin / 4.0) * 100
    budget_actions.append(action)
    budget_pcts.append(pct_remaining)

# ── Pest Control ──────────────────────────────────────────────────────────────
# State: (used_pct_bin, urgent_bin, plots_bin) → shape (5,3,3,3)
# Fix urgent_bin=1, plots_bin=1 (mid-season)
# Sweep used_pct_bin 0→4
pest_actions = []
pest_pcts    = []
for used_bin in range(5):
    urgent_bin = 1
    plots_bin  = 1
    action = int(np.argmax(Q_pest[used_bin, urgent_bin, plots_bin]))
    pct_remaining = (1.0 - used_bin / 4.0) * 100
    pest_actions.append(action)
    pest_pcts.append(pct_remaining)


# ── Print boundary summary ────────────────────────────────────────────────────
def describe_boundary(pcts, actions, action_names, task):
    print(f"\n  [{task}] decision by % resource remaining:")
    for p, a in zip(sorted(zip(pcts, actions)), []):
        pass
    for p, a in sorted(zip(pcts, actions)):
        print(f"    {p:5.1f}%  →  {action_names[a]}")

describe_boundary(er_pcts,     er_actions,     ["Serve Emergency","Serve Normal"],                  "er_queue")
describe_boundary(budget_pcts, budget_actions, ["Allocate Full","Allocate Partial","Defer"],         "budget")
describe_boundary(pest_pcts,   pest_actions,   ["Full Treatment","Partial Treatment","Defer"],        "pest_control")


# ── Plot ──────────────────────────────────────────────────────────────────────

# Map action index → normalised label (0=Aggressive/Full, 1=Partial, 2=Defer)
# ER only has 2 actions: 0=Serve Emerg (Full), 1=Serve Normal (Defer)
er_norm     = [0 if a == 0 else 2 for a in er_actions]
budget_norm = budget_actions   # already 0=Full, 1=Partial, 2=Defer
pest_norm   = pest_actions     # already 0=Full, 1=Partial, 2=Defer

fig, ax = plt.subplots(figsize=(10, 5))

COLORS = {
    "er_queue":     "#1565C0",
    "budget":       "#2E7D32",
    "pest_control": "#E65100",
}
MARKERS = {
    "er_queue":     "o",
    "budget":       "s",
    "pest_control": "^",
}

# Scatter plots — x = % resource remaining, y = action level (0=Full,1=Partial,2=Defer)
ax.scatter(er_pcts,     er_norm,     color=COLORS["er_queue"],
           marker=MARKERS["er_queue"],     s=120, zorder=3, label="ER Queue (Hospital)")
ax.scatter(budget_pcts, budget_norm, color=COLORS["budget"],
           marker=MARKERS["budget"],       s=120, zorder=3, label="Budget (Finance)")
ax.scatter(pest_pcts,   pest_norm,   color=COLORS["pest_control"],
           marker=MARKERS["pest_control"], s=120, zorder=3, label="Pest Control (Agriculture)")

# Step lines
for pcts, norm, color in [
    (er_pcts,     er_norm,     COLORS["er_queue"]),
    (budget_pcts, budget_norm, COLORS["budget"]),
    (pest_pcts,   pest_norm,   COLORS["pest_control"]),
]:
    sorted_pairs = sorted(zip(pcts, norm))
    xs = [p[0] for p in sorted_pairs]
    ys = [p[1] for p in sorted_pairs]
    ax.step(xs, ys, where="post", color=color, alpha=0.5, linewidth=1.5)

# Threshold line at ~20% (common conservation threshold)
ax.axvline(x=20, color="gray", linestyle="--", alpha=0.6, label="~20% resource threshold")

ax.set_yticks([0, 1, 2])
ax.set_yticklabels(["Full / Serve Urgent\n(most aggressive)", "Partial", "Defer\n(most conservative)"],
                   fontsize=9)
ax.set_xlabel("% Resource Remaining (normalised across domains)", fontsize=10)
ax.set_ylabel("Agent Decision Level", fontsize=10)
ax.set_title(
    "Experiment D — Cross-Domain Decision Boundaries\n"
    "Fixed urgency=1, mid-episode; x-axis normalised to % resource remaining",
    fontsize=11, fontweight="bold"
)
ax.invert_xaxis()   # left=high resource, right=low resource (more intuitive)
ax.set_xlim(105, -5)
ax.legend(fontsize=9, loc="upper right")
ax.grid(axis="y", alpha=0.3)

boundary_path = os.path.join(ROOT, "experiment_d_decision_boundary.png")
plt.tight_layout()
plt.savefig(boundary_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\n  Saved → {boundary_path}")
print("\nExperiment D complete.")