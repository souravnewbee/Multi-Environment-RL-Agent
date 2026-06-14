# UMORDA
### Universal Multi-Objective Reinforcement Learning Decision Agent

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![Gymnasium](https://img.shields.io/badge/Gymnasium-0.29+-green)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Status](https://img.shields.io/badge/Status-In%20Development-orange)

---

## Overview

UMORDA is a multi-domain AI decision-support system that combines **Large Language Models (LLMs)** and **Reinforcement Learning (RL)** to solve real-world decision-making problems across five independent domains.

The core principle: **the LLM is not the decision-maker.** It acts as an intelligent communication layer, while trained RL agents make the actual decisions.

```
User (natural language prompt)
        ↓
      LLM  ←── router.py + extractor.py  (Groq API / Llama 3)
        ↓
Correct RL Environment  (1 of 15 tasks)
        ↓
Trained Q-Table  →  Optimal Action Selected
        ↓
      LLM  ←── explainer.py  (Groq API / Llama 3)
        ↓
User sees natural language recommendation  (Streamlit Chat UI)
```

Users interact through a **ChatGPT-style Streamlit interface** — they type a real-world situation in plain language, and the system routes it to the correct trained RL agent, which returns an optimized decision explained in natural language.

---

## Key Features

- **Multi-domain RL** — one unified framework operating across 5 unrelated real-world domains
- **Multi-objective rewards** — each task balances multiple competing objectives (cost, efficiency, fairness)
- **User-controlled priority weights** — users assign weights to objectives, producing different policies from the same environment without retraining
- **LLM routing** — natural language prompts are automatically classified and routed to the correct RL environment via Groq API
- **Explainable decisions** — every RL action is converted to a human-readable explanation by Llama 3
- **Q-Learning** — lightweight, interpretable RL algorithm with persistent Q-tables (.npy)

---

## Architecture

### Three-Layer System

| Layer | Component | Role |
|---|---|---|
| Natural Language Layer | `llm/router.py` + `llm/extractor.py` | Classifies user prompt into correct domain + task, extracts state variables |
| RL Decision Layer | `environments/*.py` + `qtables/*.npy` | Trained Q-learning agent selects optimal action via argmax |
| Explanation Layer | `llm/explainer.py` | Converts RL action + Q-values into plain English justification |

### Environment Structure

Each domain is one Gymnasium environment class with 3 task modes. Each task trains its own independent Q-table.

```
5 Domains  ×  3 Tasks  =  15 Independent RL Problems  =  15 Q-Tables
```

---

## Domains & Tasks

### 1. Hospital Domain — `environments/hospital_env.py`

| Task | State Variables | Actions | Objectives |
|---|---|---|---|
| `bed_allocation` | free_beds, waiting_patients | Admit, Reject, Transfer | Throughput, cost, fairness |
| `er_queue` | emergency_queue, normal_queue | Serve Emergency, Serve Normal | Priority, workload balance |
| `staff_allocation` | available_doctors, patient_load | Assign More, Keep Current, Reduce | Treatment speed, cost efficiency |

### 2. Traffic Domain — `environments/traffic_env.py`

| Task | State Variables | Actions | Objectives |
|---|---|---|---|
| `intersection` | cars_N, cars_S, cars_E, cars_W | Green NS, Green EW | Traffic flow, wait time |
| `pedestrian` | waiting_pedestrians, waiting_vehicles | Allow Pedestrians, Allow Vehicles | Safety, wait time |
| `parking` | available_spots, incoming_vehicles | Open Zone A, Open Zone B, Close Entry | Occupancy, congestion |

### 3. Energy Domain — `environments/energy_env.py`

| Task | State Variables | Actions | Objectives |
|---|---|---|---|
| `solar_grid` | solar_output, demand | Use Solar, Use Grid | Cost, sustainability |
| `battery` | battery_level, demand | Charge, Discharge, Idle | Cost savings, battery health |
| `smarthome` | current_load, time_slot | Turn ON, Turn OFF | Energy consumption, comfort |

### 4. Agriculture Domain — `environments/agriculture_env.py`

| Task | State Variables | Actions | Objectives |
|---|---|---|---|
| `irrigation` | soil_moisture, water_reserve | Heavy, Light, No Irrigation | Crop health, water conservation |
| `fertilizer` | crop_health, growth_stage | Apply, Skip | Growth, cost efficiency |
| `pestcontrol` | pest_level, crop_health | Spray, Don't Spray | Crop protection, cost |

### 5. Supply Chain Domain — `environments/supplychain_env.py`

| Task | State Variables | Actions | Objectives |
|---|---|---|---|
| `inventory` | stock_level, demand_forecast | Order Large, Order Small, No Order | Avoid stockout, avoid overstock |
| `supplier` | price, delivery_time, quality | Supplier A, Supplier B, Supplier C | Cost, reliability |
| `warehouse` | items_per_zone, incoming_load | Route Zone A, Route Zone B, Route Zone C | Distribution balance, speed |

---

## Project Structure

```
UMORDA/
│
├── environments/           # Gymnasium environment classes (5 files, 15 task modes)
│   ├── hospital_env.py
│   ├── traffic_env.py
│   ├── energy_env.py
│   ├── agriculture_env.py
│   └── supplychain_env.py
│
├── agents/                 # Q-Learning agent logic per domain
│   ├── hospital_agent.py
│   ├── traffic_agent.py
│   ├── energy_agent.py
│   ├── agriculture_agent.py
│   └── supplychain_agent.py
│
├── qtables/                # Trained Q-tables stored as .npy files (15 total, gitignored)
│
├── training/               # Training scripts per domain
│   ├── train_hospital.py
│   ├── train_traffic.py
│   ├── train_energy.py
│   ├── train_agriculture.py
│   └── train_supplychain.py
│
├── llm/                    # LLM integration layer (Groq API + Llama 3)
│   ├── router.py           # Classifies user prompt → correct domain + task
│   ├── extractor.py        # Extracts state variables from natural language
│   └── explainer.py        # Converts RL action to human-readable response
│
├── config/
│   └── env_registry.py     # Registry of all 15 environments and their metadata
│
├── ui/
│   └── app.py              # Streamlit chat interface
│
├── test_hospital.py        # Unit test for hospital environment
├── view_qtable.py          # Q-table policy viewer
├── demo_hospital.py        # Interactive hospital demo (manual input)
└── main.py                 # Entry point
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.11 |
| RL Framework | Gymnasium 0.29+ |
| RL Algorithm | Q-Learning (tabular) |
| Numerical Computing | NumPy |
| LLM API | Groq API |
| LLM Model | Llama 3 (llama3-8b-8192) |
| UI Interface | Streamlit |
| Visualization | Matplotlib |
| RL Storage | Q-Tables (.npy) |
| Version Control | Git + GitHub |

---

## Q-Learning Implementation

The system uses **tabular Q-Learning** with epsilon-greedy exploration and Bellman equation updates:

```
Q[s][a] ← Q[s][a] + α × (r + γ × max_a' Q[s'][a'] − Q[s][a])
```

### Hyperparameters

| Parameter | Value | Description |
|---|---|---|
| Episodes | 50,000 | Training iterations per task |
| Alpha (α) | 0.1 | Learning rate |
| Gamma (γ) | 0.9 | Discount factor |
| Epsilon (ε) start | 1.0 | Fully exploratory at start |
| Epsilon (ε) min | 0.01 | Minimum exploration rate |
| Epsilon Decay | 0.995 | Rate of exploration reduction per episode |

### Multi-Objective Reward

Each environment balances three competing objectives using user-defined weights:

```
R_total = w1 × r_cost + w2 × r_performance + w3 × r_fairness
```

Where w1 + w2 + w3 = 1.0. Different weight configurations produce different Pareto-optimal policies from the same trained agent without retraining.

---

## Current Progress

### ✅ Week 1 — Hospital Domain Complete

| File | Status | Description |
|---|---|---|
| `environments/hospital_env.py` | ✅ Done | All 3 tasks implemented with state, action, reward |
| `training/train_hospital.py` | ✅ Done | Q-learning training loop |
| `test_hospital.py` | ✅ Done | All 3 tasks tested with terminal output |
| `view_qtable.py` | ✅ Done | Policy viewer — best action for every state |
| `demo_hospital.py` | ✅ Done | Interactive manual input demo |

### 🔄 Remaining Work

- [ ] Traffic environment + training
- [ ] Energy environment + training
- [ ] Agriculture environment + training
- [ ] Supply Chain environment + training
- [ ] LLM routing layer (router.py, extractor.py, explainer.py) — Groq API + Llama 3
- [ ] Streamlit chat UI (app.py)
- [ ] Full system integration (main.py)
- [ ] Multi-objective weighted reward implementation across all envs
- [ ] Final evaluation — trained vs random baseline comparison

---

## How to Run

### 1. Install dependencies
```bash
pip install numpy gymnasium streamlit groq matplotlib
```

### 2. Test the hospital environment
```bash
python test_hospital.py
```

### 3. Train the hospital agent
```bash
python training/train_hospital.py
```

### 4. View learned policy (Q-table)
```bash
python view_qtable.py
```

### 5. Interactive demo
```bash
python demo_hospital.py
```

---

## Research Contribution

Most RL systems are designed for a single domain. UMORDA demonstrates that:

> A single Multi-Objective Reinforcement Learning framework, combined with an LLM natural language interface (Groq API + Llama 3), can operate across five structurally different real-world domains while remaining fully explainable and accessible through natural language — an achievement not previously shown at the undergraduate research level.

This work is grounded in the theoretical proof by Roijers et al. that linear reward scalarisation with Q-learning produces Pareto-optimal policies for a given weight vector.

---

## Team

| Name | Student ID | Role |
|---|---|---|

| Sourav Roy | 2121856042 |

| Shouvik Gosh Ador | 2121986042 |

| Shuaib Mahmud Niloy | 2022127642 | 


**Faculty Advisor:** Dr. Mohammad Abdul Qayum, Assistant Professor
**Department:** ECE, North South University

---

## License

MIT License — see `LICENSE` for details.
