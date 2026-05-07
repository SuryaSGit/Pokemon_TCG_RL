# Pokemon TCG RL

Reinforcement learning agents trained to play a custom Pokemon Trading Card Game environment. Two decks face off — a fully playable browser UI lets you challenge the trained bots.

---

## Playable Demo — v3

**v3** is the stable demo version. Lycanroc half-deck vs Alolan Raichu half-deck (Sun & Moon Trainer Kit).

```bash
cd v3/v3GUI
pip install flask
python app.py
# Open http://localhost:5000
```

Two trained champions are included:
- `ppo_lycanroc_champion.npy` — PPO agent (Lycanroc deck)
- `dqn_raichu_champion.npy` — DQN agent (Raichu deck)

---

## In Development — v4

**v4** is the current work-in-progress. Full 60-card decks: **Mega Starmie ex** vs **Mega Lucario ex**, with evolution, abilities, tools, stadiums, status effects, and special energy. The RL agent plays Lucario.

```bash
cd v4

# Play against the bot in the terminal (you are Starmie)
python train.py play

# Train the Lucario RL agent
python train.py train --episodes 5000

# Launch the web UI (incomplete)
cd gui && python app.py
```

---

## Project Layout

```
Pokemon_TCG_RL/
├── v3/                    ← Stable demo (Lycanroc vs Raichu)
│   ├── ptcg_env.py        ← Game engine
│   ├── rl_agents.py       ← PPO + DQN agents
│   ├── play_and_train.py  ← Training CLI
│   └── v3GUI/             ← Flask web UI
│       ├── app.py
│       └── templates/
│
├── v4/                    ← In development (Starmie vs Lucario, full 60-card)
│   ├── ptcg_env.py        ← Game engine
│   ├── rl_agents.py       ← PPO + DQN agents
│   ├── train.py           ← Training & play CLI
│   ├── gui/               ← Flask web UI (WIP)
│   ├── assets/images/     ← Card images for the GUI
│   ├── docs/              ← Card descriptions and decklists
│   └── outputs/           ← Saved weights & leaderboard
│
├── v2/                    ← Archive
├── v1/                    ← Archive
└── presentation/          ← Project slides
```

---

## Agents

All agents use **NumPy-only** neural networks — no PyTorch or TensorFlow.

| Version | Agent | Algorithm |
|---------|-------|-----------|
| v3 | PPO (Lycanroc) | Proximal Policy Optimization |
| v3 | DQN (Raichu) | Double DQN with replay buffer |
| v4 | PPO + DQN (Lucario) | Trained via behavioral cloning then RL vs heuristic |

---

## Requirements

```
flask
numpy
```
