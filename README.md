# Werewolf AI Environment

This repository implements a simplified Werewolf (Mafia) environment and a set of agents for research and demo purposes.  All code lives inside the `werewolf_0613` folder.

## Game Environment

The environment is defined in [`werewolf_env/werewolf_env.py`](werewolf_0613/werewolf_env/werewolf_env.py).  It is built on top of **PettingZoo** and models a minimal version of the game with three roles:

- `WOLF`
- `VILLAGER`
- `SEER`

The game proceeds through three phases (`talk`, `vote`, `night`).  The environment exposes a parallel PettingZoo interface so multiple agents can act simultaneously.  Example episodes can be found in [`demos`](werewolf_0613/demos/).

## Agents

Agents are implemented in [`agents`](werewolf_0613/agents/).  `BaseAgent` provides a common interface and loads different strategy classes depending on the current phase.  `BeliefAgent` extends it with Bayesian belief updates over who might be a wolf or seer.

## Training

`train/train_villagers_vs_beliefwolves.py` contains a script that trains villagers (including the seer) with **PPO** from Stable-Baselines3 against fixed `BeliefAgent` wolves.  Run it with:

```bash
python werewolf_0613/train/train_villagers_vs_beliefwolves.py
```

Trained weights are saved to `ppo_villagers_vs_beliefwolves.zip`.

## Web App

A simple web interface is provided in [`webapp`](werewolf_0613/webapp/).  It uses Flask and Socket.IO to display the game state and belief updates.  After installing the requirements, start the server with:

```bash
cd werewolf_0613/webapp
pip install -r requirements.txt
python app.py
```

The app will start on `http://localhost:5000/`.

## Python & Requirements

The project requires **Python 3.10+**.  The main packages used by the web app are listed in `webapp/requirements.txt`:

```
flask==3.0.0
flask-socketio==5.3.6
numpy==1.24.3
pettingzoo==1.24.1
gymnasium==0.29.1
python-dotenv==1.0.0
```

Stable-Baselines3 is needed for running the training script.
