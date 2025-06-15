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

## Human vs AI Game

The web interface also supports playing against the AI.  When starting a game
you can pass the option `game_mode="human"` so one player is controlled by a
person while all others remain AI agents.  The specific seat the human controls
is chosen via `human_player_id` (player indices start at 0).  These parameters
are part of the payload that the client sends with the `start_game` Socket.IO
event.

To try it out:

1. Launch the server and open the site as shown above.
2. Click **人机对战** on the landing page to open the setup dialog.
3. Configure the number of players and select which position you will play via
   the **人类玩家位置** dropdown.
4. Press **创建并开始** to begin.  Whenever it is the human player's turn the
   page will display a panel to choose night actions, make a statement or vote.

The rest of the flow is identical to AI vs AI games but the interface will wait
for your choices whenever the human controlled slot must act.

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
