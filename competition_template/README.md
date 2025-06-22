# Werewolf AI Competition Template

This folder contains a minimal environment and baseline agent code for the "Biweekly Cup" Werewolf AI contest. Contestants should implement their own agents in `agents_user/` and verify they run with the provided evaluation script.

## Installation

1. Use Python **3.10+**.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Creating a Custom Agent

A custom agent must expose an `act(env)` method that receives a `WerewolfEnv` instance and returns a legal action for the current stage. You may reference `agents_user/random_agent.py` as a starting point:

```python
from competition_template.werewolf_env.werewolf_env import Role, WerewolfEnv

class MyAgent:
    def __init__(self, agent_id: int, num_agents: int, role: Role):
        self.role = role

    def act(self, env: WerewolfEnv):
        if env.stage == "talk":
            # always claim good
            return env.talk_space.sample()
        if env.stage == "vote":
            # vote for player 0 by default
            return 0
        if env.stage == "night" and self.role == Role.SEER:
            return 0  # check player 0
        return env.noop_space.sample()
```

Place your agent file in `agents_user/` and ensure the class is importable. The evaluation script will instantiate your class by name.

## Running the Example Match

A simple script to pit three random agents against each other:

```bash
python -m competition_template.demo
```

The script runs one episode and prints the event log.
