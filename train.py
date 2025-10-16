# train.py (example)
import numpy as np
from dqn import DQNAgent
from environment.game import FootballGame
from data.data_loader import load_training_players
from config import ACTION_SPACE, CONFIG
from env_helpers import map_int_action_to_dict  # implement mapping using ACTION_SPACE

# hyperparams
NUM_EPISODES = 2000
MAX_STEPS = 500  # per episode
BATCH_SIZE = 64
STATE_DIM =  44# set to state.shape[1], inspect after env.reset()
N_ACTIONS = len(ACTION_SPACE)  # your discrete action count

# create env
players = load_training_players("data/fifa_2023.csv")
env = FootballGame(players["team1"], players["team2"], CONFIG)
state = env.reset()
# state shape: (num_players_total, features)
num_players_total = state.shape[0]
state_dim = state.shape[1]

# create one agent per player
agents = [DQNAgent(state_dim, N_ACTIONS, batch_size=BATCH_SIZE) for _ in range(num_players_total)]

for ep in range(NUM_EPISODES):
    state = env.reset()
    ep_reward = np.zeros(num_players_total)
    for t in range(MAX_STEPS):
        # select actions for all players
        actions_int_team1 = {}
        actions_int_team2 = {}
        # team1 indices 0..3, team2 4..7 (adjust per your layout)
        for i in range(num_players_total):
            s_i = state[i]
            a_i = agents[i].select_action(s_i)
            # map global index -> team index (0..3) for environment mapping
            if i < len(env.team1.players):
                actions_int_team1[i] = a_i
            else:
                actions_int_team2[i - len(env.team1.players)] = a_i

        # convert integer actions to dicts for env.step
        actions_team1 = {i: map_int_action_to_dict(a) for i, a in actions_int_team1.items()}
        actions_team2 = {i: map_int_action_to_dict(a) for i, a in actions_int_team2.items()}

        next_state, (rewards1, rewards2), done = env.step(actions_team1, actions_team2)
        # env.step returns (state, (rewards_team1, rewards_team2), done)

        # combine rewards into single array aligned to agents order
        rewards = np.concatenate([rewards1, rewards2])

        # store for each agent
        for i in range(num_players_total):
            s = state[i]
            a = actions_int_team1.get(i, None) if i < len(env.team1.players) else actions_int_team2.get(i - len(env.team1.players))
            # ensure action int exists
            if a is None:
                continue
            s_next = next_state[i] if next_state is not None else None
            r = rewards[i]
            d = done
            agents[i].store_transition(s, a, r, s_next, d)

        # update/train each agent (or a subset each step)
        for a in agents:
            a.train_step()

        state = next_state
        ep_reward += rewards
        if done:
            break

    # optionally print / save
    print(f"Episode {ep} total rewards per agent: {ep_reward}")
    if ep % 50 == 0:
        for idx, a in enumerate(agents):
            a.save(f"models/agent_{idx}_ep{ep}.pt")
