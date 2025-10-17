import os
import pygame
from agent_utils import get_team_actions
from agents.coach_agent import CoachAgent
from agents.player_agent import PlayerAgent
from config import ACTION_SPACE, CONFIG, NUM_ACTIONS
from data.data_loader import load_training_players
from dqn import DQNAgent
import numpy as np
from environment.game import FootballGame


def run_episode(stats, team1_coach, team1_players, team2_coach, team2_players, max_steps=400, render=False):
    env = FootballGame(stats["team1"], stats["team2"], config=CONFIG)

    # **FIX: Update player references to the new environment's players**
    for i, player_agent in enumerate(team1_players):
        player_agent.player = env.team1.players[i]
    for i, player_agent in enumerate(team2_players):
        player_agent.player = env.team2.players[i]

    done = False
    total_r1 = total_r2 = 0.0
    obs = env.reset()

    import random
    if random.random() < 0.5:
        # Give ball to random Team 2 player
        random_player = random.choice(env.team2.players)
        random_player.has_ball = True
        env.ball.x = random_player.x
        env.ball.y = random_player.y
        # Remove ball from Team 1 if they had it
        for p in env.team1.players:
            p.has_ball = False

    last_states_t1 = [None] * len(env.team1.players)
    last_states_t2 = [None] * len(env.team2.players)
    last_actions_t1 = [None] * len(env.team1.players)
    last_actions_t2 = [None] * len(env.team2.players)

    step_count = 0

    while not done and step_count < max_steps:
        # --- Get current states for each player ---
        current_states_t1 = [env._get_player_state(p) for p in env.team1.players]
        current_states_t2 = [env._get_player_state(p) for p in env.team2.players]

        # --- Get team actions using DQN ---
        team1_actions = get_team_actions(team1_coach, team1_players, env, team_id=1, 
                                        states=current_states_t1)
        team2_actions = get_team_actions(team2_coach, team2_players, env, team_id=2,
                                        states=current_states_t2)

        # --- Store the actual actions chosen ---
        last_actions_t1 = list(team1_actions.values())
        last_actions_t2 = list(team2_actions.values())

        # --- Step environment ---
        next_obs, (rew1, rew2), game_done = env.step(team1_actions, team2_actions)

        # --- Get next states ---
        next_states_t1 = [env._get_player_state(p) for p in env.team1.players]
        next_states_t2 = [env._get_player_state(p) for p in env.team2.players]

        # --- Store transitions and train for Team 1 ---
        for i, player_agent in enumerate(team1_players):
            if hasattr(player_agent, 'player_agent') and player_agent.player_agent is not None:
                s = current_states_t1[i]
                s_next = next_states_t1[i]
                r = rew1[i]
                # Get the actual action index chosen
                action_idx = player_agent.last_action_idx if hasattr(player_agent, 'last_action_idx') else 0
                
                player_agent.player_agent.store_transition(s, action_idx, r, s_next, done or game_done)
                player_agent.player_agent.train_step()

        # --- Store transitions and train for Team 2 ---
        for i, player_agent in enumerate(team2_players):
            if hasattr(player_agent, 'player_agent') and player_agent.player_agent is not None:
                s = current_states_t2[i]
                s_next = next_states_t2[i]
                r = rew2[i]
                action_idx = player_agent.last_action_idx if hasattr(player_agent, 'last_action_idx') else 0
                
                player_agent.player_agent.store_transition(s, action_idx, r, s_next, done or game_done)
                player_agent.player_agent.train_step()

        total_r1 += np.mean(rew1)
        total_r2 += np.mean(rew2)

        step_count += 1

        # Check if episode should end early (e.g., large score difference)
        if abs(env.score_left - env.score_right) >= 5:
            done = True

        if render:
            env.render()

    print(f"Episode ended after {step_count} steps. Score: {env.score_left}-{env.score_right}")
    return total_r1, total_r2, step_count


import os
import pickle

def train_8_agents():
    # Initialize dummy environment for dimensions
    dummy_stats = load_training_players("data/fifa_2023.csv")
    dummy_env = FootballGame(dummy_stats["team1"], dummy_stats["team2"], config=CONFIG)

    # Initialize DQN agents for both teams
    team1_dqns = [DQNAgent(state_dim=len(dummy_env._get_player_state(p)), n_actions=NUM_ACTIONS)
                  for p in dummy_env.team1.players]
    team2_dqns = [DQNAgent(state_dim=len(dummy_env._get_player_state(p)), n_actions=NUM_ACTIONS)
                  for p in dummy_env.team2.players]

    # Wrap DQNs in PlayerAgent objects
    team1_players = [PlayerAgent(p, role="goalkeeper" if i == 0 else "field") 
                     for i, p in enumerate(dummy_env.team1.players)]
    team2_players = [PlayerAgent(p, role="goalkeeper" if i == 0 else "field") 
                     for i, p in enumerate(dummy_env.team2.players)]

    # Assign DQN agent references
    for i, player_agent in enumerate(team1_players):
        player_agent.player_agent = team1_dqns[i]
    for i, player_agent in enumerate(team2_players):
        player_agent.player_agent = team2_dqns[i]

    # Create one coach per team
    team1_coach = CoachAgent(dummy_env.team1)
    team2_coach = CoachAgent(dummy_env.team2)

    num_episodes = 1000
    max_steps = 400
    render_last_n = 5

    for ep in range(num_episodes):
        render = ep >= num_episodes - render_last_n
        render = True
        stats = load_training_players("data/fifa_2023.csv")

        # Train one full game
        r1, r2, steps = run_episode(stats, team1_coach, team1_players, team2_coach, 
                                    team2_players, max_steps=max_steps, render=render)

        if ep % 10 == 0 or render:
            eps = team1_dqns[0].epsilon
            print(f"Episode {ep} | Steps: {steps} | Team1: {r1:.2f} | Team2: {r2:.2f} | Epsilon: {eps:.3f}")

    # Save all agents
    os.makedirs("checkpoints3", exist_ok=True)
    for i, agent in enumerate(team1_dqns):
        agent.save(f"checkpoints3/team1_agent_{i}.pt")
    for i, agent in enumerate(team2_dqns):
        agent.save(f"checkpoints3/team2_agent_{i}.pt")

    # Save coaches using pickle
    with open("checkpoints3/team1_coach.pkl", "wb") as f:
        pickle.dump(team1_coach, f)
    with open("checkpoints3/team2_coach.pkl", "wb") as f:
        pickle.dump(team2_coach, f)

    print("All agents and coaches saved in checkpoints3/")



if __name__ == "__main__":
    train_8_agents()