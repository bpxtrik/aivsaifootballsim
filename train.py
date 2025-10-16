import os
import pygame
from config import ACTION_SPACE, CONFIG, NUM_ACTIONS
from data.data_loader import load_training_players
from dqn import DQNAgent
from environment.game import FootballGame


def run_episode(stats, team1_agents, team2_agents, render=False):
    env = FootballGame(stats["team1"], stats["team2"], config=CONFIG)
    env.reset()
    done = False
    step = 0
    max_steps = 400
    total_reward_team1 = 0.0
    total_reward_team2 = 0.0

    while not done and step < max_steps:
        step += 1
        actions_team1 = {}
        actions_team2 = {}

        # Team1 actions
        for i, player in enumerate(env.team1.players):
            state = env._get_player_state(player)
            action_idx = team1_agents[i].select_action(state)
            actions_team1[i] = ACTION_SPACE[action_idx]
            player.state = state
            player.action_idx = action_idx

        # Team2 actions
        for i, player in enumerate(env.team2.players):
            state = env._get_player_state(player)
            action_idx = team2_agents[i].select_action(state)
            actions_team2[i] = ACTION_SPACE[action_idx]
            player.state = state
            player.action_idx = action_idx

        next_state, reward_tuple, done = env.step(actions_team1, actions_team2)
        rewards_team1, rewards_team2 = reward_tuple

        # Train agents immediately
        for i, player in enumerate(env.team1.players):
            next_s = env._get_player_state(player)
            r = float(rewards_team1[i])
            team1_agents[i].store_transition(player.state, player.action_idx, r, next_s, done)
            team1_agents[i].train_step()
            total_reward_team1 += r

        for i, player in enumerate(env.team2.players):
            next_s = env._get_player_state(player)
            r = float(rewards_team2[i])
            team2_agents[i].store_transition(player.state, player.action_idx, r, next_s, done)
            team2_agents[i].train_step()
            total_reward_team2 += r

        if render:
            env.render()
            pygame.time.delay(20)

    return total_reward_team1, total_reward_team2


def train_8_agents():
    # Initialize agents once using a dummy environment
    dummy_stats = load_training_players("data/fifa_2023.csv")
    dummy_env = FootballGame(dummy_stats["team1"], dummy_stats["team2"], config=CONFIG)

    team1_agents = [DQNAgent(state_dim=len(dummy_env._get_player_state(p)), n_actions=NUM_ACTIONS)
                    for p in dummy_env.team1.players]
    team2_agents = [DQNAgent(state_dim=len(dummy_env._get_player_state(p)), n_actions=NUM_ACTIONS)
                    for p in dummy_env.team2.players]

    num_episodes = 1000
    render_last_n = 5

    for ep in range(num_episodes):
        render = ep >= num_episodes - render_last_n
        # Load new random teams each episode
        stats = load_training_players("data/fifa_2023.csv")
        r1, r2 = run_episode(stats, team1_agents, team2_agents, render=True)

        if ep % 10 == 0 or render:
            print(f"Episode {ep} | Team1 reward: {r1:.2f} | Team2 reward: {r2:.2f}")

    # Save all agents
    os.makedirs("checkpoints", exist_ok=True)
    for i, agent in enumerate(team1_agents):
        agent.save(f"checkpoints/team1_agent_{i}.pt")
    for i, agent in enumerate(team2_agents):
        agent.save(f"checkpoints/team2_agent_{i}.pt")

    print("All agents saved in checkpoints/")


if __name__ == "__main__":
    train_8_agents()
