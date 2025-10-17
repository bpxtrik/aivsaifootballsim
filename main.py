from dqn import DQNAgent
from environment.game import FootballGame
from config import ACTION_SPACE, CONFIG, NUM_ACTIONS
from data.data_loader import load_tournament_players, load_training_players
import pygame
import numpy as np

def main():
    # Load in data
    stats = load_training_players("data/fifa_2023.csv")

    # Create environment
    env = FootballGame(stats["team1"], stats["team2"], config=CONFIG)
    state = env.reset()

    running = True
    while running:
        for event in pygame.event.get():
            if (event.type) == pygame.QUIT:
                running = False

        # Simple keyboard control for blue ATT (index 3) and red ATT (index 3)
        keys = pygame.key.get_pressed()
        ax_b = 0.0
        ay_b = 0.0
        shoot_b = False
        if keys[pygame.K_w]:
            ay_b -= 1.0
        if keys[pygame.K_s]:
            ay_b += 1.0
        if keys[pygame.K_a]:
            ax_b -= 1.0
        if keys[pygame.K_d]:
            ax_b += 1.0
        if keys[pygame.K_SPACE]:
            shoot_b = True

        ax_r = 0.0
        ay_r = 0.0
        shoot_r = False
        if keys[pygame.K_UP]:
            ay_r -= 1.0
        if keys[pygame.K_DOWN]:
            ay_r += 1.0
        if keys[pygame.K_LEFT]:
            ax_r -= 1.0
        if keys[pygame.K_RIGHT]:
            ax_r += 1.0
        if keys[pygame.K_RCTRL] or keys[pygame.K_RSHIFT]:
            shoot_r = True

        actions_team1 = {3: {"move": (ax_b, ay_b), "shoot": shoot_b}}
        actions_team2 = {3: {"move": (ax_r, ay_r), "shoot": shoot_r}}

        state, reward, done = env.step(actions_team1, actions_team2)

        env.render()

    # Create trainer
    # trainer = Trainer(env, config=CONFIG)

    # Start training
    # trainer.train(num_episodes=CONFIG["episodes"])

import os
from dqn import DQNAgent
from config import NUM_ACTIONS
from environment.game import FootballGame
from data.data_loader import load_training_players
from config import CONFIG

def load_all_agents(path):
    """
    Loads 8 trained agents from checkpoints folder:
    team1_agent_0.pt ... team1_agent_3.pt
    team2_agent_0.pt ... team2_agent_3.pt
    Returns: (team1_agents, team2_agents)
    """
    # Build env once to get state_dim
    stats = load_training_players("data/fifa_2023.csv")
    env = FootballGame(stats["team1"], stats["team2"], config=CONFIG)
    state_dim = len(env._get_player_state(env.team1.players[0]))

    checkpoint_dir = path
    team1_agents, team2_agents = [], []

    for i in range(4):
        agent1_path = os.path.join(checkpoint_dir, f"team1_agent_{i}.pt")
        agent2_path = os.path.join(checkpoint_dir, f"team2_agent_{i}.pt")

        agent1 = DQNAgent(state_dim, NUM_ACTIONS)
        agent2 = DQNAgent(state_dim, NUM_ACTIONS)

        agent1.load(agent1_path)
        agent2.load(agent2_path)

        team1_agents.append(agent1)
        team2_agents.append(agent2)

    print("✅ All agents loaded successfully.")
    return team1_agents, team2_agents

import pygame
from environment.game import FootballGame
from data.data_loader import load_training_players
from config import CONFIG


from train import train_8_agents
from evaulation import evaluate_8_agents
if __name__ == "__main__":
    train_8_agents()
    # train_8_agents_parallel()
    # (team1_agents, team2_agents) = load_all_agents("checkpoints")
    # evaluate_8_agents(team1_agents, team2_agents, render_last_n=10)