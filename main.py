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

def train_main():
    stats = load_tournament_players("data/fifa_2023.csv", "France", "Germany")
    env = FootballGame(stats["team1"], stats["team2"], config=CONFIG)

    # === Create agents per player ===
    team1_agents = []
    team2_agents = []

    dummy_state = env._get_player_state(env.team1.players[0])
    STATE_DIM = len(dummy_state)

    for _ in range(len(env.team1.players)):
        team1_agents.append(DQNAgent(state_dim=STATE_DIM, n_actions=NUM_ACTIONS))
    for _ in range(len(env.team2.players)):
        team2_agents.append(DQNAgent(state_dim=STATE_DIM, n_actions=NUM_ACTIONS))

    num_episodes = 500
    max_steps = 500

    for ep in range(num_episodes):
        env.reset()
        done = False
        total_reward_team1 = 0.0
        total_reward_team2 = 0.0
        step = 0

        while not done and step < max_steps:
            step += 1

            # === Choose actions for each player ===
            actions_team1 = {}
            actions_team2 = {}

            for i, player in enumerate(env.team1.players):
                state = env._get_player_state(player)
                action_idx = team1_agents[i].select_action(state)
                actions_team1[i] = ACTION_SPACE[action_idx]

            for i, player in enumerate(env.team2.players):
                state = env._get_player_state(player)
                action_idx = team2_agents[i].select_action(state)
                actions_team2[i] = ACTION_SPACE[action_idx]

            # === Step environment ===
            next_state, reward_tuple, done = env.step(actions_team1, actions_team2)

            # === Update each player’s agent ===
            rewards_team1, rewards_team2 = reward_tuple

            for i, player in enumerate(env.team1.players):
                state = env._get_player_state(player)
                next_s = env._get_player_state(player)
                r = float(rewards_team1[i])
                team1_agents[i].store_transition(state, action_idx, r, next_s, done)
                team1_agents[i].train_step()
                total_reward_team1 += r

            for i, player in enumerate(env.team2.players):
                state = env._get_player_state(player)
                next_s = env._get_player_state(player)
                r = float(rewards_team2[i])
                team2_agents[i].store_transition(state, action_idx, r, next_s, done)
                team2_agents[i].train_step()
                total_reward_team2 += r

            env.render()
            pygame.time.delay(20)

        print(f"Episode {ep} | Team1 total reward: {total_reward_team1:.2f} | Team2: {total_reward_team2:.2f}")



if __name__ == "__main__":

    # main()
    train_main()
