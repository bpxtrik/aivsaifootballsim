import pygame
from config import ACTION_SPACE, CONFIG
from data.data_loader import load_tournament_players, load_training_players
from environment.game import FootballGame


def run_episode_eval(stats, team1_agents, team2_agents, render=False):
    env = FootballGame(stats["team1"], stats["team2"], config=CONFIG)
    env.reset()
    done = False
    step = 0
    max_steps = 1600
    total_reward_team1 = 0.0
    total_reward_team2 = 0.0

    while not done and step < max_steps:
        step += 1
        actions_team1 = {}
        actions_team2 = {}

        # Team1 actions
        for i, player in enumerate(env.team1.players):
            state = env._get_player_state(player)
            action_idx = team1_agents[i].select_action(state, eval_mode=True)
            actions_team1[i] = ACTION_SPACE[action_idx]
            player.state = state
            player.action_idx = action_idx

        # Team2 actions
        for i, player in enumerate(env.team2.players):
            state = env._get_player_state(player)
            action_idx = team2_agents[i].select_action(state, eval_mode=True)
            actions_team2[i] = ACTION_SPACE[action_idx]
            player.state = state
            player.action_idx = action_idx

        _, reward_tuple, done = env.step(actions_team1, actions_team2)
        rewards_team1, rewards_team2 = reward_tuple

        total_reward_team1 += sum(rewards_team1)
        total_reward_team2 += sum(rewards_team2)

        if render:
            env.render()
            pygame.time.delay(20)

    return total_reward_team1, total_reward_team2, env.score_left, env.score_right


def evaluate_8_agents(team1_agents, team2_agents, num_matches=10, render_last_n=0):
    # stats = load_tournament_players("data/fifa_2023.csv", "France", "Germany")
    stats = load_training_players("data/fifa_2023.csv")

    results = {
        "team1_wins": 0,
        "team2_wins": 0,
        "draws": 0,
        "scores": []
    }

    for ep in range(num_matches):
        render = ep >= num_matches - render_last_n
        r1, r2, score1, score2 = run_episode_eval(stats, team1_agents, team2_agents, render=render)

        results["scores"].append((score1, score2))
        if score1 > score2:
            results["team1_wins"] += 1
        elif score2 > score1:
            results["team2_wins"] += 1
        else:
            results["draws"] += 1

        print(f"Match {ep} | Team1 reward: {r1:.2f} | Team2 reward: {r2:.2f} | Score: {score1}-{score2}")

    print("\n=== Evaluation Summary ===")
    print(f"Team1 Wins: {results['team1_wins']}")
    print(f"Team2 Wins: {results['team2_wins']}")
    print(f"Draws: {results['draws']}")
    print(f"Scores: {results['scores']}")
    return results
