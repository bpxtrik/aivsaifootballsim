import os
import pickle
import random
import numpy as np
import torch
from agent_utils import get_team_actions
from agents.coach_agent import CoachAgent
from agents.player_agent import PlayerAgent
from config import ACTION_SPACE, CONFIG, NUM_ACTIONS
from data.data_loader import load_top15_rankings, load_tournament_players
from dqn import DQNAgent
from environment.game import FootballGame

CHECKPOINT_DIR = "checkpoints3"


def get_goalkeeper_action(env, player_agent, team_id):
    """
    Get goalkeeper action - same logic as in agent_utils.py
    """
    goal_height = 200
    goal_area_depth = 80
    goal_center_y = env.height // 2
    
    if team_id == 1:  # left goal
        x_min, x_max = 5, goal_area_depth
    else:  # right goal
        x_min, x_max = env.width - goal_area_depth, env.width - 5
    
    y_min, y_max = goal_center_y - goal_height // 2, goal_center_y + goal_height // 2
    
    current_x = player_agent.player.x
    current_y = player_agent.player.y
    
    ball_x = env.ball.x
    ball_y = env.ball.y
    
    # Calculate ideal target within goal area
    target_x = ball_x if x_min <= ball_x <= x_max else (x_min if ball_x < x_min else x_max)
    target_y = max(y_min, min(y_max, ball_y))
    
    # Calculate movement delta
    max_speed = player_agent.player.max_speed if hasattr(player_agent.player, 'max_speed') else 5
    
    dx = target_x - current_x
    dy = target_y - current_y
    
    dx = max(-max_speed, min(max_speed, dx))
    dy = max(-max_speed, min(max_speed, dy))
    
    next_x = max(x_min, min(x_max, current_x + dx))
    next_y = max(y_min, min(y_max, current_y + dy))
    
    final_dx = next_x - current_x
    final_dy = next_y - current_y
    
    return {"move": (final_dx, final_dy), "shoot": False, "pass_to": None}


def load_trained_agents():
    """
    Load the trained DQN agents and coaches from checkpoints.
    Returns agents and coaches that can be reused across matches.
    """
    # Create dummy environment to get state dimensions
    from data.data_loader import load_training_players
    dummy_stats = load_training_players("data/fifa_2023.csv")
    dummy_env = FootballGame(dummy_stats["team1"], dummy_stats["team2"], config=CONFIG)
    state_dim = len(dummy_env._get_player_state(dummy_env.team1.players[0]))
    
    # Load Team 1 agents
    team1_dqns = []
    for i in range(4):
        agent = DQNAgent(state_dim=state_dim, n_actions=NUM_ACTIONS)
        agent_path = os.path.join(CHECKPOINT_DIR, f"team1_agent_{i}.pt")
        if os.path.exists(agent_path):
            agent.load(agent_path)
            print(f"✓ Loaded team1_agent_{i}")
        else:
            print(f"⚠ Warning: {agent_path} not found, using untrained agent")
        team1_dqns.append(agent)
    
    # Load Team 2 agents
    team2_dqns = []
    for i in range(4):
        agent = DQNAgent(state_dim=state_dim, n_actions=NUM_ACTIONS)
        agent_path = os.path.join(CHECKPOINT_DIR, f"team2_agent_{i}.pt")
        if os.path.exists(agent_path):
            agent.load(agent_path)
            print(f"✓ Loaded team2_agent_{i}")
        else:
            print(f"⚠ Warning: {agent_path} not found, using untrained agent")
        team2_dqns.append(agent)
    
    # Load coaches
    coach1_path = os.path.join(CHECKPOINT_DIR, "team1_coach.pkl")
    coach2_path = os.path.join(CHECKPOINT_DIR, "team2_coach.pkl")
    
    if os.path.exists(coach1_path):
        with open(coach1_path, "rb") as f:
            team1_coach = pickle.load(f)
        print("✓ Loaded team1_coach")
    else:
        print("⚠ Warning: team1_coach not found, creating new coach")
        team1_coach = CoachAgent(dummy_env.team1)
    
    if os.path.exists(coach2_path):
        with open(coach2_path, "rb") as f:
            team2_coach = pickle.load(f)
        print("✓ Loaded team2_coach")
    else:
        print("⚠ Warning: team2_coach not found, creating new coach")
        team2_coach = CoachAgent(dummy_env.team2)
    
    return team1_dqns, team2_dqns, team1_coach, team2_coach


def simulate_match(nation1, nation2, team1_dqns, team2_dqns, team1_coach_template, team2_coach_template, 
                   csv_players="data/fifa_2023.csv", csv_rankings="data/fifa_ranking-2024-06-20.csv",
                   max_steps=400, render=False, tournament_mode=True):
    """
    Simulate a single match between two nations using trained agents.
    
    Args:
        tournament_mode: If True, uses greedy action selection (no exploration)
        csv_rankings: Path to FIFA rankings CSV (used for overtime tiebreaker)
    """
    print(f"\n{'='*60}")
    print(f"  {nation1} vs {nation2}")
    print(f"{'='*60}")
    
    # Load player stats for both nations
    teams = load_tournament_players(csv_players, nation1, nation2)
    team1_player_stats = teams["team1"]
    team2_player_stats = teams["team2"]
    
    # Safety net: Create default player stats dict if any player is None or missing
    def create_default_player_stats(position_index, team_id):
        """Create a default player stats dictionary with average stats"""
        return {
            "name": f"Default_Player_{position_index}",
            "PAC": 60,
            "SHO": 60,
            "PAS": 60,
            "DRI": 60,
            "DEF": 60,
            "PHY": 60
        }
    
    # Validate and fix team1 - ensure we have 4 player stat dicts
    if team1_player_stats is None or len(team1_player_stats) == 0:
        print(f"⚠ WARNING: No players found for {nation1}, creating default team")
        team1_player_stats = [create_default_player_stats(i, 1) for i in range(4)]
    else:
        # Ensure we have exactly 4 players
        while len(team1_player_stats) < 4:
            idx = len(team1_player_stats)
            print(f"⚠ WARNING: {nation1} has only {len(team1_player_stats)} players, adding default player {idx}")
            team1_player_stats.append(create_default_player_stats(idx, 1))
        
        # Check each player for None or missing stats
        for i in range(len(team1_player_stats)):
            if team1_player_stats[i] is None:
                print(f"⚠ WARNING: Player {i} is None in {nation1}, using default")
                team1_player_stats[i] = create_default_player_stats(i, 1)
            elif isinstance(team1_player_stats[i], dict):
                # Ensure all required stats exist
                required_stats = ["PAC", "SHO", "PAS", "DRI", "DEF", "PHY"]
                for stat in required_stats:
                    if stat not in team1_player_stats[i] or team1_player_stats[i][stat] is None:
                        print(f"⚠ WARNING: {nation1} Player {i} missing {stat}, setting to 60")
                        team1_player_stats[i][stat] = 60
    
    # Validate and fix team2
    if team2_player_stats is None or len(team2_player_stats) == 0:
        print(f"⚠ WARNING: No players found for {nation2}, creating default team")
        team2_player_stats = [create_default_player_stats(i, 2) for i in range(4)]
    else:
        # Ensure we have exactly 4 players
        while len(team2_player_stats) < 4:
            idx = len(team2_player_stats)
            print(f"⚠ WARNING: {nation2} has only {len(team2_player_stats)} players, adding default player {idx}")
            team2_player_stats.append(create_default_player_stats(idx, 2))
        
        # Check each player for None or missing stats
        for i in range(len(team2_player_stats)):
            if team2_player_stats[i] is None:
                print(f"⚠ WARNING: Player {i} is None in {nation2}, using default")
                team2_player_stats[i] = create_default_player_stats(i, 2)
            elif isinstance(team2_player_stats[i], dict):
                # Ensure all required stats exist
                required_stats = ["PAC", "SHO", "PAS", "DRI", "DEF", "PHY"]
                for stat in required_stats:
                    if stat not in team2_player_stats[i] or team2_player_stats[i][stat] is None:
                        print(f"⚠ WARNING: {nation2} Player {i} missing {stat}, setting to 60")
                        team2_player_stats[i][stat] = 60
    
    # Verify player stats are loaded correctly
    print(f"Team 1 ({nation1}): {len(team1_player_stats)} players")
    for i, p in enumerate(team1_player_stats):
        print(f"  Player {i}: {p.get('name', 'Unknown')}, DRI={p.get('DRI')}, PAS={p.get('PAS')}")
    
    print(f"Team 2 ({nation2}): {len(team2_player_stats)} players")
    for i, p in enumerate(team2_player_stats):
        print(f"  Player {i}: {p.get('name', 'Unknown')}, DRI={p.get('DRI')}, PAS={p.get('PAS')}")
    
    # Create game environment with these players and country names
    env = FootballGame(team1_player_stats, team2_player_stats, config=CONFIG, 
                       team1_name=nation1, team2_name=nation2)
    
    # Create PlayerAgent wrappers for team 1
    team1_players = [PlayerAgent(p, role="goalkeeper" if i == 0 else "field") 
                     for i, p in enumerate(env.team1.players)]
    
    # Create PlayerAgent wrappers for team 2
    team2_players = [PlayerAgent(p, role="goalkeeper" if i == 0 else "field") 
                     for i, p in enumerate(env.team2.players)]
    
    # Assign DQN agents (set eval mode for tournament)
    for i, player_agent in enumerate(team1_players):
        player_agent.player_agent = team1_dqns[i]
    
    for i, player_agent in enumerate(team2_players):
        player_agent.player_agent = team2_dqns[i]
    
    # Create coaches for this match (using the template team structures)
    team1_coach = CoachAgent(env.team1)
    team2_coach = CoachAgent(env.team2)
    
    # Run the match
    done = False
    total_r1 = total_r2 = 0.0
    obs = env.reset()
    
    # Random ball possession
    if random.random() < 0.5:
        random_player = random.choice(env.team2.players)
        random_player.has_ball = True
        env.ball.x = random_player.x
        env.ball.y = random_player.y
        for p in env.team1.players:
            p.has_ball = False
    
    step_count = 0
    overtime = False
    max_overtime_steps = 600  # Maximum total steps including overtime
    
    while not done and (step_count < max_steps or (overtime and step_count < max_overtime_steps)):
        # Get current states
        current_states_t1 = [env._get_player_state(p) for p in env.team1.players]
        current_states_t2 = [env._get_player_state(p) for p in env.team2.players]
        
        # Manually select actions with eval_mode for tournament
        team1_actions = {}
        team2_actions = {}
        
        # Team 1 actions
        for i, player_agent in enumerate(team1_players):
            if i == 0:  # Goalkeeper - handled by coach logic
                team1_actions[i] = get_goalkeeper_action(env, player_agent, team_id=1)
            else:  # Field players
                if hasattr(player_agent, 'player_agent') and player_agent.player_agent is not None:
                    # Use eval_mode for tournament (greedy action selection)
                    action_idx = player_agent.player_agent.select_action(current_states_t1[i], eval_mode=tournament_mode)
                    player_agent.last_action_idx = action_idx
                    action_dict = player_agent.act(env, action_int=action_idx)
                    team1_actions[i] = action_dict
                else:
                    team1_actions[i] = {"move": (0, 0), "shoot": False, "pass_to": None}
        
        # Team 2 actions
        for i, player_agent in enumerate(team2_players):
            if i == 0:  # Goalkeeper
                team2_actions[i] = get_goalkeeper_action(env, player_agent, team_id=2)
            else:  # Field players
                if hasattr(player_agent, 'player_agent') and player_agent.player_agent is not None:
                    action_idx = player_agent.player_agent.select_action(current_states_t2[i], eval_mode=tournament_mode)
                    player_agent.last_action_idx = action_idx
                    action_dict = player_agent.act(env, action_int=action_idx)
                    team2_actions[i] = action_dict
                else:
                    team2_actions[i] = {"move": (0, 0), "shoot": False, "pass_to": None}
        
        # Step environment
        next_obs, (rew1, rew2), game_done = env.step(team1_actions, team2_actions)
        
        total_r1 += np.mean(rew1)
        total_r2 += np.mean(rew2)
        step_count += 1
        
        # Check if we hit max_steps with 0-0 score
        if step_count == max_steps and env.score_left == 0 and env.score_right == 0:
            print(f"⚽ 0-0 after {max_steps} steps! Entering sudden death - next goal wins!")
            overtime = True
        
        # Check if overtime has gone too long (1200 steps total)
        if overtime and step_count >= max_overtime_steps and env.score_left == 0 and env.score_right == 0:
            print(f"⏱️ Match reached {max_overtime_steps} steps with no goal! Using FIFA rankings to decide winner...")
            done = True
            break
        
        # In overtime, end on first goal
        if overtime and (env.score_left > 0 or env.score_right > 0):
            print(f"⚽ GOAL! Match ends in overtime at step {step_count}")
            done = True
        
        # Check if episode should end early (large score difference)
        if abs(env.score_left - env.score_right) >= 5:
            done = True
        
        if render:
            env.render()
    
    # Determine winner by score
    score1 = env.score_left
    score2 = env.score_right
    
    if overtime:
        print(f"\n⚽ OVERTIME MATCH! Ended after {step_count} steps")
    
    print(f"\nFinal Score: {nation1} {score1} - {score2} {nation2}")
    print(f"Total Steps: {step_count}")
    
    # Winner is determined ONLY by score (no reward comparison)
    if score1 > score2:
        winner = nation1
    elif score2 > score1:
        winner = nation2
    else:
        # If still 0-0 after 1200 steps, use FIFA rankings
        print(f"⚠ Match still 0-0 after {step_count} steps! Deciding by FIFA rankings...")
        
        # Load rankings to determine higher ranked team
        try:
            rankings = load_top15_rankings(csv_rankings)
            rankings_dict = {r["country_full"]: r["rank"] for r in rankings}
            
            rank1 = rankings_dict.get(nation1, 999)  # Default to 999 if not in top 15
            rank2 = rankings_dict.get(nation2, 999)
            
            if rank1 < rank2:  # Lower rank number = better team
                winner = nation1
                print(f"📊 {nation1} (Rank {rank1}) beats {nation2} (Rank {rank2}) by ranking")
            elif rank2 < rank1:
                winner = nation2
                print(f"📊 {nation2} (Rank {rank2}) beats {nation1} (Rank {rank1}) by ranking")
            else:
                # Both unranked or same rank - random winner
                winner = nation1
                print(f"📊 Both teams have same rank, {nation1} advances by default")
        except Exception as e:
            print(f"⚠ Error loading rankings: {e}")
            print(f"📊 {nation1} advances by default")
            winner = nation1
    
    print(f"🏆 Winner: {winner}")
    
    return winner, score1, score2


def run_ai_tournament(csv_rankings="data/fifa_ranking-2024-06-20.csv", 
                     csv_players="data/fifa_2023.csv",
                     extra_country="Serbia",
                     render_finals=False):
    """
    Run a 16-team knockout tournament using trained AI agents.
    """
    print("\n" + "="*60)
    print("  AI FOOTBALL TOURNAMENT")
    print("="*60)
    
    # Load trained agents once (reuse across all matches)
    print("\nLoading trained agents...")
    team1_dqns, team2_dqns, team1_coach, team2_coach = load_trained_agents()
    
    # Get top 15 countries + 1 extra
    print("\nSelecting teams...")
    top15 = load_top15_rankings(csv_rankings)
    countries = [c["country_full"] for c in top15] + [extra_country]
    random.shuffle(countries)
    
    print(f"\nTournament participants:")
    for i, country in enumerate(countries, 1):
        print(f"  {i:2d}. {country}")
    
    # Create initial bracket (Round of 16)
    bracket = [(countries[i], countries[i+1]) for i in range(0, 16, 2)]
    round_num = 1
    round_names = {1: "Round of 16", 2: "Quarter-finals", 3: "Semi-finals", 4: "Final"}
    
    # Track results
    all_results = []
    
    while len(bracket) > 0:
        round_name = round_names.get(round_num, f"Round {round_num}")
        print(f"\n{'#'*60}")
        print(f"  {round_name}")
        print(f"{'#'*60}")
        
        next_round = []
        
        for match_num, (nation1, nation2) in enumerate(bracket, 1):
            print(f"\nMatch {match_num}/{len(bracket)}")
            
            # Render only the final match if requested
            should_render = render_finals and round_num == 4
            
            # Simulate match
            winner, score1, score2 = simulate_match(
                nation1, nation2, 
                team1_dqns, team2_dqns,
                team1_coach, team2_coach,
                csv_players=csv_players,
                csv_rankings=csv_rankings,  # Pass rankings for tiebreaker
                max_steps=400,
                render=should_render,
                tournament_mode=True  # Use greedy (no exploration) for tournament
            )
            
            # Record result
            all_results.append({
                'round': round_name,
                'match': f"{nation1} vs {nation2}",
                'score': f"{score1}-{score2}",
                'winner': winner
            })
            
            next_round.append(winner)
        
        # Prepare next round bracket
        if len(next_round) > 1:
            bracket = [(next_round[i], next_round[i+1]) for i in range(0, len(next_round), 2)]
            round_num += 1
        else:
            bracket = []
    
    # Print tournament summary
    print("\n" + "="*60)
    print("  TOURNAMENT SUMMARY")
    print("="*60)
    
    current_round = ""
    for result in all_results:
        if result['round'] != current_round:
            current_round = result['round']
            print(f"\n{current_round}:")
        print(f"  {result['match']:40s} {result['score']:8s} → {result['winner']}")
    
    champion = next_round[0]
    print("\n" + "🏆"*30)
    print(f"  CHAMPION: {champion}")
    print("🏆"*30 + "\n")
    
    return champion


if __name__ == "__main__":
    # Run the tournament
    champion = run_ai_tournament(
        csv_rankings="data/fifa_ranking-2024-06-20.csv",
        csv_players="data/fifa_2023.csv",
        extra_country="Serbia",
        render_finals=False  # Set to True to watch the final match
    )