import os
import pickle
import numpy as np
import torch
from multiprocessing import Pool, Manager, cpu_count
from agent_utils import get_team_actions
from agents.coach_agent import CoachAgent
from agents.player_agent import PlayerAgent
from config import ACTION_SPACE, CONFIG, NUM_ACTIONS
from data.data_loader import load_training_players
from dqn import DQNAgent
from environment.game import FootballGame


def run_episode_worker(args):
    """
    Worker function to run a single episode.
    Returns experiences for all agents.
    Note: render is always False in workers to avoid multiple pygame windows.
    """
    ep_num, team1_state_dicts, team2_state_dicts, epsilon_value, max_steps, render = args
    
    # Force render to False in worker processes to avoid pygame window spam
    render = False
    
    # Load fresh stats
    stats = load_training_players("data/fifa_2023.csv")
    env = FootballGame(stats["team1"], stats["team2"], config=CONFIG)
    
    # Create fresh DQN agents
    team1_dqns = [DQNAgent(state_dim=len(env._get_player_state(p)), n_actions=NUM_ACTIONS)
                  for p in env.team1.players]
    team2_dqns = [DQNAgent(state_dim=len(env._get_player_state(p)), n_actions=NUM_ACTIONS)
                  for p in env.team2.players]
    
    # Load weights from main process
    for i, agent in enumerate(team1_dqns):
        agent.q_net.load_state_dict(team1_state_dicts[i])
        agent.target_net.load_state_dict(team1_state_dicts[i])
        agent.learn_steps = int(epsilon_value * agent.epsilon_decay)  # Sync epsilon
    
    for i, agent in enumerate(team2_dqns):
        agent.q_net.load_state_dict(team2_state_dicts[i])
        agent.target_net.load_state_dict(team2_state_dicts[i])
        agent.learn_steps = int(epsilon_value * agent.epsilon_decay)
    
    # Wrap in PlayerAgent objects
    team1_players = [PlayerAgent(p, role="goalkeeper" if i == 0 else "field") 
                     for i, p in enumerate(env.team1.players)]
    team2_players = [PlayerAgent(p, role="goalkeeper" if i == 0 else "field") 
                     for i, p in enumerate(env.team2.players)]
    
    for i, player_agent in enumerate(team1_players):
        player_agent.player_agent = team1_dqns[i]
    for i, player_agent in enumerate(team2_players):
        player_agent.player_agent = team2_dqns[i]
    
    # Create coaches
    team1_coach = CoachAgent(env.team1)
    team2_coach = CoachAgent(env.team2)
    
    # Run episode
    done = False
    total_r1 = total_r2 = 0.0
    obs = env.reset()
    
    # Random ball possession
    import random
    if random.random() < 0.5:
        random_player = random.choice(env.team2.players)
        random_player.has_ball = True
        env.ball.x = random_player.x
        env.ball.y = random_player.y
        for p in env.team1.players:
            p.has_ball = False
    
    # Collect experiences
    team1_experiences = [[] for _ in range(len(team1_players))]
    team2_experiences = [[] for _ in range(len(team2_players))]
    
    step_count = 0
    
    while not done and step_count < max_steps:
        # Get current states
        current_states_t1 = [env._get_player_state(p) for p in env.team1.players]
        current_states_t2 = [env._get_player_state(p) for p in env.team2.players]
        
        # Get team actions
        team1_actions = get_team_actions(team1_coach, team1_players, env, team_id=1, 
                                        states=current_states_t1)
        team2_actions = get_team_actions(team2_coach, team2_players, env, team_id=2,
                                        states=current_states_t2)
        
        # Step environment
        next_obs, (rew1, rew2), game_done = env.step(team1_actions, team2_actions)
        
        # Get next states
        next_states_t1 = [env._get_player_state(p) for p in env.team1.players]
        next_states_t2 = [env._get_player_state(p) for p in env.team2.players]
        
        # Collect experiences for Team 1
        for i, player_agent in enumerate(team1_players):
            if hasattr(player_agent, 'player_agent') and player_agent.player_agent is not None:
                action_idx = player_agent.last_action_idx if hasattr(player_agent, 'last_action_idx') else 0
                experience = {
                    'state': current_states_t1[i],
                    'action': action_idx,
                    'reward': rew1[i],
                    'next_state': next_states_t1[i],
                    'done': done or game_done
                }
                team1_experiences[i].append(experience)
        
        # Collect experiences for Team 2
        for i, player_agent in enumerate(team2_players):
            if hasattr(player_agent, 'player_agent') and player_agent.player_agent is not None:
                action_idx = player_agent.last_action_idx if hasattr(player_agent, 'last_action_idx') else 0
                experience = {
                    'state': current_states_t2[i],
                    'action': action_idx,
                    'reward': rew2[i],
                    'next_state': next_states_t2[i],
                    'done': done or game_done
                }
                team2_experiences[i].append(experience)
        
        total_r1 += np.mean(rew1)
        total_r2 += np.mean(rew2)
        step_count += 1
        
        if abs(env.score_left - env.score_right) >= 5:
            done = True
        
        if render:
            env.render()
    
    return {
        'ep_num': ep_num,
        'team1_experiences': team1_experiences,
        'team2_experiences': team2_experiences,
        'total_r1': total_r1,
        'total_r2': total_r2,
        'steps': step_count,
        'score': (env.score_left, env.score_right)
    }


def train_8_agents_parallel(num_workers=None):
    """
    Parallel training using your custom DQN.
    """
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)
    
    print(f"Starting parallel training with {num_workers} workers...")
    
    # Initialize dummy environment for dimensions
    dummy_stats = load_training_players("data/fifa_2023.csv")
    dummy_env = FootballGame(dummy_stats["team1"], dummy_stats["team2"], config=CONFIG)
    
    # Initialize DQN agents for both teams
    team1_dqns = [DQNAgent(state_dim=len(dummy_env._get_player_state(p)), n_actions=NUM_ACTIONS)
                  for p in dummy_env.team1.players]
    team2_dqns = [DQNAgent(state_dim=len(dummy_env._get_player_state(p)), n_actions=NUM_ACTIONS)
                  for p in dummy_env.team2.players]
    
    num_episodes = 1000
    max_steps = 400
    render_last_n = 10
    batch_size = num_workers  # Run this many episodes in parallel
    
    # Training loop
    for batch_start in range(0, num_episodes, batch_size):
        batch_end = min(batch_start + batch_size, num_episodes)
        current_batch_size = batch_end - batch_start
        
        # Get current network weights
        team1_state_dicts = [agent.q_net.state_dict() for agent in team1_dqns]
        team2_state_dicts = [agent.q_net.state_dict() for agent in team2_dqns]
        epsilon_value = team1_dqns[0].epsilon
        
        # Prepare worker arguments
        args_list = []
        for ep in range(batch_start, batch_end):
            render = ep >= num_episodes - render_last_n
            args_list.append((ep, team1_state_dicts, team2_state_dicts, epsilon_value, max_steps, render))
        
        # Run episodes in parallel (but NOT for rendering episodes)
        should_render_any = any(ep >= num_episodes - render_last_n for ep in range(batch_start, batch_end))
        
        if current_batch_size > 1 and not should_render_any:
            # Parallel execution (no rendering)
            with Pool(processes=num_workers) as pool:
                results = pool.map(run_episode_worker, args_list)
        else:
            # Sequential for rendering episodes - this allows pygame to work properly
            results = [run_episode_worker(args) for args in args_list]
        
        # Process results and update agents
        for result in results:
            ep_num = result['ep_num']
            
            # Update Team 1 agents
            for i, agent in enumerate(team1_dqns):
                for exp in result['team1_experiences'][i]:
                    agent.store_transition(
                        exp['state'],
                        exp['action'],
                        exp['reward'],
                        exp['next_state'],
                        exp['done']
                    )
                    agent.train_step()
            
            # Update Team 2 agents
            for i, agent in enumerate(team2_dqns):
                for exp in result['team2_experiences'][i]:
                    agent.store_transition(
                        exp['state'],
                        exp['action'],
                        exp['reward'],
                        exp['next_state'],
                        exp['done']
                    )
                    agent.train_step()
            
            # Logging
            if ep_num % 1 == 0 or ep_num >= num_episodes - render_last_n:
                eps = team1_dqns[0].epsilon
                print(f"Episode {ep_num} | Steps: {result['steps']} | "
                      f"Team1: {result['total_r1']:.2f} | Team2: {result['total_r2']:.2f} | "
                      f"Score: {result['score'][0]}-{result['score'][1]} | Epsilon: {eps:.3f}")
        
        # Periodic checkpoint saving
        if batch_end % 100 == 0:
            print(f"Checkpoint at episode {batch_end}...")
            os.makedirs("checkpoints3", exist_ok=True)
            for i, agent in enumerate(team1_dqns):
                agent.save(f"checkpoints3/team1_agent_{i}_ep{batch_end}.pt")
            for i, agent in enumerate(team2_dqns):
                agent.save(f"checkpoints3/team2_agent_{i}_ep{batch_end}.pt")
    
    # Final save
    print("\nSaving final models...")
    os.makedirs("checkpoints3", exist_ok=True)
    for i, agent in enumerate(team1_dqns):
        agent.save(f"checkpoints3/team1_agent_{i}.pt")
    for i, agent in enumerate(team2_dqns):
        agent.save(f"checkpoints3/team2_agent_{i}.pt")
    
    # Save coaches
    dummy_stats = load_training_players("data/fifa_2023.csv")
    dummy_env = FootballGame(dummy_stats["team1"], dummy_stats["team2"], config=CONFIG)
    team1_coach = CoachAgent(dummy_env.team1)
    team2_coach = CoachAgent(dummy_env.team2)
    
    with open("checkpoints3/team1_coach.pkl", "wb") as f:
        pickle.dump(team1_coach, f)
    with open("checkpoints3/team2_coach.pkl", "wb") as f:
        pickle.dump(team2_coach, f)
    
    print("All agents and coaches saved in checkpoints3/")


def train_8_agents():
    """
    Original sequential training (keeping for reference).
    """
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
    render_last_n = 10

    for ep in range(num_episodes):
        render = ep >= num_episodes - render_last_n
        stats = load_training_players("data/fifa_2023.csv")

        # Create new environment
        env = FootballGame(stats["team1"], stats["team2"], config=CONFIG)
        
        # Update player references
        for i, player_agent in enumerate(team1_players):
            player_agent.player = env.team1.players[i]
        for i, player_agent in enumerate(team2_players):
            player_agent.player = env.team2.players[i]

        done = False
        total_r1 = total_r2 = 0.0
        obs = env.reset()

        import random
        if random.random() < 0.5:
            random_player = random.choice(env.team2.players)
            random_player.has_ball = True
            env.ball.x = random_player.x
            env.ball.y = random_player.y
            for p in env.team1.players:
                p.has_ball = False

        step_count = 0

        while not done and step_count < max_steps:
            current_states_t1 = [env._get_player_state(p) for p in env.team1.players]
            current_states_t2 = [env._get_player_state(p) for p in env.team2.players]

            team1_actions = get_team_actions(team1_coach, team1_players, env, team_id=1, 
                                            states=current_states_t1)
            team2_actions = get_team_actions(team2_coach, team2_players, env, team_id=2,
                                            states=current_states_t2)

            next_obs, (rew1, rew2), game_done = env.step(team1_actions, team2_actions)

            next_states_t1 = [env._get_player_state(p) for p in env.team1.players]
            next_states_t2 = [env._get_player_state(p) for p in env.team2.players]

            for i, player_agent in enumerate(team1_players):
                if hasattr(player_agent, 'player_agent') and player_agent.player_agent is not None:
                    s = current_states_t1[i]
                    s_next = next_states_t1[i]
                    r = rew1[i]
                    action_idx = player_agent.last_action_idx if hasattr(player_agent, 'last_action_idx') else 0
                    
                    player_agent.player_agent.store_transition(s, action_idx, r, s_next, done or game_done)
                    player_agent.player_agent.train_step()

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

            if abs(env.score_left - env.score_right) >= 5:
                done = True

            if render:
                env.render()

        if ep % 10 == 0 or render:
            eps = team1_dqns[0].epsilon
            print(f"Episode {ep} | Steps: {step_count} | Team1: {total_r1:.2f} | Team2: {total_r2:.2f} | Epsilon: {eps:.3f}")

    # Save all agents
    os.makedirs("checkpoints3", exist_ok=True)
    for i, agent in enumerate(team1_dqns):
        agent.save(f"checkpoints3/team1_agent_{i}.pt")
    for i, agent in enumerate(team2_dqns):
        agent.save(f"checkpoints3/team2_agent_{i}.pt")

    with open("checkpoints3/team1_coach.pkl", "wb") as f:
        pickle.dump(team1_coach, f)
    with open("checkpoints3/team2_coach.pkl", "wb") as f:
        pickle.dump(team2_coach, f)

    print("All agents and coaches saved in checkpoints3/")


# if __name__ == "__main__":
#     # Choose parallel or sequential training
    
#     # Parallel training (recommended - faster)
#     train_8_agents_parallel(num_workers=8)
    
#     # Sequential training (original)
#     # train_8_agents()