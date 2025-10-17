"""
Ray RLlib implementation for 8-agent football training.
This is MUCH faster and more efficient than custom parallel code.

Install: pip install ray[rllib] gymnasium
"""

import numpy as np
from typing import Dict, Tuple
import gymnasium as gym
from gymnasium import spaces
from ray import tune
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.policy.policy import PolicySpec

from environment.game import FootballGame
from data.data_loader import load_training_players
from config import CONFIG, NUM_ACTIONS, ACTION_SPACE


class FootballMultiAgentEnv(MultiAgentEnv):
    """
    Wrapper to make your football game compatible with RLlib's multi-agent API.
    """
    def __init__(self, config=None):
        super().__init__()
        self.config = config or {}
        
        # Load teams
        stats = load_training_players("data/fifa_2023.csv")
        self.game = FootballGame(stats["team1"], stats["team2"], config=CONFIG)
        
        # Define agent IDs
        self._agent_ids = set()
        for i in range(len(self.game.team1.players)):
            self._agent_ids.add(f"team1_player_{i}")
        for i in range(len(self.game.team2.players)):
            self._agent_ids.add(f"team2_player_{i}")
        
        # Get state dimension from first player
        sample_state = self.game._get_player_state(self.game.team1.players[0])
        state_dim = len(sample_state)
        
        # Define observation and action spaces as DICTIONARIES for multi-agent
        single_obs_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32
        )
        single_action_space = spaces.Discrete(NUM_ACTIONS)
        
        # RLlib expects dictionaries mapping agent_id -> space
        self.observation_space = spaces.Dict({
            agent_id: single_obs_space for agent_id in self._agent_ids
        })
        self.action_space = spaces.Dict({
            agent_id: single_action_space for agent_id in self._agent_ids
        })
        
        self.max_steps = 400
        self.current_step = 0
        
        # Required by RLlib for multi-agent envs
        self.agents = list(self._agent_ids)
        self.possible_agents = list(self._agent_ids)
    
    def reset(self, *, seed=None, options=None):
        """Reset the environment."""
        # Reload fresh stats for variety
        stats = load_training_players("data/fifa_2023.csv")
        self.game = FootballGame(stats["team1"], stats["team2"], config=CONFIG)
        
        # Random ball possession
        import random
        if random.random() < 0.5:
            random_player = random.choice(self.game.team2.players)
            random_player.has_ball = True
            self.game.ball.x = random_player.x
            self.game.ball.y = random_player.y
            for p in self.game.team1.players:
                p.has_ball = False
        
        self.current_step = 0
        
        # Get initial observations for all agents
        obs = {}
        for i, player in enumerate(self.game.team1.players):
            obs[f"team1_player_{i}"] = self.game._get_player_state(player).astype(np.float32)
        for i, player in enumerate(self.game.team2.players):
            obs[f"team2_player_{i}"] = self.game._get_player_state(player).astype(np.float32)
        
        return obs, {}
    
    def step(self, action_dict: Dict[str, int]) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
        """
        Take a step in the environment.
        action_dict: {agent_id: action_index}
        """
        # Convert action indices to actual actions using ACTION_SPACE
        team1_actions = {}
        team2_actions = {}
        
        for agent_id, action_idx in action_dict.items():
            # Convert action index to actual action tuple from ACTION_SPACE
            actual_action = ACTION_SPACE[int(action_idx)]
            
            if agent_id.startswith("team1_player_"):
                idx = int(agent_id.split("_")[-1])
                team1_actions[idx] = actual_action
            else:  # team2
                idx = int(agent_id.split("_")[-1])
                team2_actions[idx] = actual_action
        
        # Step the game
        _, (rew1, rew2), game_done = self.game.step(team1_actions, team2_actions)
        
        # Get next observations
        obs = {}
        for i, player in enumerate(self.game.team1.players):
            obs[f"team1_player_{i}"] = self.game._get_player_state(player).astype(np.float32)
        for i, player in enumerate(self.game.team2.players):
            obs[f"team2_player_{i}"] = self.game._get_player_state(player).astype(np.float32)
        
        # Get rewards
        rewards = {}
        for i in range(len(self.game.team1.players)):
            rewards[f"team1_player_{i}"] = float(rew1[i])
        for i in range(len(self.game.team2.players)):
            rewards[f"team2_player_{i}"] = float(rew2[i])
        
        # Check termination
        self.current_step += 1
        terminated = game_done or self.current_step >= self.max_steps
        terminated = terminated or abs(self.game.score_left - self.game.score_right) >= 5
        
        terminateds = {agent_id: terminated for agent_id in self._agent_ids}
        terminateds["__all__"] = terminated
        
        truncateds = {agent_id: False for agent_id in self._agent_ids}
        truncateds["__all__"] = False
        
        infos = {agent_id: {} for agent_id in self._agent_ids}
        
        return obs, rewards, terminateds, truncateds, infos


def train_with_rllib():
    """
    Train using Ray RLlib - handles parallelization automatically!
    """
    
    # Register the environment
    from ray.tune.registry import register_env
    register_env("football", lambda config: FootballMultiAgentEnv(config))
    
    # Create a dummy env to get agent IDs
    dummy_env = FootballMultiAgentEnv()
    agent_ids = list(dummy_env._agent_ids)
    
    # Configure DQN for multi-agent (updated API)
    config = (
        DQNConfig()
        .environment("football")
        .framework("torch")
        .training(
            lr=1e-3,
            gamma=0.99,
            train_batch_size=64,
            replay_buffer_config={
                "type": "MultiAgentReplayBuffer",
                "capacity": 50000,
            },
            target_network_update_freq=150,
            double_q=True,
            dueling=True,
            n_step=1,
        )
        .multi_agent(
            policies={
                agent_id: PolicySpec(
                    config=DQNConfig.overrides(
                        exploration_config={
                            "type": "EpsilonGreedy",
                            "initial_epsilon": 1.0,
                            "final_epsilon": 0.05,
                            "epsilon_timesteps": 250000,
                        }
                    )
                ) for agent_id in agent_ids
            },
            policy_mapping_fn=lambda agent_id, *args, **kwargs: agent_id,
        )
        .env_runners(
            num_env_runners=4,  # Parallel workers - adjust based on CPU cores
            num_envs_per_env_runner=1,
            rollout_fragment_length=50,
        )
        .resources(
            num_gpus=1 if torch.cuda.is_available() else 0,
        )
        .debugging(
            log_level="WARN",
        )
    )
    
    # Build the algorithm
    algo = config.build()
    
    # Training loop
    num_iterations = 1000
    checkpoint_freq = 100
    
    print("Starting Ray RLlib training...")
    print(f"Using {config.num_env_runners} parallel workers")
    
    for i in range(num_iterations):
        result = algo.train()
        
        # Logging
        if i % 10 == 0:
            print(f"\nIteration {i}")
            print(f"  Episode Reward Mean: {result.get('episode_reward_mean', 0):.2f}")
            print(f"  Episode Length Mean: {result.get('episode_len_mean', 0):.1f}")
            print(f"  Timesteps Total: {result.get('timesteps_total', 0)}")
            
            # Print per-agent info if available
            if 'policy_reward_mean' in result:
                for agent_id in agent_ids[:2]:  # Print first 2 agents
                    if agent_id in result['policy_reward_mean']:
                        print(f"  {agent_id}: {result['policy_reward_mean'][agent_id]:.2f}")
        
        # Checkpoint
        if (i + 1) % checkpoint_freq == 0:
            checkpoint_dir = algo.save(checkpoint_dir="./ray_checkpoints")
            print(f"Checkpoint saved at iteration {i + 1}: {checkpoint_dir}")
    
    # Final checkpoint
    final_checkpoint = algo.save(checkpoint_dir="./ray_checkpoints")
    print(f"\nTraining complete! Final checkpoint: {final_checkpoint}")
    
    algo.stop()


if __name__ == "__main__":
    import torch
    import ray
    
    # Initialize Ray
    ray.init(ignore_reinit_error=True)
    
    try:
        train_with_rllib()
    finally:
        ray.shutdown()