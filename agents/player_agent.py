import numpy as np
from config import ACTION_SPACE, NUM_ACTIONS

class PlayerAgent:
    def __init__(self, player, role="field"):
        self.player = player
        self.role = role

    def act(self, game_state, action_int=None):
        """
        If action_int is None, choose randomly.
        Otherwise, map integer to dictionary action.
        """
        if action_int is None:
            action_int = np.random.randint(0, NUM_ACTIONS)
        action = ACTION_SPACE[action_int]

        # Goalkeeper simple override
        if self.role == "goalkeeper":
            goal_x = 10 if self.player.team_id == 1 else game_state.width - 10
            goal_y = game_state.height // 2
            dy = game_state.ball.y - self.player.y
            dy = max(-self.player.max_speed, min(self.player.max_speed, dy))
            return {"move": (0, dy)}  # keep GK on goal

        return action



