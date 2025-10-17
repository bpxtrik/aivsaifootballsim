import numpy as np
from config import ACTION_SPACE, NUM_ACTIONS

class PlayerAgent:
    def __init__(self, player, role="field"):
        self.player = player
        self.role = role

    def act(self, game_state, action_int=None):
        if action_int is None:
            action_int = np.random.randint(0, NUM_ACTIONS)
        action = ACTION_SPACE[action_int]

        if self.role == "goalkeeper":
            # Define goal area
            goal_area_depth = 80
            goal_height = 200
            goal_top = (game_state.height - goal_height) // 2
            goal_bottom = (game_state.height + goal_height) // 2

            if self.player.team_id == 1:  # left goal
                x_min, x_max = 5, goal_area_depth
            else:  # right goal
                x_min, x_max = game_state.width - goal_area_depth, game_state.width - 5

            # Calculate desired movement toward ball
            dx = game_state.ball.x - self.player.x
            dy = game_state.ball.y - self.player.y
            
            # Limit movement to max_speed
            dx = max(-self.player.max_speed, min(self.player.max_speed, dx))
            dy = max(-self.player.max_speed, min(self.player.max_speed, dy))

            # Calculate tentative next position
            next_x = self.player.x + dx
            next_y = self.player.y + dy

            # Clamp to goal area
            clamped_x = max(x_min, min(x_max, next_x))
            clamped_y = max(goal_top, min(goal_bottom, next_y))

            # Return the DIFFERENCE (delta) from current position
            return {"move": (clamped_x - self.player.x, clamped_y - self.player.y),
                    "shoot": False,
                    "pass_to": None}

        return action



