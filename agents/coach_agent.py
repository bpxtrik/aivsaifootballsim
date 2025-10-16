import random

class CoachAgent:
    def __init__(self, team):
        self.team = team

    def strategy(self, state):
        """
        Returns a dictionary mapping player index to actions.
        Example: {0: {"move": (vx, vy), "shoot": True, "pass_to": 2}, ...}
        """
        actions = {}
        for i, player in enumerate(self.team.players):
            if i == 0:  # goalkeeper handled separately
                continue
            # Assign heuristic: move toward ball, pass if teammate nearby
            dx = state['ball_x'] - player.x
            dy = state['ball_y'] - player.y
            speed = min((dx**2 + dy**2)**0.5, player.max_speed)
            vx, vy = (dx / ((dx**2 + dy**2)**0.5) * speed if dx or dy else 0,
                      dy / ((dx**2 + dy**2)**0.5) * speed if dx or dy else 0)
            
            # Randomly pass to teammate if in possession
            pass_to = None
            if player.has_ball:
                teammates = [j for j in range(len(self.team.players)) if j != i]
                if teammates:
                    pass_to = random.choice(teammates)
            
            actions[i] = {"move": (vx, vy), "shoot": False, "pass_to": pass_to}
        return actions
