import pygame
import numpy as np
from environment.player import Player
from environment.ball import Ball
from environment.team import Team
from environment.utils import resolve_circle_circle_collision

class FootballGame:
    def __init__(self, config):
        self.config = config
        self.width = config["field_width"]
        self.height = config["field_height"]

        # Entities
        self.ball = Ball(self.width // 2, self.height // 2)
        self.team1 = Team(1, config, "left")
        self.team2 = Team(2, config, "right")

        # Scoring
        self.score_left = 0
        self.score_right = 0
        self.last_scored = None

        self.done = False

        # Pygame setup (optional for rendering)
        if hasattr(pygame, "init"):
            pygame.init()  # pylint: disable=no-member
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.clock = pygame.time.Clock()

    def reset(self):
        self.score_left = 0
        self.score_right = 0
        self.last_scored = None
        self._kickoff(starting_side="left")
        self.done = False
        return self._get_state()

    def step(self, actions_team1=None, actions_team2=None):
        if actions_team1 is None:
            actions_team1 = {}
        if actions_team2 is None:
            actions_team2 = {}

        # Update players
        self.team1.update(actions_team1, self.ball)
        self.team2.update(actions_team2, self.ball)

        # Possession/dribble: handle pickup and shooting, then attach ball if possessed
        self._update_possession(actions_team1, actions_team2)

        # Player-player collisions
        self._resolve_all_player_collisions()

        # Player-ball collisions (pure contact)
        for player in self.team1.players + self.team2.players:
            self._resolve_player_ball(player)

        # Update ball with goal openings
        goal_top = (self.height - 200) // 2
        goal_bottom = (self.height + 200) // 2
        self.ball.update(self.width, self.height, goal_opening=(goal_top, goal_bottom))

        # Check goals
        goal = self._check_goal(goal_top, goal_bottom)
        if goal == "left":
            self.score_right += 1
            self.last_scored = "right"
            self._kickoff(starting_side="right")
        elif goal == "right":
            self.score_left += 1
            self.last_scored = "left"
            self._kickoff(starting_side="left")

        rewards = self._calculate_rewards()
        state = self._get_state()
        return state, rewards, self.done

    def _calculate_rewards(self):
        # TODO: implement reward logic later
        return 0

    def _get_state(self):
        # TODO: encode positions + stats into numpy array
        return np.array([])

    def _resolve_player_ball(self, player: Player):
        # If player has ball, keep it attached just in front of player
        if player.has_ball:
            offset = player.radius + self.ball.radius - 1
            # Attach ball ahead in direction of velocity or default to facing right/left by team
            vx, vy = player.vx, player.vy
            speed = (vx * vx + vy * vy) ** 0.5
            if speed > 0.01:
                nx, ny = vx / speed, vy / speed
            else:
                nx, ny = (1.0, 0.0) if player.team_id == 1 else (-1.0, 0.0)
            self.ball.x = player.x + nx * offset
            self.ball.y = player.y + ny * offset
            # Dribble carries ball velocity
            self.ball.vx = player.vx
            self.ball.vy = player.vy
            return

        # Otherwise collide normally
        x1, y1, vx1, vy1 = player.x, player.y, player.vx, player.vy
        x2, y2, vx2, vy2 = self.ball.x, self.ball.y, self.ball.vx, self.ball.vy
        x1, y1, vx1, vy1, x2, y2, vx2, vy2 = resolve_circle_circle_collision(
            x1, y1, vx1, vy1, player.radius, player.mass,
            x2, y2, vx2, vy2, self.ball.radius, self.ball.mass,
            restitution=0.8,
        )
        player.x, player.y, player.vx, player.vy = x1, y1, vx1, vy1
        self.ball.x, self.ball.y, self.ball.vx, self.ball.vy = x2, y2, vx2, vy2

    def _kickoff(self, starting_side: str = "left"):
        self.team1.reset()
        self.team2.reset()
        self.ball.reset(self.width // 2, self.height // 2)

        # Pass back to own team: give a small impulse towards own half
        impulse = 5.0
        if starting_side == "left":
            self.ball.apply_impulse(-impulse, 0.0)
        else:
            self.ball.apply_impulse(impulse, 0.0)

    def _check_goal(self, goal_top: int, goal_bottom: int):
        # Left goal
        if self.ball.x - self.ball.radius <= 0 and goal_top <= self.ball.y <= goal_bottom:
            return "left"
        # Right goal
        if self.ball.x + self.ball.radius >= self.width and goal_top <= self.ball.y <= goal_bottom:
            return "right"
        return None

    def _resolve_all_player_collisions(self):
        players = self.team1.players + self.team2.players
        n = len(players)
        for i in range(n):
            for j in range(i + 1, n):
                p1 = players[i]
                p2 = players[j]
                x1, y1, vx1, vy1 = p1.x, p1.y, p1.vx, p1.vy
                x2, y2, vx2, vy2 = p2.x, p2.y, p2.vx, p2.vy
                x1, y1, vx1, vy1, x2, y2, vx2, vy2 = resolve_circle_circle_collision(
                    x1, y1, vx1, vy1, p1.radius, p1.mass,
                    x2, y2, vx2, vy2, p2.radius, p2.mass,
                    restitution=0.2,
                )
                p1.x, p1.y, p1.vx, p1.vy = x1, y1, vx1, vy1
                p2.x, p2.y, p2.vx, p2.vy = x2, y2, vx2, vy2

    def _update_possession(self, actions_team1, actions_team2):
        # Determine if any player can take possession
        control_radius = 6
        best_player = None
        best_dist2 = 1e9
        for p in self.team1.players + self.team2.players:
            dx = self.ball.x - p.x
            dy = self.ball.y - p.y
            dist2 = dx * dx + dy * dy
            threshold = (p.radius + self.ball.radius + control_radius)
            if dist2 <= threshold * threshold and self.ball.pickup_cooldown == 0:
                if dist2 < best_dist2:
                    best_dist2 = dist2
                    best_player = p

        # Clear previous possession
        for p in self.team1.players + self.team2.players:
            p.has_ball = False

        if best_player is not None:
            best_player.has_ball = True
            # Handle shooting intent: if action includes shoot, kick the ball
            shoot = False
            if best_player.team_id == 1:
                act = actions_team1.get(3) if best_player is self.team1.players[3] else None
            else:
                act = actions_team2.get(3) if best_player is self.team2.players[3] else None
            if isinstance(act, dict):
                shoot = bool(act.get("shoot", False))

            if shoot:
                # Compute target towards enemy goal center
                if best_player.team_id == 1:
                    gx, gy = self.width - 5, self.height // 2
                else:
                    gx, gy = 5, self.height // 2
                dx = gx - best_player.x
                dy = gy - best_player.y
                mag = (dx * dx + dy * dy) ** 0.5
                if mag > 0:
                    nx, ny = dx / mag, dy / mag
                else:
                    nx, ny = (1.0, 0.0)
                power = best_player.kick_power
                self.ball.vx = best_player.vx + nx * power
                self.ball.vy = best_player.vy + ny * power
                self.ball.pickup_cooldown = 20
                best_player.has_ball = False

    def render(self):
        # Fill background green (field)
        self.screen.fill((0, 128, 0))

        # --- FIELD LINES ---
        white = (255, 255, 255)
        field_rect = pygame.Rect(0, 0, self.width, self.height)

        # Outer border
        pygame.draw.rect(self.screen, white, field_rect, 5)

        # Midfield line
        pygame.draw.line(
            self.screen,
            white,
            (self.width // 2, 0),
            (self.width // 2, self.height),
            3,
        )

        # Center circle
        pygame.draw.circle(
            self.screen,
            white,
            (self.width // 2, self.height // 2),
            70,
            3,
        )

        # Goals (simple rectangles)
        goal_height = 200

        # Left goal
        left_goal = pygame.Rect(0, (self.height - goal_height) // 2, 10, goal_height)
        pygame.draw.rect(self.screen, white, left_goal)

        # Right goal
        right_goal = pygame.Rect(self.width - 10, (self.height - goal_height) // 2, 10, goal_height)
        pygame.draw.rect(self.screen, white, right_goal)

        # --- ENTITIES ---
        self.ball.draw(self.screen)
        self.team1.draw(self.screen)
        self.team2.draw(self.screen)

        # Scoreboard
        font = pygame.font.SysFont(None, 24)
        score_text = font.render(f"Blue {self.score_left} - {self.score_right} Red", True, white)
        self.screen.blit(score_text, (self.width // 2 - score_text.get_width() // 2, 10))

        # Update display
        pygame.display.flip()
        self.clock.tick(60)  # limit FPS

