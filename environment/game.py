import pygame
import numpy as np
from agents.coach_agent import CoachAgent
from agents.player_agent import PlayerAgent
from environment.player import Player
from environment.ball import Ball
from environment.team import Team
from environment.utils import resolve_circle_circle_collision
import random

class FootballGame:
    def __init__(self, team1, team2, config):
        self.config = config
        self.width = config["field_width"]
        self.height = config["field_height"]

        # Entities
        self.ball = Ball(self.width // 2, self.height // 2)
        self.team1 = Team(1, config, "left", team1)
        self.team2 = Team(2, config, "right", team2)

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

        self.agents_team1 = [PlayerAgent(p, role="goalkeeper" if i == 0 else "field") 
                     for i, p in enumerate(self.team1.players)]
        self.agents_team2 = [PlayerAgent(p, role="goalkeeper" if i == 0 else "field") 
                            for i, p in enumerate(self.team2.players)]

        # Coach agents
        self.coach_team1 = CoachAgent(self.team1)
        self.coach_team2 = CoachAgent(self.team2)

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

        # Detect if we’re in RL mode (actions are ints) or manual/AI mode (dicts)
        if len(actions_team1) > 0 and isinstance(next(iter(actions_team1.values())), int):
            actions_team1 = {
                i: self.agents_team1[i].act(self, action_int=a)
                for i, a in actions_team1.items()
            }
        if len(actions_team2) > 0 and isinstance(next(iter(actions_team2.values())), int):
            actions_team2 = {
                i: self.agents_team2[i].act(self, action_int=a)
                for i, a in actions_team2.items()
            }

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
    
    def step_with_agents(self, player_agents_team1, player_agents_team2):
        # Build a minimal state representation for agents
        state = {
            'ball_x': self.ball.x,
            'ball_y': self.ball.y,
            'field_width': self.width,
            'field_height': self.height,
            'goal_top': (self.height - 200)//2,
            'goal_bottom': (self.height + 200)//2
        }

        # Get actions from all player agents
        actions_team1 = {i: agent.act(state) for i, agent in enumerate(player_agents_team1)}
        actions_team2 = {i: agent.act(state) for i, agent in enumerate(player_agents_team2)}

        # Call the original step with agent actions
        return self.step(actions_team1, actions_team2)


    def _calculate_rewards(self):
        """
        Balanced reward system for DQN training:
        - Positive rewards for scoring and moving toward goal
        - Small incentives for ball chasing
        - Rewards clamped per step to prevent spikes
        """
        rewards_team1 = np.zeros(len(self.team1.players))
        rewards_team2 = np.zeros(len(self.team2.players))

        # --- Goal rewards (applied once per goal) ---
        if self.last_scored == "left":
            rewards_team1 += 5.0   # reward for scoring
            rewards_team2 -= 2.5   # penalty for conceding
        elif self.last_scored == "right":
            rewards_team1 -= 2.5
            rewards_team2 += 5.0

        # --- Movement / possession rewards ---
        for i, p in enumerate(self.team1.players):
            reward = 0.0
            if p.has_ball:
                # Reward for having the ball
                reward += 0.05
                # Progress toward opponent goal
                progress = p.x / self.width
                reward += progress * 0.02
                # Small reward for moving forward
                reward += max(0, p.vx) * 0.005
            else:
                # Reward for moving toward the ball
                dx = self.ball.x - p.x
                dy = self.ball.y - p.y
                dist_to_ball = max((dx**2 + dy**2)**0.5, 1e-5)  # avoid div by zero
                ball_dx = dx / dist_to_ball
                ball_dy = dy / dist_to_ball
                velocity_toward_ball = p.vx * ball_dx + p.vy * ball_dy
                reward += max(0, velocity_toward_ball) * 0.01
                # Small reward for being close to ball
                if dist_to_ball < 100:
                    reward += 0.01 * (100 - dist_to_ball) / 100

            # Clamp per-step reward to prevent spikes
            rewards_team1[i] = np.clip(reward, -1.0, 1.0)

        for i, p in enumerate(self.team2.players):
            reward = 0.0
            if p.has_ball:
                reward += 0.05
                progress = (self.width - p.x) / self.width
                reward += progress * 0.02
                reward += max(0, -p.vx) * 0.005
            else:
                dx = self.ball.x - p.x
                dy = self.ball.y - p.y
                dist_to_ball = max((dx**2 + dy**2)**0.5, 1e-5)
                ball_dx = dx / dist_to_ball
                ball_dy = dy / dist_to_ball
                velocity_toward_ball = p.vx * ball_dx + p.vy * ball_dy
                reward += max(0, velocity_toward_ball) * 0.01
                if dist_to_ball < 100:
                    reward += 0.01 * (100 - dist_to_ball) / 100

            rewards_team2[i] = np.clip(reward, -1.0, 1.0)

        return rewards_team1, rewards_team2

    def _get_state(self):
        """
        Returns the game state as a vector for all agents.
        Each agent sees:
        - Its own position and velocity
        - Ball position and velocity
        - Relative positions/velocities of teammates
        - Relative positions/velocities of opponents
        - Ball possession
        """
        state = []

        # Ball info
        ball_info = [
            self.ball.x / self.width,
            self.ball.y / self.height,
            self.ball.vx / 10.0,  # normalize by reasonable speed
            self.ball.vy / 10.0,
            self.ball.pickup_cooldown / 20.0
        ]

        for team, opponents in [(self.team1, self.team2), (self.team2, self.team1)]:
            for player in team.players:
                # Own player info
                player_info = [
                    player.x / self.width,
                    player.y / self.height,
                    player.vx / 10.0,
                    player.vy / 10.0,
                    1.0 if player.has_ball else 0.0
                ]

                # Teammates relative info
                for mate in team.players:
                    if mate is player:
                        continue
                    player_info.extend([
                        (mate.x - player.x) / self.width,
                        (mate.y - player.y) / self.height,
                        (mate.vx - player.vx) / 10.0,
                        (mate.vy - player.vy) / 10.0,
                        1.0 if mate.has_ball else 0.0
                    ])

                # Opponents relative info
                for opp in opponents.players:
                    player_info.extend([
                        (opp.x - player.x) / self.width,
                        (opp.y - player.y) / self.height,
                        (opp.vx - player.vx) / 10.0,
                        (opp.vy - player.vy) / 10.0,
                        1.0 if opp.has_ball else 0.0
                    ])

                # Add ball info relative to player
                player_info.extend([
                    (self.ball.x - player.x) / self.width,
                    (self.ball.y - player.y) / self.height,
                    self.ball.vx / 10.0,
                    self.ball.vy / 10.0,
                    1.0 if player.has_ball else 0.0
                ])

                state.append(player_info)

            state = np.array(state, dtype=np.float32)
            return state.flatten()


    def _resolve_player_ball(self, player: Player):
        """
        Handles ball-player interaction:
        - Smooth dribbling when player has possession
        - Normal physics when not possessed
        """
        # If player has ball or is last owner and still possessing
        if player.has_ball or (self.ball.last_owner == player and self.ball.has_possessor):
            offset = player.radius + self.ball.radius - 1
            vx, vy = player.vx, player.vy
            speed = (vx*vx + vy*vy)**0.5

            # Determine direction to carry the ball
            if speed > 0.01:
                nx, ny = vx / speed, vy / speed
            else:
                # Default direction: toward opponent goal
                nx, ny = (1.0, 0.0) if player.team_id == 1 else (-1.0, 0.0)

            target_x = player.x + nx * offset
            target_y = player.y + ny * offset

            # Interpolate ball position for smooth dribbling
            interp_factor = 0.6
            self.ball.x += (target_x - self.ball.x) * interp_factor
            self.ball.y += (target_y - self.ball.y) * interp_factor

            # Carry player's velocity
            self.ball.vx = player.vx
            self.ball.vy = player.vy

            # Update possession
            self.ball.last_owner = player
            self.ball.has_possessor = True
            return

        # Normal collision if player does not have possession
        x1, y1, vx1, vy1 = player.x, player.y, player.vx, player.vy
        x2, y2, vx2, vy2 = self.ball.x, self.ball.y, self.ball.vx, self.ball.vy

        x1, y1, vx1, vy1, x2, y2, vx2, vy2 = resolve_circle_circle_collision(
            x1, y1, vx1, vy1, player.radius, player.mass,
            x2, y2, vx2, vy2, self.ball.radius, self.ball.mass,
            restitution=0.8
        )

        player.x, player.y, player.vx, player.vy = x1, y1, vx1, vy1
        self.ball.x, self.ball.y, self.ball.vx, self.ball.vy = x2, y2, vx2, vy2





    def _kickoff(self, starting_side: str = "left"):
        self.team1.reset()
        self.team2.reset()
        self.ball.reset(self.width // 2, self.height // 2)

        # Pass back to own team: small impulse toward own half
        power = 5.0
        if starting_side == "left":
            direction = (-1.0, 0.0)
        else:
            direction = (1.0, 0.0)

        # Use the new signature (no player needed here)
        self.ball.apply_impulse(None, direction, power, kick_type="pass")

    # def _check_goal(self, goal_top: int, goal_bottom: int):
    #     # Left goal
    #     if self.ball.x - self.ball.radius <= 0 and goal_top <= self.ball.y <= goal_bottom:
    #         return "left"
    #     # Right goal
    #     if self.ball.x + self.ball.radius >= self.width and goal_top <= self.ball.y <= goal_bottom:
    #         return "right"
    #     return None

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
        """
        Updates which player possesses the ball and executes actions:
        - Shoot: ball moves toward goal with SHO precision and physics
        - Pass: ball moves toward teammate, chance to intercept by opponent
        - Dribble: player keeps ball (handled by _resolve_player_ball)
        """
        control_radius = 6
        nearby_players = []

        # Find players near the ball (only consider if pickup cooldown is over)
        for p in self.team1.players + self.team2.players:
            dx = self.ball.x - p.x
            dy = self.ball.y - p.y
            if dx*dx + dy*dy <= (p.radius + self.ball.radius + control_radius)**2 and self.ball.pickup_cooldown == 0:
                nearby_players.append(p)

        if not nearby_players:
            return

        # Resolve duels only if no teammate already has possession nearby
        for p in nearby_players:
            team = self.team1.players if p.team_id == 1 else self.team2.players
            teammates_nearby = any(
                t.has_ball or (self.ball.last_owner == t and self.ball.has_possessor)
                for t in team if t != p
            )
            if teammates_nearby:
                # If a teammate already controls the ball nearby, skip duel
                return

        # Determine winner of the duel
        winner = nearby_players[0]
        for p in nearby_players[1:]:
            if winner.team_id != p.team_id:
                winner = winner if self.physical_duel(winner, p) else p

        # Assign possession
        for p in self.team1.players + self.team2.players:
            p.has_ball = False
        winner.has_ball = True
        self.ball.last_owner = winner
        self.ball.has_possessor = True

        # Determine action
        team_actions = actions_team1 if winner.team_id == 1 else actions_team2
        action = team_actions.get((self.team1.players if winner.team_id == 1 else self.team2.players).index(winner))

        if not isinstance(action, dict):
            return

        # --- SHOOT ---
        if action.get("shoot", False):
            # Determine target and impulse
            target_x = self.width if winner.team_id == 1 else 0
            target_y = self.ball.y
            dx = target_x - winner.x
            dy = target_y - winner.y
            # Apply impulse to ball
            self.ball.apply_impulse(winner, (dx / 5, dy / 5), power=1.0, kick_type="shoot")
            winner.has_ball = False
            self.ball.pickup_cooldown = 20

        # --- PASS ---
        elif "pass_to" in action:
            teammate_idx = action["pass_to"]
            teammate = (self.team1.players if winner.team_id == 1 else self.team2.players)[teammate_idx]
            opponents = self.team2.players if winner.team_id == 1 else self.team1.players
            closest_opp = min(opponents, key=lambda o: (o.x - teammate.x)**2 + (o.y - teammate.y)**2)
            success_prob = pass_success(winner, teammate, closest_opp)
            if random.random() < success_prob:
                dx = teammate.x - winner.x
                dy = teammate.y - winner.y
                self.ball.apply_impulse(winner, (dx / 5, dy / 5), power=1.0, kick_type="pass")
                winner.has_ball = False
            else:
                # Opponent intercepts
                self.ball.last_owner = closest_opp
                self.ball.has_possessor = True
                winner.has_ball = False
                closest_opp.has_ball = True

        # --- DRIBBLE / MOVE ---
        elif action.get("dribble", False) or ("move" in action and action["move"] != (0, 0)):
            # Ball remains attached to player
            self.ball.last_owner = winner
            self.ball.has_possessor = True
            # _resolve_player_ball handles smooth dribbling



    def _check_goal(self, goal_top, goal_bottom):
        """Return 'left', 'right', or None. Only if ball crosses line within goal opening."""
        if self.ball.x - self.ball.radius <= 0 and goal_top <= self.ball.y <= goal_bottom:
            if self.ball.kick_type == "shoot":
                scorer = self.ball.last_kicker
                if scorer:
                    self.last_scored = scorer.team_id
            return "left"
        if self.ball.x + self.ball.radius >= self.width and goal_top <= self.ball.y <= goal_bottom:
            if self.ball.kick_type == "shoot":
                scorer = self.ball.last_kicker
                if scorer:
                    self.last_scored = scorer.team_id
            return "right"
        return None




    def physical_duel(self, player1, player2):
        """
        Resolve a duel between attacker and defender.
        Returns True if attacker keeps possession, False if defender wins.
        """
        # Probabilities for success
        dribble_prob = dribble_success(player1, player2)  # between 0 and 1
        tackle_prob = tackle_success(player2, player1)    # between 0 and 1

        # Combine probabilities into a single possession chance for attacker
        # Attacker succeeds if their dribble beats defender's tackle
        # One simple model: attacker success = dribble_prob * (1 - tackle_prob)
        attacker_success_chance = dribble_prob * (1 - tackle_prob)

        # Randomly resolve outcome
        if random.random() < attacker_success_chance:
            return True  # attacker keeps ball
        else:
            return False  # defender wins ball

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

    def _get_player_state(self, player):
        """
        Return a normalized state vector for a single player.
        Includes:
        - Own position, velocity, possession
        - Relative teammates info
        - Relative opponents info
        - Ball position/velocity relative to player
        """
        # Identify which team and which opponents
        team = self.team1 if player in self.team1.players else self.team2
        opponents = self.team2 if team is self.team1 else self.team1

        state = [
            player.x / self.width,
            player.y / self.height,
            player.vx / 10.0,
            player.vy / 10.0,
            1.0 if player.has_ball else 0.0
        ]

        # Relative teammates
        for mate in team.players:
            if mate is player:
                continue
            state.extend([
                (mate.x - player.x) / self.width,
                (mate.y - player.y) / self.height,
                (mate.vx - player.vx) / 10.0,
                (mate.vy - player.vy) / 10.0,
                1.0 if mate.has_ball else 0.0
            ])

        # Relative opponents
        for opp in opponents.players:
            state.extend([
                (opp.x - player.x) / self.width,
                (opp.y - player.y) / self.height,
                (opp.vx - player.vx) / 10.0,
                (opp.vy - player.vy) / 10.0,
                1.0 if opp.has_ball else 0.0
            ])

        # Ball info relative to player
        state.extend([
            (self.ball.x - player.x) / self.width,
            (self.ball.y - player.y) / self.height,
            self.ball.vx / 10.0,
            self.ball.vy / 10.0,
            1.0 if player.has_ball else 0.0
        ])

        return np.array(state, dtype=np.float32)


def dribble_success(dribbler, defender):
    return dribbler.DRI / (dribbler.DRI + defender.DEF)

def shot_success(player: Player, goalkeeper: Player) -> float:
    """
    Returns a probability (0 to 1) that the player's shot beats the goalkeeper.
    Factors considered:
      - SHO (player shooting skill)
      - PHY (player shot power)
      - GK PHY (goalkeeper physical)
      - GK DEF (goalkeeper defense)
    """
    # Normalize skills
    shoot_skill = player.SHO / 100.0
    shot_power = player.PHY / 100.0
    gk_def = goalkeeper.DEF / 100.0
    gk_phy = goalkeeper.PHY / 100.0

    # Weighted probability formula
    base_prob = 0.4 * shoot_skill + 0.4 * shot_power
    gk_factor = 0.3 * gk_def + 0.2 * gk_phy

    success_prob = max(0.05, min(0.95, base_prob - gk_factor + 0.5))  # clamp between 0.05 and 0.95
    return success_prob


def pass_success(player, teammate, opponent=None, intercept_radius=10.0):
    """
    Calculate probability of a successful pass.
    
    player: the player making the pass
    teammate: the intended recipient
    opponent: nearest opponent (optional)
    intercept_radius: only consider opponent if within this distance
    """
    base_prob = 0.8
    # Base factor: passer's skill vs teammate's dribbling (higher DRI of teammate makes it easier to receive)
    pass_vs_teammate = player.PAS / (player.PAS + (100 - teammate.DRI))

    opponent_factor = 0.0
    if opponent:
        # Calculate distance to opponent
        distance = ((teammate.x - opponent.x)**2 + (teammate.y - opponent.y)**2)**0.5
        if distance < intercept_radius:
            # Closer opponent reduces success
            opponent_factor = min(opponent.PAC / 100, 1.0) * (1 - distance / intercept_radius)

    prob = base_prob * pass_vs_teammate * (1 - opponent_factor)
    return np.clip(prob, 0, 1)



def tackle_success(defender, attacker):
    return (defender.DEF * 0.7 + defender.PHY * 0.3) / ((defender.DEF * 0.7 + defender.PHY * 0.3) + 
                (attacker.DRI * 0.7 + attacker.PHY * 0.3))
