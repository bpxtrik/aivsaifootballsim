import pygame
from environment.utils import resolve_circle_wall_collision

class Ball:
    def __init__(self, x, y, radius=10, color=(255, 255, 255)):
        self.x = x
        self.y = y
        self.vx = 0
        self.vy = 0
        self.radius = radius
        self.color = color
        self.mass = 0.45  # approx kg (scaled)
        self.restitution = 0.8
        self.friction = 0.99  # rolling resistance per tick
        self.pickup_cooldown = 0  # frames until possession allowed

        # New attributes for possession tracking
        self.last_owner = None   # Player who last had the ball
        self.has_possessor = False
        self.kick_type = None    # "shoot", "pass", etc.
        self.last_kicker = None  # Player who last kicked/shooted

    def reset(self, x, y):
        self.x, self.y = x, y
        self.vx, self.vy = 0, 0
        self.pickup_cooldown = 0
        self.last_owner = None
        self.has_possessor = False
        self.kick_type = None
        self.last_kicker = None

    def apply_impulse(self, player, direction, power, kick_type="pass"):
        """
        Apply an impulse to the ball based on direction, power, and kick type.
        """
        dx, dy = direction
        strength = power

        if kick_type == "shoot":
            strength *= 1.5  # more force for shots
        elif kick_type == "dribble":
            strength *= 0.3  # gentle control touches

        self.vx += dx * strength / (self.mass if self.mass > 0 else 1)
        self.vy += dy * strength / (self.mass if self.mass > 0 else 1)

        self.pickup_cooldown = 10
        self.last_owner = player



    def update(self, width, height, goal_opening=None):
        self.x += self.vx
        self.y += self.vy

        # friction
        self.vx *= self.friction
        self.vy *= self.friction

        # cooldown decrement
        if self.pickup_cooldown > 0:
            self.pickup_cooldown -= 1

        # wall collisions except in goal opening
        if goal_opening is not None:
            goal_y_top, goal_y_bottom = goal_opening
        else:
            goal_y_top, goal_y_bottom = None, None

        # Left wall: blocked except gap for goal mouth
        if self.x - self.radius < 0:
            if goal_y_top is not None and goal_y_top <= self.y <= goal_y_bottom:
                pass
            else:
                self.x, self.y, self.vx, self.vy = resolve_circle_wall_collision(
                    self.x, self.y, self.vx, self.vy, self.radius, width, height, self.restitution
                )
                return

        # Right wall: special-case similarly
        if self.x + self.radius > width:
            if goal_y_top is not None and goal_y_top <= self.y <= goal_y_bottom:
                pass
            else:
                self.x, self.y, self.vx, self.vy = resolve_circle_wall_collision(
                    self.x, self.y, self.vx, self.vy, self.radius, width, height, self.restitution
                )
                return

        # Top/bottom walls
        self.x, self.y, self.vx, self.vy = resolve_circle_wall_collision(
            self.x, self.y, self.vx, self.vy, self.radius, width, height, self.restitution
        )

    def draw(self, screen):
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.radius)
