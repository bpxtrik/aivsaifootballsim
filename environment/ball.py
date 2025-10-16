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

    def reset(self, x, y):
        self.x, self.y = x, y
        self.vx, self.vy = 0, 0
        self.pickup_cooldown = 0

    def apply_impulse(self, ix, iy):
        self.vx += ix / (self.mass if self.mass > 0 else 1)
        self.vy += iy / (self.mass if self.mass > 0 else 1)

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
                # allow crossing (goal)
                pass
            else:
                self.x, self.y, self.vx, self.vy = resolve_circle_wall_collision(
                    self.x, self.y, self.vx, self.vy, self.radius, width, height, self.restitution
                )
                return

        # Right wall special-case similarly
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
