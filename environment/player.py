import pygame
from environment.utils import resolve_circle_wall_collision

class Player:
    def __init__(self, x, y, team_id, radius=15, color=(0, 0, 255)):
        self.x = x
        self.y = y
        self.team_id = team_id
        self.radius = radius
        self.color = color
        self.vx = 0.0
        self.vy = 0.0
        self.max_speed = 4.0
        self.acceleration = 0.6
        self.damping = 0.9
        self.mass = 70.0
        self.has_ball = False
        self.kick_power = 12.0

    def move(self, action, bounds):
        """action: dict or tuple; if dict expects {'move':(dx,dy), 'shoot':bool}; bounds=(w,h)"""
        width, height = bounds
        if isinstance(action, dict):
            ax, ay = action.get('move', (0.0, 0.0))
        else:
            ax, ay = action
        self.vx += ax * self.acceleration
        self.vy += ay * self.acceleration

        # limit speed
        speed_sq = self.vx * self.vx + self.vy * self.vy
        max_speed_sq = self.max_speed * self.max_speed
        if speed_sq > max_speed_sq:
            scale = self.max_speed / (speed_sq ** 0.5)
            self.vx *= scale
            self.vy *= scale

        # integrate
        self.x += self.vx
        self.y += self.vy

        # damping
        self.vx *= self.damping
        self.vy *= self.damping

        # keep inside box
        self.x, self.y, self.vx, self.vy = resolve_circle_wall_collision(
            self.x, self.y, self.vx, self.vy, self.radius, width, height, restitution=0.4
        )

    def draw(self, screen):
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.radius)
        if self.has_ball:
            pygame.draw.circle(screen, (255, 255, 0), (int(self.x), int(self.y)), self.radius + 3, 2)
