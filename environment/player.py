import pygame
from environment.utils import resolve_circle_wall_collision

class Player:
    def __init__(self, x, y, team_id, radius=15, color=(0, 0, 255), stats=None):
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
        
        self.PAC = None
        self.SHO = None
        self.PAS = None
        self.DRI = None
        self.DEF = None
        self.PHY = None
        
        if stats:
            self.load_stats(stats)
            
    def load_stats(self, stats): 
        self.PAC = stats.get("PAC")
        self.SHO = stats.get("SHO")
        self.PAS = stats.get("PAS")
        self.DRI = stats.get("DRI")
        self.DEF = stats.get("DEF")
        self.PHY = stats.get("PHY")
        
        if self.PAC:
            self.max_speed = 2.0 + (self.PAC / 100) * 4.0
            self.acceleration = 0.3 + (self.PAC / 100) * 0.5

    def move(self, action, bounds):
        width, height = bounds

        # Extract move vector
        if isinstance(action, dict):
            ax, ay = action.get('move', (0.0, 0.0))
        else:
            ax, ay = action

        # Normalize move vector for consistent acceleration
        mag = (ax**2 + ay**2)**0.5
        if mag > 1e-5:
            ax /= mag
            ay /= mag
        else:
            ax, ay = 0.0, 0.0

        # Smooth velocity update
        self.vx += ax * self.acceleration
        self.vy += ay * self.acceleration

        # Limit speed
        speed_sq = self.vx**2 + self.vy**2
        max_speed_sq = self.max_speed**2
        if speed_sq > max_speed_sq:
            scale = self.max_speed / (speed_sq**0.5)
            self.vx *= scale
            self.vy *= scale

        # Apply velocity to position
        self.x += self.vx
        self.y += self.vy

        # Apply damping for smooth movement
        self.vx *= self.damping
        self.vy *= self.damping

        # Threshold small velocities to zero to prevent jitter
        if abs(self.vx) < 0.01: self.vx = 0.0
        if abs(self.vy) < 0.01: self.vy = 0.0

        # Keep inside field
        self.x, self.y, self.vx, self.vy = resolve_circle_wall_collision(
            self.x, self.y, self.vx, self.vy, self.radius, width, height, restitution=0.4
        )




    def draw(self, screen):
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.radius)
        if self.has_ball:
            pygame.draw.circle(screen, (255, 255, 0), (int(self.x), int(self.y)), self.radius + 3, 2)

    def __str__(self):
        return (f"{self.PAC}-{self.SHO}-{self.PAS}-{self.DRI}-{self.DEF}-{self.PHY}")