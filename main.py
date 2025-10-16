from environment.game import FootballGame
# from training.trainer import Trainer
from config import CONFIG
import pygame

def main():
    # Create environment
    env = FootballGame(config=CONFIG)
    state = env.reset()

    running = True
    while running:
        for event in pygame.event.get():
            if (event.type) == pygame.QUIT:
                running = False

        # Simple keyboard control for blue ATT (index 3) and red ATT (index 3)
        keys = pygame.key.get_pressed()
        ax_b = 0.0
        ay_b = 0.0
        shoot_b = False
        if keys[pygame.K_w]:
            ay_b -= 1.0
        if keys[pygame.K_s]:
            ay_b += 1.0
        if keys[pygame.K_a]:
            ax_b -= 1.0
        if keys[pygame.K_d]:
            ax_b += 1.0
        if keys[pygame.K_SPACE]:
            shoot_b = True

        ax_r = 0.0
        ay_r = 0.0
        shoot_r = False
        if keys[pygame.K_UP]:
            ay_r -= 1.0
        if keys[pygame.K_DOWN]:
            ay_r += 1.0
        if keys[pygame.K_LEFT]:
            ax_r -= 1.0
        if keys[pygame.K_RIGHT]:
            ax_r += 1.0
        if keys[pygame.K_RCTRL] or keys[pygame.K_RSHIFT]:
            shoot_r = True

        actions_team1 = {3: {"move": (ax_b, ay_b), "shoot": shoot_b}}
        actions_team2 = {3: {"move": (ax_r, ay_r), "shoot": shoot_r}}

        state, reward, done = env.step(actions_team1, actions_team2)

        env.render()

    # Create trainer
    # trainer = Trainer(env, config=CONFIG)

    # Start training
    # trainer.train(num_episodes=CONFIG["episodes"])

if __name__ == "__main__":

    main()
