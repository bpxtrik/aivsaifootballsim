CONFIG = {
    "field_width": 800,
    "field_height": 500,
    "num_players": 8,   # 4v4
    "players_per_team": 4,
    "episodes": 1000,
    "learning_rate": 0.001,
    "discount_factor": 0.99,
    "epsilon_start": 1.0,
    "epsilon_min": 0.05,
    "epsilon_decay": 0.995,
    "batch_size": 64
}

ACTION_SPACE = {
    0: {"move": (0, -1)},  # up
    1: {"move": (0, 1)},   # down
    2: {"move": (-1, 0)},  # left
    3: {"move": (1, 0)},   # right
    4: {"move": (0, 0), "shoot": True},  # shoot
    5: {"move": (0, 0), "pass_to": 1},   # pass to teammate 1
    6: {"move": (0, 0), "pass_to": 2},   # pass to teammate 2
    7: {"move": (0, 0), "pass_to": 3},   # pass to teammate 3
}
NUM_ACTIONS = len(ACTION_SPACE)