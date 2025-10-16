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
    # Movement without ball (or dribbling with ball)
    0: {"move": (0, -1)},   # up
    1: {"move": (0, 1)},    # down
    2: {"move": (-1, 0)},   # left
    3: {"move": (1, 0)},    # right
    # Diagonal directions
    4: {"move": (-0.707, -0.707)},  # up-left
    5: {"move": (0.707, -0.707)},   # up-right
    6: {"move": (-0.707, 0.707)},   # down-left
    7: {"move": (0.707, 0.707)},    # down-right
    # Special actions (only when in possession)
    8: {"move": (0, 0), "shoot": True},  # shoot
    9: {"move": (0, 0), "pass_to": 1},   # pass to teammate 1
    10: {"move": (0, 0), "pass_to": 2},  # pass to teammate 2
    11: {"move": (0, 0), "pass_to": 3},  # pass to teammate 3
    12: {"move": (0, 0), "dribble": True},  # dribble (keep ball and move)
}
NUM_ACTIONS = len(ACTION_SPACE)