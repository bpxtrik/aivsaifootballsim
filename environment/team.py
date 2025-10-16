from environment.player import Player

class Team:
    def __init__(self, team_id, config, side, players):
        self.team_id = team_id
        self.players = [Player(0,0,team_id, stats=players[0]), Player(0,0,team_id, stats=players[1]),
                        Player(0,0,team_id, stats=players[2]), Player(0,0,team_id, stats=players[3])]
        self.color = (0, 0, 255) if team_id == 1 else (255, 0, 0)
        self.side = side
        self.config = config

        # create players (GK + 3 field players)
        self._spawn_default()

    def reset(self):
        self._spawn_default()
            

    def update(self, actions, ball):
        width = self.config["field_width"]
        height = self.config["field_height"]
        for i, player in enumerate(self.players):
            if i in actions:
                player.move(actions[i], (width, height))
            else:
                # idle damping step
                player.move((0.0, 0.0), (width, height))

    def draw(self, screen):
        for p in self.players:
            p.draw(screen)

    def _spawn_default(self):
        self.players = []
        roles = ["GK", "DEF", "MID", "ATT"]

        if self.side == "left":
            pos = {
                "GK": (50, self.config["field_height"] // 2),
                "DEF": (200, self.config["field_height"] // 3),
                "MID": (200, self.config["field_height"] // 2),
                "ATT": (200, 2 * self.config["field_height"] // 3),
            }
        else:
            pos = {
                "GK": (self.config["field_width"] - 50, self.config["field_height"] // 2),
                "DEF": (self.config["field_width"] - 200, self.config["field_height"] // 3),
                "MID": (self.config["field_width"] - 200, self.config["field_height"] // 2),
                "ATT": (self.config["field_width"] - 200, 2 * self.config["field_height"] // 3),
            }

        for role in roles:
            x, y = pos[role]
            self.players.append(Player(x, y, self.team_id, color=self.color))
