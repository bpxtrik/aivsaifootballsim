import random
import math


class CoachAgent:
    def __init__(self, team, formation="1-1-1"):
        self.team = team
        self.formation = formation
        self.positions = self._get_formation_positions()
        
    def _get_formation_positions(self):
        """Define base positions for different formations (relative, 0-1 scale)"""
        # For 4v4: 1 GK + 3 field players
        formations = {
            "1-1-1": [
                # GK handled separately, these are the 3 field players
                (0.3, 0.5),   # Defender (stays back)
                (0.5, 0.5),   # Midfielder (central)
                (0.7, 0.5),   # Forward (attacks)
            ],
            "1-2": [
                # More defensive: 1 defender, 2 attackers
                (0.25, 0.5),  # Defender (center back)
                (0.6, 0.3),   # Attacker 1 (left wing)
                (0.6, 0.7),   # Attacker 2 (right wing)
            ],
            "2-1": [
                # More attacking: 2 defenders, 1 striker
                (0.3, 0.35),  # Defender 1
                (0.3, 0.65),  # Defender 2
                (0.7, 0.5),   # Striker (center forward)
            ]
        }
        return formations.get(self.formation, formations["1-1-1"])

    def strategy(self, game_state):
        """
        Returns tactical actions based on game state and formation.
        """
        actions = {}
        ball_x = game_state.ball.x
        ball_y = game_state.ball.y
        
        # Determine if team has possession
        team_has_ball = any(p.has_ball for p in self.team.players)
        closest_to_ball = self._find_closest_player(game_state)
        
        # Get field dimensions
        field_width = game_state.width
        field_height = game_state.height
        
        # Determine if we're team 1 (left) or team 2 (right)
        is_left_team = self.team.players[0].team_id == 1
        
        for i, player in enumerate(self.team.players):
            if i == 0:  # goalkeeper handled separately
                continue
                
            # Get player's formation position
            base_x_rel, base_y_rel = self.positions[i - 1]
            
            # Mirror for right team
            if not is_left_team:
                base_x_rel = 1.0 - base_x_rel
            
            base_x = base_x_rel * field_width
            base_y = base_y_rel * field_height
            
            # Decide behavior based on role and game state
            if i == closest_to_ball:
                # Closest player presses the ball
                action = self._press_ball(player, ball_x, ball_y, game_state)
            elif team_has_ball:
                # Supporting attack - move forward but maintain shape
                action = self._support_attack(player, base_x, base_y, ball_x, ball_y, 
                                              game_state, is_left_team)
            else:
                # Defensive positioning
                action = self._defend(player, base_x, base_y, ball_x, ball_y, 
                                     game_state, is_left_team)
            
            actions[i] = action
        
        return actions
    
    def _find_closest_player(self, game_state):
        """Find index of player closest to ball"""
        min_dist = float('inf')
        closest_idx = 1  # Start from 1 (skip goalkeeper)
        
        for i, player in enumerate(self.team.players):
            if i == 0:  # skip goalkeeper
                continue
            dist = math.hypot(game_state.ball.x - player.x, 
                            game_state.ball.y - player.y)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
        
        return closest_idx
    
    def _press_ball(self, player, ball_x, ball_y, game_state):
        """Aggressively move toward ball"""
        dx = ball_x - player.x
        dy = ball_y - player.y
        dist = math.hypot(dx, dy)
        
        if dist > 0:
            vx = (dx / dist) * player.max_speed
            vy = (dy / dist) * player.max_speed
        else:
            vx, vy = 0, 0
        
        # Shoot if close to opponent goal and has ball
        shoot = False
        if player.has_ball:
            is_left_team = player.team_id == 1
            goal_x = game_state.width if is_left_team else 0
            dist_to_goal = abs(player.x - goal_x)
            if dist_to_goal < 150:  # Within shooting range
                shoot = True
        
        # Pass if has ball and teammate is better positioned
        pass_to = None
        if player.has_ball and not shoot:
            pass_to = self._find_best_pass_target(player, game_state)
        
        return {"move": (vx, vy), "shoot": shoot, "pass_to": pass_to}
    
    def _support_attack(self, player, base_x, base_y, ball_x, ball_y, game_state, is_left_team):
        """Move forward to support attack while maintaining formation"""
        # Blend between formation position and moving toward attack
        attack_zone_x = game_state.width * 0.75 if is_left_team else game_state.width * 0.25
        
        # Move toward attack zone but stay near formation position
        target_x = base_x * 0.4 + attack_zone_x * 0.6
        target_y = base_y * 0.6 + ball_y * 0.4
        
        dx = target_x - player.x
        dy = target_y - player.y
        dist = math.hypot(dx, dy)
        
        if dist > 0:
            vx = (dx / dist) * player.max_speed * 0.7  # Move at 70% speed
            vy = (dy / dist) * player.max_speed * 0.7
        else:
            vx, vy = 0, 0
        
        return {"move": (vx, vy), "shoot": False, "pass_to": None}
    
    def _defend(self, player, base_x, base_y, ball_x, ball_y, game_state, is_left_team):
        """Defensive positioning - stay between ball and goal"""
        own_goal_x = 0 if is_left_team else game_state.width
        
        # Position between ball and goal
        defensive_x = (base_x + ball_x + own_goal_x) / 3
        defensive_y = (base_y + ball_y) / 2
        
        dx = defensive_x - player.x
        dy = defensive_y - player.y
        dist = math.hypot(dx, dy)
        
        if dist > 0:
            vx = (dx / dist) * player.max_speed * 0.5  # Move at 50% speed
            vy = (dy / dist) * player.max_speed * 0.5
        else:
            vx, vy = 0, 0
        
        return {"move": (vx, vy), "shoot": False, "pass_to": None}
    
    def _find_best_pass_target(self, player, game_state):
        """Find best teammate to pass to (forward and open)"""
        is_left_team = player.team_id == 1
        best_score = -1
        best_target = None
        
        for i, teammate in enumerate(self.team.players):
            if i == 0 or teammate == player:  # Skip GK and self
                continue
            
            # Score based on: forward position, distance from opponents
            forward_score = teammate.x if is_left_team else (game_state.width - teammate.x)
            
            # Simple openness check - distance to nearest opponent
            min_opp_dist = float('inf')
            opponent_team = game_state.team2 if is_left_team else game_state.team1
            for opp in opponent_team.players:
                dist = math.hypot(teammate.x - opp.x, teammate.y - opp.y)
                min_opp_dist = min(min_opp_dist, dist)
            
            # Combined score
            score = forward_score + min_opp_dist * 0.5
            
            if score > best_score:
                best_score = score
                best_target = i
        
        return best_target