def get_team_actions(coach, player_agents, game_state, team_id, states=None):
    """
    Returns final actions for a team by combining coach guidance with DQN player agents.
    Goalkeepers are constrained to their own goal area and can wander around naturally.
    """
    coach_suggestions = coach.strategy(game_state)
    final_actions = {}

    # Get states if not provided
    if states is None:
        team = game_state.team1 if team_id == 1 else game_state.team2
        states = [game_state._get_player_state(p) for p in team.players]

    # Define goal area dimensions
    goal_height = 200
    goal_area_depth = 80
    goal_center_y = game_state.height // 2

    for i, player_agent in enumerate(player_agents):
        # Goalkeeper
        if i == 0:
            if team_id == 1:  # left goal
                x_min, x_max = 5, goal_area_depth
            else:  # right goal
                x_min, x_max = game_state.width - goal_area_depth, game_state.width - 5

            y_min, y_max = goal_center_y - goal_height // 2, goal_center_y + goal_height // 2

            current_x = player_agent.player.x
            current_y = player_agent.player.y
            
            # Get ball position
            ball_x = game_state.ball.x
            ball_y = game_state.ball.y
            
            # Calculate ideal target within goal area
            # Move toward ball but stay in bounds
            target_x = ball_x if x_min <= ball_x <= x_max else (x_min if ball_x < x_min else x_max)
            target_y = max(y_min, min(y_max, ball_y))
                        
            # Calculate movement delta (limited by max_speed)
            max_speed = player_agent.player.max_speed if hasattr(player_agent.player, 'max_speed') else 5
            
            dx = target_x - current_x
            dy = target_y - current_y
            
            # Limit to max_speed
            dx = max(-max_speed, min(max_speed, dx))
            dy = max(-max_speed, min(max_speed, dy))
            
            # Verify next position is still in bounds
            next_x = current_x + dx
            next_y = current_y + dy
            
            # Final clamp (safety check)
            next_x = max(x_min, min(x_max, next_x))
            next_y = max(y_min, min(y_max, next_y))
            
            # Recalculate actual movement
            final_dx = next_x - current_x
            final_dy = next_y - current_y
            
            final_actions[i] = {"move": (final_dx, final_dy), "shoot": False, "pass_to": None}

        # Field players
        else:
            if hasattr(player_agent, 'player_agent') and player_agent.player_agent is not None:
                action_idx = player_agent.player_agent.select_action(states[i])
                player_agent.last_action_idx = action_idx
                action_dict = player_agent.act(game_state, action_int=action_idx)
            else:
                action_dict = coach_suggestions.get(i, {"move": (0, 0), "shoot": False, "pass_to": None})
                player_agent.last_action_idx = 0

            final_actions[i] = action_dict

    return final_actions