import pandas as pd

def load_tournament_players(csv_path, nation1, nation2):
    df = pd.read_csv(csv_path)

    # Keep only rows with card_type "common" or "rare"
    df = df[df["card_type"].isin(["common", "rare"])]
    df = df[df["nation"].isin([nation1, nation2])]

    team1 = [None, None, None, None]
    team2 = [None, None, None, None]
    for _, row in df.iterrows():
        nation = row["nation"].strip().lower()
        pos = row["position"]

        if nation == nation1.strip().lower():  # team1
            if pos == "CB" and team1[1] is None:
                team1[1] = pack_stats(row)
            elif pos == "CM" and team1[2] is None:
                team1[2] = pack_stats(row)
            elif pos == "ST" and team1[3] is None:
                team1[3] = pack_stats(row)

        elif nation == nation2.strip().lower():  # team2
            if pos == "CB" and team2[1] is None:
                team2[1] = pack_stats(row)
            elif pos == "CM" and team2[2] is None:
                team2[2] = pack_stats(row)
            elif pos == "ST" and team2[3] is None:
                team2[3] = pack_stats(row)

    default_gk = {
        "name": "Default GK",
        "rating": 80,
        "card_type": "common",
        "position": "GK",
        "PAC": 50,
        "SHO": 40,
        "PAS": 60,
        "DRI": 50,
        "DEF": 80,
        "PHY": 85
    }

    team1[0] = default_gk
    team2[0] = default_gk
            
    return {"team1" : team1, "team2": team2}

import random

def load_training_players(csv_path, team_size=4):
    df = pd.read_csv(csv_path)

    # Keep only "common" or "rare"
    df = df[df["card_type"].isin(["common", "rare"])]

    teams = {"team1": [], "team2": []}

    for team_key in ["team1", "team2"]:
        # Assign GK
        gk_candidates = df[df["position"] == "GK"]
        if not gk_candidates.empty:
            gk_row = gk_candidates.sample(1).iloc[0]
            teams[team_key].append(pack_stats(gk_row))
        else:
            teams[team_key].append({
                "name": "Default GK",
                "rating": 80,
                "card_type": "common",
                "position": "GK",
                "PAC": 50,
                "SHO": 40,
                "PAS": 60,
                "DRI": 50,
                "DEF": 80,
                "PHY": 85
            })

        # Assign field players: CB, CM, ST
        for pos in ["CB", "CM", "ST"]:
            pos_candidates = df[df["position"] == pos]
            if not pos_candidates.empty:
                row = pos_candidates.sample(1).iloc[0]
                teams[team_key].append(pack_stats(row))
            else:
                teams[team_key].append({
                    "name": f"Default {pos}",
                    "rating": 75,
                    "card_type": "common",
                    "position": pos,
                    "PAC": 70,
                    "SHO": 70,
                    "PAS": 70,
                    "DRI": 70,
                    "DEF": 70,
                    "PHY": 70
                })

    return teams

import csv
from typing import List, Dict

def load_top15_rankings(filename: str, date_filter: str = "2024-06-20") -> List[Dict]:
    """
    Load CSV and return only rows that have rank_date equal to date_filter.
    
    Returns a list of dictionaries with keys:
    'rank', 'country_full', 'country_abrv', 'total_points', 
    'previous_points', 'rank_change', 'confederation', 'rank_date'
    """
    filtered_rows = []
    
    with open(filename, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row["rank_date"] == date_filter:
                try:
                    filtered_rows.append({
                        "rank": float(row["rank"]),
                        "country_full": row["country_full"]
                    })
                except:
                    continue
    
        top15 = sorted(filtered_rows, key=lambda x: x["rank"])[:15]
        return top15


def pack_stats(row):
    return {
            "name": row["name"],
            "nation": row["nation"],
            "rating": row["rating"],
            "card_type": row["card_type"],
            "position": row["position"],
            "PAC": row["PAC"],
            "SHO": row["SHO"],
            "PAS": row["PAS"],
            "DRI": row["DRI"],
            "DEF": row["DEF"],
            "PHY": row["PHY"]
        }