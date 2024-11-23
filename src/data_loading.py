import csv
from datetime import datetime
import numpy as np 
from typing import Tuple, List

def load_halo_data(ffa_path: str = 'data/FreeForAll.csv',
                   h2h_path: str = 'data/HeadToHead.csv',
                   dt_str: str = '%d %B %Y %H:%M:%S',
                   dt_lim: str = '06 August 2004 18:13:50') -> Tuple[np.ndarray, List[Tuple]]:
    """Load and preprocess Halo dataset"""
    dt_lim = datetime.strptime(dt_lim, dt_str)
    players = set()
    matches = []
    
    cur_game = -1
    cur_players = []
    cur_scores = []
    
    with open(ffa_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            date = datetime.strptime(row[0], dt_str)
            if date < dt_lim:
                game = int(row[1])
                player = row[4]
                score = int(row[6])
                
                if game == cur_game:
                    cur_players.append(player)
                    cur_scores.append(score)
                else:
                    if cur_game > 0 and np.sum(np.abs(cur_scores)):
                        matches.append((cur_players, cur_scores))
                        players.update(cur_players)
                    
                    cur_game = game
                    cur_players = [player]
                    cur_scores = [score]
            else:
                break
                
    return np.array(list(players)), matches