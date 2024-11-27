from typing import List, Dict
import numpy as np
import csv

def evaluate_head_to_head(rankings: np.ndarray, 
                         universe: np.ndarray,
                         h2h_path: str = 'data/HeadToHead.csv',
                         dt_str: str = '%d %B %Y %H:%M:%S') -> Dict:
    """Evaluate rankings on head-to-head matches"""
    results = []
    cur_game = -1
    cur_players = []
    cur_scores = []
    
    with open(h2h_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            game = int(row[1])
            player = row[4]
            score = int(row[6])
            
            if game == cur_game:
                cur_players.append(player)
                cur_scores.append(score)
            else:
                if cur_game > 0 and np.sum(np.abs(cur_scores)) > 0:
                    result = evaluate_match(cur_players, cur_scores, 
                                         universe, rankings)
                    if result is not None:
                        results.append(result)
                
                cur_game = game
                cur_players = [player]
                cur_scores = [score]
                
    return {
        'accuracy': np.mean(results),
        'total_matches': len(results)
    }

def evaluate_match(players: List, 
                  scores: List,
                  universe: np.ndarray, 
                  rankings: np.ndarray) -> bool:
    """Evaluate single head-to-head match"""
    players_ranked = [p for p in players if p in universe]
    if len(players_ranked) == 2:
        scores_ranked = [scores[players.index(p)] for p in players_ranked]
        ranks_ranked = [rankings[np.where(universe == p)[0][0]] 
                       for p in players_ranked]
        
        if scores_ranked[0] != scores_ranked[1]:
            return (np.argsort(scores_ranked) == np.argsort(ranks_ranked)).all()
    return None