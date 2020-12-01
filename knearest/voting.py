from typing import List
from collections import Counter

def majority_vote(labels: List[str]) -> str:
    """Assumes that labels are ordered from nearest to farthest"""
    vote_counts = Counter(labels)
    winner, winner_count = vote_counts.most_common(1)[0]
    num_winners = len([count for count in vote_counts.values() if count == winner_count])
    if num_winners == 1:
        return winner
    else:
        return majority_vote(labels[:-1]) #Try again without the farthest

assert majority_vote(['a', 'b', 'c', 'b', 'a']) == 'b'