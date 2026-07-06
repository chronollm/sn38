from dataclasses import dataclass, asdict


@dataclass
class RoundResults:
    stage1: dict
    qualified: list
    stage2: dict
    final_scores: dict
    winner: int | None
    weights: dict

    @classmethod
    def no_qualified(cls, leak_scores, owner_uid):
        return cls(
            stage1={str(uid): {"eval_score": s} for uid, s in leak_scores.items()},
            qualified=[],
            stage2={},
            final_scores={},
            winner=None,
            weights={str(owner_uid): 1.0},
        )

    @classmethod
    def with_winner(cls, leak_scores, qualified, win_rates, final_scores, winner, uids, weights):
        stage2 = {}
        if win_rates is not None:
            total = max(1, len(qualified) - 1)
            stage2 = {str(uid): {"wins": int(win_rates[uid] * total), "win_rate": float(win_rates[uid])} for uid, _ in qualified}
        return cls(
            stage1={str(uid): {"eval_score": s} for uid, s in leak_scores.items()},
            qualified=[uid for uid, _ in qualified],
            stage2=stage2,
            final_scores={str(uid): float(final_scores[uid]) for uid, _ in qualified},
            winner=winner,
            weights={str(u): w for u, w in zip(uids, weights)},
        )

    def to_dict(self):
        return asdict(self)
