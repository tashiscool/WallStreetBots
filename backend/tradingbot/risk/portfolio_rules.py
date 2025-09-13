from __future__ import annotations
from typing import Dict, List
import numpy as np

def sector_cap_check(weights: Dict[str, float], sector_by_symbol: Dict[str, str], cap: float = 0.35) -> bool:
    tot = sum(abs(w) for w in weights.values())
    sector_w = {}
    for sym, w in weights.items():
        sec = sector_by_symbol.get(sym, "UNK")
        sector_w[sec] = sector_w.get(sec, 0.0) + abs(w)
    return all((w / tot) <= cap for w in sector_w.values())

def simple_corr_guard(returns: np.ndarray, threshold: float = 0.9) -> bool:
    """
    returns: shape [T, N]
    True if max pairwise correlation <= threshold.
    """
    if returns.shape[1] < 2: return True
    c = np.corrcoef(returns.T)
    upper = c[np.triu_indices_from(c, k=1)]
    return float(np.nanmax(upper)) <= threshold