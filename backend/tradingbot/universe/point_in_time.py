from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    import datetime as dt
else:
    import datetime as dt  # noqa: TC003

@dataclass(frozen=True)
class Membership:
    symbol: str
    start: dt.date
    end: dt.date  # inclusive

class UniverseProvider:
    """Resolve membership as-of a date (point-in-time)."""
    def __init__(self, memberships: Dict[str, List[Membership]]):
        self.memberships = memberships  # e.g., {"SP500": [Membership(...), ...]}

    def members(self, name: str, on: dt.date) -> List[str]:
        out = []
        for m in self.memberships.get(name, []):
            if m.start <= on <= m.end:
                out.append(m.symbol)
        return sorted(out)