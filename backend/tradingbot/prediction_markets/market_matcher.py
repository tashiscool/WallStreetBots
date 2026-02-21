"""
Cross-Platform Market Matcher.

Synthesized from:
- ImMike/polymarket-arbitrage: Domain-specific text matching
- CarlosIbCu: Time zone-aware market URL generation
- antevorta: Entity-based similarity matching

Matches equivalent markets across Polymarket and Kalshi.
"""

from dataclasses import dataclass, field
from datetime import datetime, date
from decimal import Decimal
from enum import Enum
from typing import ClassVar, Dict, List, Optional, Set, Tuple
import re
import logging
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)


class MarketCategory(Enum):
    """Categories for market classification."""
    POLITICS = "politics"
    SPORTS = "sports"
    CRYPTO = "crypto"
    ECONOMICS = "economics"
    ENTERTAINMENT = "entertainment"
    SCIENCE = "science"
    WEATHER = "weather"
    OTHER = "other"


@dataclass
class MarketInfo:
    """Normalized market information for matching."""
    market_id: str
    platform: str
    title: str
    description: str = ""
    category: MarketCategory = MarketCategory.OTHER
    expiration: Optional[datetime] = None
    tags: List[str] = field(default_factory=list)
    entities: Set[str] = field(default_factory=set)

    # Extracted features
    teams: List[str] = field(default_factory=list)
    people: List[str] = field(default_factory=list)
    dates: List[date] = field(default_factory=list)
    numbers: List[float] = field(default_factory=list)


@dataclass
class MarketPair:
    """Matched pair of markets across platforms."""
    polymarket: MarketInfo
    kalshi: MarketInfo
    similarity_score: float
    match_type: str  # "exact", "fuzzy", "entity", "sports"
    matched_entities: Set[str] = field(default_factory=set)


class EntityExtractor:
    """
    Extracts entities from market titles.

    From ImMike: Domain-specific entity extraction.
    """

    # Common sports teams (NFL, NBA, etc.)
    NFL_TEAMS: ClassVar[set] = {
        "chiefs", "eagles", "bills", "49ers", "cowboys", "ravens",
        "bengals", "dolphins", "lions", "jaguars", "chargers", "jets",
        "saints", "packers", "vikings", "seahawks", "rams", "cardinals",
        "buccaneers", "falcons", "panthers", "giants", "commanders", "bears",
        "browns", "steelers", "raiders", "broncos", "colts", "texans", "titans"
    }

    NBA_TEAMS: ClassVar[set] = {
        "lakers", "celtics", "warriors", "bucks", "nets", "heat",
        "suns", "76ers", "nuggets", "grizzlies", "mavericks", "clippers",
        "hawks", "bulls", "cavaliers", "knicks", "pelicans", "raptors",
        "jazz", "thunder", "timberwolves", "trail blazers", "magic", "pistons",
        "pacers", "wizards", "hornets", "rockets", "spurs", "kings"
    }

    # Politicians and public figures
    POLITICIANS: ClassVar[set] = {
        "biden", "trump", "harris", "desantis", "newsom", "pence",
        "obama", "clinton", "sanders", "warren", "buttigieg", "haley"
    }

    # Cryptocurrencies
    CRYPTO_COINS: ClassVar[set] = {
        "bitcoin", "btc", "ethereum", "eth", "solana", "sol",
        "cardano", "ada", "dogecoin", "doge", "xrp", "ripple"
    }

    def extract_entities(self, text: str) -> Set[str]:
        """Extract all entities from text."""
        text_lower = text.lower()
        entities = set()

        # Sports teams
        for team in self.NFL_TEAMS | self.NBA_TEAMS:
            if team in text_lower:
                entities.add(team)

        # Politicians
        for person in self.POLITICIANS:
            if person in text_lower:
                entities.add(person)

        # Crypto
        for coin in self.CRYPTO_COINS:
            if coin in text_lower:
                entities.add(coin)

        return entities

    def extract_teams(self, text: str) -> List[str]:
        """Extract sports teams from text."""
        text_lower = text.lower()
        teams = []

        for team in self.NFL_TEAMS | self.NBA_TEAMS:
            if team in text_lower:
                teams.append(team)

        return teams

    def extract_numbers(self, text: str) -> List[float]:
        """Extract numbers (prices, percentages, etc.)."""
        # Match numbers with optional $ or % suffix
        pattern = r'\$?([\d,]+(?:\.\d+)?)\%?'
        matches = re.findall(pattern, text)
        return [float(m.replace(',', '')) for m in matches if m]

    def extract_dates(self, text: str) -> List[date]:
        """Extract dates from text."""
        dates = []

        # Pattern: "Month Day" or "Month Day, Year"
        month_pattern = r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2})(?:,?\s+(\d{4}))?'

        for match in re.finditer(month_pattern, text, re.IGNORECASE):
            month_name = match.group(1)
            day = int(match.group(2))
            year = int(match.group(3)) if match.group(3) else datetime.now().year

            month_map = {
                'january': 1, 'february': 2, 'march': 3, 'april': 4,
                'may': 5, 'june': 6, 'july': 7, 'august': 8,
                'september': 9, 'october': 10, 'november': 11, 'december': 12
            }
            month = month_map.get(month_name.lower(), 1)

            try:
                dates.append(date(year, month, day))
            except ValueError:
                pass

        return dates


class MarketCategorizer:
    """
    Categorizes markets for efficient matching.

    From ImMike: Category-based pre-filtering reduces comparisons.
    """

    POLITICS_KEYWORDS: ClassVar[set] = {
        "election", "president", "congress", "senate", "vote",
        "republican", "democrat", "biden", "trump", "poll"
    }

    SPORTS_KEYWORDS: ClassVar[set] = {
        "nfl", "nba", "mlb", "nhl", "game", "match", "win",
        "championship", "playoffs", "super bowl", "world series"
    }

    CRYPTO_KEYWORDS: ClassVar[set] = {
        "bitcoin", "btc", "ethereum", "eth", "crypto", "price",
        "above", "below", "reach"
    }

    ECONOMICS_KEYWORDS: ClassVar[set] = {
        "fed", "interest rate", "inflation", "gdp", "unemployment",
        "recession", "stock", "market"
    }

    def categorize(self, title: str, description: str = "") -> MarketCategory:
        """Determine market category from title/description."""
        text = (title + " " + description).lower()

        # Check each category
        if any(kw in text for kw in self.POLITICS_KEYWORDS):
            return MarketCategory.POLITICS
        if any(kw in text for kw in self.SPORTS_KEYWORDS):
            return MarketCategory.SPORTS
        if any(kw in text for kw in self.CRYPTO_KEYWORDS):
            return MarketCategory.CRYPTO
        if any(kw in text for kw in self.ECONOMICS_KEYWORDS):
            return MarketCategory.ECONOMICS

        return MarketCategory.OTHER


class MarketMatcher:
    """
    Matches equivalent markets across platforms.

    Multi-strategy matching from ImMike:
    1. Category-based pre-filtering
    2. Sports matchup detection (teams + date)
    3. Entity overlap matching
    4. Fuzzy text similarity
    """

    def __init__(
        self,
        min_similarity: float = 0.7,
        require_category_match: bool = True,
    ):
        self._min_similarity = min_similarity
        self._require_category_match = require_category_match
        self._entity_extractor = EntityExtractor()
        self._categorizer = MarketCategorizer()

    def find_matches(
        self,
        polymarket_markets: List[MarketInfo],
        kalshi_markets: List[MarketInfo],
    ) -> List[MarketPair]:
        """
        Find matching markets across platforms.

        From ImMike: Category-based pre-filtering dramatically reduces
        comparisons from N*M to much smaller subsets.
        """
        matches = []

        # Group by category
        poly_by_cat: Dict[MarketCategory, List[MarketInfo]] = {}
        kalshi_by_cat: Dict[MarketCategory, List[MarketInfo]] = {}

        for market in polymarket_markets:
            cat = market.category
            if cat not in poly_by_cat:
                poly_by_cat[cat] = []
            poly_by_cat[cat].append(market)

        for market in kalshi_markets:
            cat = market.category
            if cat not in kalshi_by_cat:
                kalshi_by_cat[cat] = []
            kalshi_by_cat[cat].append(market)

        # Compare within categories
        for category in MarketCategory:
            poly_markets = poly_by_cat.get(category, [])
            kalshi_list = kalshi_by_cat.get(category, [])

            if not poly_markets or not kalshi_list:
                continue

            for poly in poly_markets:
                best_match = self._find_best_match(poly, kalshi_list)
                if best_match:
                    matches.append(best_match)

        return matches

    def _find_best_match(
        self,
        poly: MarketInfo,
        kalshi_markets: List[MarketInfo],
    ) -> Optional[MarketPair]:
        """Find best Kalshi match for a Polymarket market."""
        best_match = None
        best_score = 0.0

        for kalshi in kalshi_markets:
            score, match_type, entities = self._calculate_similarity(poly, kalshi)

            if score > best_score and score >= self._min_similarity:
                best_score = score
                best_match = MarketPair(
                    polymarket=poly,
                    kalshi=kalshi,
                    similarity_score=score,
                    match_type=match_type,
                    matched_entities=entities,
                )

        return best_match

    def _calculate_similarity(
        self,
        poly: MarketInfo,
        kalshi: MarketInfo,
    ) -> Tuple[float, str, Set[str]]:
        """
        Calculate similarity between two markets.

        Returns: (score, match_type, matched_entities)
        """
        # Strategy 1: Sports matchup (teams + date)
        if poly.category == MarketCategory.SPORTS:
            score, entities = self._sports_similarity(poly, kalshi)
            if score > 0.8:
                return score, "sports", entities

        # Strategy 2: Entity overlap
        entity_score, entities = self._entity_similarity(poly, kalshi)
        if entity_score > 0.7:
            return entity_score, "entity", entities

        # Strategy 3: Fuzzy text similarity
        text_score = self._text_similarity(poly.title, kalshi.title)
        if text_score > self._min_similarity:
            return text_score, "fuzzy", set()

        # Strategy 4: Number matching (for crypto/price markets)
        if poly.numbers and kalshi.numbers:
            num_score = self._number_similarity(poly.numbers, kalshi.numbers)
            if num_score > 0.9:
                # Also require some text similarity
                combined = (num_score + text_score) / 2
                if combined > self._min_similarity:
                    return combined, "number", set()

        return 0.0, "", set()

    def _sports_similarity(
        self,
        poly: MarketInfo,
        kalshi: MarketInfo,
    ) -> Tuple[float, Set[str]]:
        """Calculate sports matchup similarity."""
        poly_teams = set(poly.teams)
        kalshi_teams = set(kalshi.teams)

        if not poly_teams or not kalshi_teams:
            return 0.0, set()

        # Teams must match
        common_teams = poly_teams & kalshi_teams
        if len(common_teams) < 2:  # Need at least 2 teams for a matchup
            return 0.0, set()

        # Check date proximity if available
        if poly.dates and kalshi.dates:
            poly_date = poly.dates[0]
            kalshi_date = kalshi.dates[0]
            if poly_date != kalshi_date:
                return 0.0, set()

        # High confidence match
        return 0.95, common_teams

    def _entity_similarity(
        self,
        poly: MarketInfo,
        kalshi: MarketInfo,
    ) -> Tuple[float, Set[str]]:
        """Calculate entity overlap similarity."""
        poly_entities = poly.entities
        kalshi_entities = kalshi.entities

        if not poly_entities or not kalshi_entities:
            return 0.0, set()

        common = poly_entities & kalshi_entities
        total = poly_entities | kalshi_entities

        if not total:
            return 0.0, set()

        jaccard = len(common) / len(total)
        return jaccard, common

    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate fuzzy text similarity."""
        # Normalize text
        t1 = text1.lower().strip()
        t2 = text2.lower().strip()

        # Use SequenceMatcher for ratio
        return SequenceMatcher(None, t1, t2).ratio()

    def _number_similarity(
        self,
        nums1: List[float],
        nums2: List[float],
    ) -> float:
        """Check if key numbers match (prices, percentages)."""
        if not nums1 or not nums2:
            return 0.0

        # Check for exact or very close matches
        for n1 in nums1:
            for n2 in nums2:
                if abs(n1 - n2) < 0.01 * max(n1, n2, 1):  # Within 1%
                    return 1.0

        return 0.0

    def enrich_market_info(self, market: MarketInfo) -> MarketInfo:
        """Enrich market with extracted features."""
        market.entities = self._entity_extractor.extract_entities(market.title)
        market.teams = self._entity_extractor.extract_teams(market.title)
        market.numbers = self._entity_extractor.extract_numbers(market.title)
        market.dates = self._entity_extractor.extract_dates(market.title)
        market.category = self._categorizer.categorize(
            market.title,
            market.description
        )
        return market


class MarketSlugGenerator:
    """
    Generates market slugs/URLs for hourly markets.

    From CarlosIbCu: Time-based market URL generation.
    """

    @staticmethod
    def generate_polymarket_btc_slug(target_time: datetime) -> str:
        """
        Generate Polymarket BTC price market slug.

        Format: bitcoin-up-or-down-{month}-{day}-{hour}{am/pm}-et
        """
        month = target_time.strftime("%B").lower()
        day = target_time.day
        hour = target_time.strftime("%I").lstrip("0")
        period = target_time.strftime("%p").lower()

        return f"bitcoin-up-or-down-{month}-{day}-{hour}{period}-et"

    @staticmethod
    def generate_kalshi_btc_ticker(target_time: datetime) -> str:
        """
        Generate Kalshi BTC price market ticker.

        Format: KXBTC-{YYMMDD}-T{HH}
        """
        date_str = target_time.strftime("%y%m%d")
        hour = target_time.strftime("%H")
        return f"KXBTC-{date_str}-T{hour}"

    @staticmethod
    def get_next_hourly_market_time() -> datetime:
        """Get the next hourly market time."""
        now = datetime.now()
        # Round up to next hour
        next_hour = now.replace(minute=0, second=0, microsecond=0)
        if now.minute > 0 or now.second > 0:
            next_hour = next_hour.replace(hour=next_hour.hour + 1)
        return next_hour


# Global matcher instance
_market_matcher: Optional[MarketMatcher] = None


def get_market_matcher() -> MarketMatcher:
    """Get or create the global market matcher."""
    global _market_matcher
    if _market_matcher is None:
        _market_matcher = MarketMatcher()
    return _market_matcher
