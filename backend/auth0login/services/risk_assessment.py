"""
Risk Assessment Service

Manages the risk questionnaire flow, scoring, and profile determination.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from django.contrib.auth.models import User
from django.utils import timezone

logger = logging.getLogger(__name__)


@dataclass
class QuestionOption:
    """A single answer option for a question."""
    value: str
    label: str
    score: int  # 1-5 scale


@dataclass
class Question:
    """A risk assessment question."""
    id: str
    text: str
    options: list[QuestionOption]
    weight: float = 1.0  # Weight for scoring


@dataclass
class QuestionnaireResult:
    """Result of questionnaire scoring."""
    total_score: int
    max_possible_score: int
    recommended_profile: str
    profile_explanation: str
    score_breakdown: dict


# Define the questionnaire questions
RISK_QUESTIONS = [
    Question(
        id="loss_reaction",
        text="If your portfolio dropped 15% in a week, you would:",
        options=[
            QuestionOption("sell_all", "Sell everything to prevent further losses", 1),
            QuestionOption("sell_some", "Sell some positions to reduce exposure", 2),
            QuestionOption("hold", "Hold and wait for recovery", 3),
            QuestionOption("buy_more", "Buy more at the lower prices", 5),
        ],
        weight=1.5,  # Higher weight - key indicator
    ),
    Question(
        id="timeline",
        text="Your investment timeline is:",
        options=[
            QuestionOption("less_1_year", "Less than 1 year", 1),
            QuestionOption("1_3_years", "1-3 years", 2),
            QuestionOption("3_5_years", "3-5 years", 4),
            QuestionOption("5_plus_years", "More than 5 years", 5),
        ],
        weight=1.2,
    ),
    Question(
        id="savings_portion",
        text="What portion of your total savings is this investment?",
        options=[
            QuestionOption("emergency", "My emergency fund / significant portion", 1),
            QuestionOption("significant", "A significant amount, but not everything", 2),
            QuestionOption("discretionary", "Discretionary savings I can afford to lose some of", 4),
            QuestionOption("play_money", "Play money - I'm okay if I lose it all", 5),
        ],
        weight=1.3,  # Important risk factor
    ),
    Question(
        id="experience",
        text="Your experience with trading:",
        options=[
            QuestionOption("none", "None - I'm completely new", 1),
            QuestionOption("stocks_only", "Stocks only - basic buy and hold", 2),
            QuestionOption("options", "Some options trading experience", 4),
            QuestionOption("active", "Active trader with options experience", 5),
        ],
        weight=1.0,
    ),
    Question(
        id="holding_period",
        text="Your preferred holding period for trades:",
        options=[
            QuestionOption("minutes", "Minutes to hours (day trading)", 5),
            QuestionOption("days", "Days (swing trading)", 4),
            QuestionOption("weeks", "Weeks", 3),
            QuestionOption("months", "Months or longer", 2),
        ],
        weight=0.8,
    ),
    Question(
        id="monitoring",
        text="How often will you check the dashboard?",
        options=[
            QuestionOption("constantly", "Constantly throughout the day", 5),
            QuestionOption("daily", "Once or twice daily", 3),
            QuestionOption("weekly", "A few times per week", 2),
            QuestionOption("rarely", "Weekly or less", 1),
        ],
        weight=0.7,
    ),
]


# Profile thresholds based on weighted score
# Min possible = 6 (all 1s, no weights), Max possible depends on weights
# With weights: approximately 6-14 Conservative, 15-22 Moderate, 23+ Aggressive
PROFILE_THRESHOLDS = {
    'conservative': (0, 14),
    'moderate': (15, 22),
    'aggressive': (23, 100),
}


PROFILE_EXPLANATIONS = {
    'conservative': (
        "Your responses indicate a preference for capital preservation over aggressive growth. "
        "We recommend strategies with lower volatility, defined risk, and longer time horizons. "
        "This includes The Wheel strategy, LEAPS, and Index tracking."
    ),
    'moderate': (
        "You show a balanced approach to risk and reward. "
        "We recommend a mix of income-generating strategies with some growth opportunities. "
        "This includes Momentum plays, Earnings Protection, and Iron Condors."
    ),
    'aggressive': (
        "Your responses suggest comfort with higher volatility and shorter timeframes. "
        "You may be suited for strategies with higher risk/reward profiles. "
        "This includes WSB Dip Bot, Lotto Scanner, and Straddle plays."
    ),
}


class RiskAssessmentService:
    """Service for managing risk assessment questionnaires."""

    def __init__(self):
        self.questions = RISK_QUESTIONS

    def get_questions(self) -> list[dict]:
        """Get all questions in serializable format."""
        return [
            {
                'id': q.id,
                'text': q.text,
                'options': [
                    {'value': opt.value, 'label': opt.label}
                    for opt in q.options
                ],
            }
            for q in self.questions
        ]

    def calculate_score(self, responses: dict[str, str]) -> QuestionnaireResult:
        """
        Calculate the risk score from questionnaire responses.

        Args:
            responses: Dict mapping question_id to selected option value

        Returns:
            QuestionnaireResult with score and recommended profile
        """
        total_weighted_score = 0.0
        max_weighted_score = 0.0
        score_breakdown = {}

        for question in self.questions:
            max_weighted_score += 5 * question.weight  # Max score per question is 5

            selected_value = responses.get(question.id)
            if not selected_value:
                continue

            # Find the score for the selected option
            for option in question.options:
                if option.value == selected_value:
                    weighted_score = option.score * question.weight
                    total_weighted_score += weighted_score
                    score_breakdown[question.id] = {
                        'answer': selected_value,
                        'raw_score': option.score,
                        'weight': question.weight,
                        'weighted_score': weighted_score,
                    }
                    break

        # Round to integer for simplicity
        final_score = int(round(total_weighted_score))

        # Determine profile
        recommended_profile = 'moderate'  # default
        for profile, (min_score, max_score) in PROFILE_THRESHOLDS.items():
            if min_score <= final_score <= max_score:
                recommended_profile = profile
                break

        return QuestionnaireResult(
            total_score=final_score,
            max_possible_score=int(round(max_weighted_score)),
            recommended_profile=recommended_profile,
            profile_explanation=PROFILE_EXPLANATIONS[recommended_profile],
            score_breakdown=score_breakdown,
        )

    def submit_assessment(
        self,
        user: User,
        responses: dict[str, str],
        selected_profile: Optional[str] = None,
        override_acknowledged: bool = False,
    ) -> dict[str, Any]:
        """
        Submit and save a risk assessment.

        Args:
            user: The user completing the assessment
            responses: Dict mapping question_id to selected option value
            selected_profile: Optional override profile if user wants different
            override_acknowledged: Whether user acknowledged override warning

        Returns:
            Dict with assessment result and saved assessment ID
        """
        from backend.auth0login.models import RiskAssessment

        # Calculate score
        result = self.calculate_score(responses)

        # Build responses with scores for storage
        responses_with_scores = {}
        for q_id, answer_value in responses.items():
            responses_with_scores[q_id] = {
                'answer': answer_value,
                'score': result.score_breakdown.get(q_id, {}).get('raw_score', 0),
            }

        # Determine if there's an override
        effective_profile = selected_profile if selected_profile else result.recommended_profile
        is_override = (
            selected_profile is not None
            and selected_profile != result.recommended_profile
        )

        # Create or update assessment
        assessment = RiskAssessment.objects.create(
            user=user,
            responses=responses_with_scores,
            calculated_score=result.total_score,
            recommended_profile=result.recommended_profile,
            selected_profile=selected_profile if is_override else None,
            profile_override=is_override,
            override_acknowledged=override_acknowledged if is_override else False,
            completed_at=timezone.now(),
        )

        return {
            'status': 'success',
            'assessment_id': assessment.id,
            'total_score': result.total_score,
            'max_score': result.max_possible_score,
            'recommended_profile': result.recommended_profile,
            'selected_profile': effective_profile,
            'profile_explanation': result.profile_explanation,
            'is_override': is_override,
            'score_breakdown': result.score_breakdown,
        }

    def get_user_assessment(self, user: User) -> Optional[dict]:
        """
        Get the latest assessment for a user.

        Returns:
            Dict with assessment data or None if not found
        """
        from backend.auth0login.models import RiskAssessment

        assessment = RiskAssessment.get_latest_for_user(user)
        if not assessment:
            return None

        return {
            'id': assessment.id,
            'responses': assessment.responses,
            'total_score': assessment.calculated_score,
            'recommended_profile': assessment.recommended_profile,
            'selected_profile': assessment.selected_profile,
            'effective_profile': assessment.effective_profile,
            'profile_override': assessment.profile_override,
            'override_acknowledged': assessment.override_acknowledged,
            'profile_explanation': PROFILE_EXPLANATIONS.get(
                assessment.effective_profile, ''
            ),
            'completed_at': assessment.completed_at.isoformat() if assessment.completed_at else None,
            'version': assessment.version,
        }

    def check_profile_override_warning(
        self,
        recommended_profile: str,
        selected_profile: str,
    ) -> Optional[str]:
        """
        Generate a warning message if user selects a riskier profile than recommended.

        Returns:
            Warning message or None if no warning needed
        """
        profile_risk_order = ['conservative', 'moderate', 'aggressive']

        recommended_idx = profile_risk_order.index(recommended_profile)
        selected_idx = profile_risk_order.index(selected_profile)

        if selected_idx > recommended_idx:
            risk_diff = selected_idx - recommended_idx
            if risk_diff >= 2:  # Conservative -> Aggressive
                return (
                    "You're selecting a significantly more aggressive profile than we recommend "
                    "based on your answers. This means higher potential for both gains AND losses. "
                    "Are you sure you understand the risks?"
                )
            else:  # One step more aggressive
                return (
                    f"You're selecting a more aggressive profile ({selected_profile}) than "
                    f"recommended ({recommended_profile}). This increases your risk exposure. "
                    "Please confirm you're comfortable with potential larger losses."
                )

        return None


# Singleton instance
risk_assessment_service = RiskAssessmentService()
