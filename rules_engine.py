"""
Rules Engine for Football Play Analysis.

Based on the official IFAB (International Football Association Board) Laws of the Game.
"""
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

class FoulType(Enum):
    """Enumeration for foul types."""
    NO_FOUL = "no_foul"
    CARELESS = "careless"  # Results in a foul
    RECKLESS = "reckless"  # Results in a yellow card
    EXCESSIVE_FORCE = "excessive_force"  # Results in a red card
    HANDBALL = "handball"
    HOLDING = "holding"
    DANGEROUS_PLAY = "dangerous_play"

class CardType(Enum):
    """Enumeration for card types."""
    NONE = "none"
    YELLOW = "yellow"
    RED = "red"

class DecisionType(Enum):
    """Enumeration for decision types."""
    NO_FOUL = "no_foul"
    FOUL = "foul"
    PENALTY = "penalty"
    YELLOW_CARD = "yellow_card"
    RED_CARD = "red_card"

@dataclass
class IncidentContext:
    """Data class for the context of an in-game incident."""
    location: str  # e.g., 'penalty_area', 'midfield', 'defensive_third'
    action_type: str  # e.g., 'kick', 'tackle', 'push', 'jump', 'handball'
    contact_intensity: float  # 0.0 to 1.0
    ball_proximity: float  # 0.0 (very close) to 1.0 (far)
    player_movement: str  # e.g., 'towards_ball', 'towards_player', 'stationary'
    body_position: str  # e.g., 'upright', 'sliding', 'jumping', 'falling'
    has_possession: bool  # Whether the player has possession of the ball
    denies_goal_opportunity: bool  # Whether the action denies an obvious goal-scoring opportunity

@dataclass
class RefereeDecision:
    """Represents the final decision of the analysis."""
    decision: DecisionType
    card: CardType
    is_penalty: bool
    confidence: float
    reasoning: List[str]
    rule_references: List[str]

class FootballRulesEngine:
    """A rules engine for analyzing football plays based on IFAB Laws of the Game."""
    
    def __init__(self):
        """Initializes the rules engine."""
        self.rules = self._load_rules()
    
    def _load_rules(self) -> Dict:
        """
        Loads the official IFAB rules.
        
        Returns:
            A dictionary with the structured rules.
        """
        return {
            'fouls': {
                'direct_free_kick_offences': [
                    'charges', 'jumps_at', 'kicks_or_attempts_to_kick', 'pushes',
                    'strikes_or_attempts_to_strike', 'tackles_or_challenges', 'trips_or_attempts_to_trip',
                    'handball', 'holds_opponent', 'impedes_with_contact', 'bites_or_spits', 'throws_object'
                ]
            },
            'yellow_card_offences': [
                'delay_restart', 'dissent', 'entering_leaving_field_without_permission',
                'not_respecting_distance', 'persistent_offences', 'unsporting_behaviour', 'reckless_challenge'
            ],
            'red_card_offences': [
                'denying_goal_with_handball', 'denying_goal_with_foul', 'serious_foul_play',
                'biting_or_spitting', 'violent_conduct', 'offensive_language', 'second_yellow_card', 'excessive_force'
            ],
            'penalty_conditions': [
                'foul_in_penalty_area', 'by_defending_team', 'ball_in_play'
            ]
        }
    
    def analyze_incident(self, 
                        context: IncidentContext,
                        foul_classification: Dict,
                        pose_analysis: Optional[Dict] = None) -> RefereeDecision:
        """
        Analyzes an incident and returns a decision.
        
        Args:
            context: The context of the incident.
            foul_classification: The result from the foul classifier model.
            pose_analysis: The result from the pose analysis model (optional).
            
        Returns:
            The final referee decision.
        """
        reasoning = []
        rule_references = []
        decision = DecisionType.NO_FOUL
        card = CardType.NONE
        is_penalty = False
        confidence = foul_classification.get('confidence', 0.5)
        
        is_foul, foul_type = self._check_foul(context, foul_classification)
        
        if not is_foul:
            reasoning.append("No foul detected - Clean play.")
            return RefereeDecision(DecisionType.NO_FOUL, CardType.NONE, False, confidence, reasoning, [])
        
        decision = DecisionType.FOUL
        reasoning.append(f"Foul detected: {foul_type.value}")
        rule_references.append("IFAB Law 12 - Fouls and Misconduct")
        
        if context.location == 'penalty_area':
            is_penalty = True
            decision = DecisionType.PENALTY
            reasoning.append("Foul committed inside the penalty area - PENALTY KICK.")
            rule_references.append("IFAB Law 14 - The Penalty Kick")
        
        card, card_reasoning = self._determine_card(context, foul_type)
        reasoning.extend(card_reasoning)
        
        if card == CardType.YELLOW:
            decision = DecisionType.YELLOW_CARD
            rule_references.append("IFAB Law 12 - Cautionable Offences")
        elif card == CardType.RED:
            decision = DecisionType.RED_CARD
            rule_references.append("IFAB Law 12 - Sending-off Offences")
        
        special_checks = self._special_rule_checks(context, foul_type)
        reasoning.extend(special_checks['reasoning'])
        rule_references.extend(special_checks['rules'])
        
        if special_checks.get('upgrade_to_red', False):
            card = CardType.RED
            decision = DecisionType.RED_CARD
        
        return RefereeDecision(decision, card, is_penalty, confidence, reasoning, list(set(rule_references)))
    
    def _check_foul(self, context: IncidentContext, classification: Dict) -> tuple[bool, Optional[FoulType]]:
        """Checks if a foul occurred based on model classification and context."""
        if classification['class'] == 'no_foul':
            return False, None
        
        if context.action_type == 'handball':
            return True, FoulType.HANDBALL
        
        if context.contact_intensity < 0.3:
            return False, None  # Minimal contact is not a foul
        
        if context.contact_intensity >= 0.8:
            return True, FoulType.EXCESSIVE_FORCE
        
        if context.contact_intensity >= 0.6:
            return True, FoulType.RECKLESS
        
        return True, FoulType.CARELESS
    
    def _determine_card(self, context: IncidentContext, foul_type: FoulType) -> tuple[CardType, List[str]]:
        """Determines the card based on the foul type and context."""
        reasoning = []
        
        if foul_type == FoulType.EXCESSIVE_FORCE:
            reasoning.append("Use of excessive force - endangers the safety of an opponent.")
            return CardType.RED, reasoning
        
        if context.denies_goal_opportunity:
            reasoning.append("Denies an opponent an obvious goal-scoring opportunity.")
            return CardType.RED, reasoning
        
        if foul_type == FoulType.RECKLESS:
            reasoning.append("Reckless challenge - acts with disregard to the danger to the opponent.")
            return CardType.YELLOW, reasoning
        
        if context.action_type == 'tackle' and context.body_position == 'sliding' and context.contact_intensity > 0.5:
            reasoning.append("Reckless sliding tackle.")
            return CardType.YELLOW, reasoning
        
        if foul_type == FoulType.CARELESS:
            reasoning.append("Careless foul - no further disciplinary action needed.")
            return CardType.NONE, reasoning
        
        return CardType.NONE, reasoning
    
    def _special_rule_checks(self, context: IncidentContext, foul_type: FoulType) -> Dict:
        """Performs special rule checks, such as for handball or DOGSO."""
        reasoning = []
        rules = []
        upgrade_to_red = False
        
        if foul_type == FoulType.HANDBALL:
            if context.location == 'penalty_area':
                reasoning.append("Handball inside the penalty area.")
            if context.denies_goal_opportunity:
                reasoning.append("Denies a goal with a deliberate handball - Red Card.")
                upgrade_to_red = True
                rules.append("IFAB Law 12 - Denying goal with handball")
        
        if context.denies_goal_opportunity and context.location in ['penalty_area', 'defensive_third']:
            reasoning.append("Last line of defense - denies an obvious goal-scoring opportunity.")
            upgrade_to_red = True
        
        if context.ball_proximity > 0.7 and context.contact_intensity > 0.5:
            reasoning.append("Foul committed away from the ball - unsporting behaviour.")
            rules.append("IFAB Law 12 - Unsporting Behaviour")
        
        return {'reasoning': reasoning, 'rules': rules, 'upgrade_to_red': upgrade_to_red}
    
    def explain_decision(self, decision: RefereeDecision) -> str:
        """Generates a detailed textual explanation of the decision."""
        explanation = []
        explanation.append("=" * 60)
        explanation.append("ANALYSIS RESULT")
        explanation.append("=" * 60)
        explanation.append("")
        
        explanation.append(f"DECISION: {decision.decision.value.upper()}")
        explanation.append(f"Confidence: {decision.confidence:.1%}")
        explanation.append("")
        
        if decision.card != CardType.NONE:
            card_emoji = "🟨" if decision.card == CardType.YELLOW else "🟥"
            explanation.append(f"CARD: {card_emoji} {decision.card.value.upper()}")
            explanation.append("")
        
        if decision.is_penalty:
            explanation.append("⚽ PENALTY KICK")
            explanation.append("")
        
        explanation.append("ANALYSIS:")
        for i, reason in enumerate(decision.reasoning, 1):
            explanation.append(f"  {i}. {reason}")
        explanation.append("")
        
        if decision.rule_references:
            explanation.append("APPLIED RULES:")
            for rule in decision.rule_references:
                explanation.append(f"  • {rule}")
        
        explanation.append("=" * 60)
        
        return "\n".join(explanation)

if __name__ == "__main__":
    engine = FootballRulesEngine()
    print("Rules engine initialized successfully!")
    print(f"Loaded rules: {list(engine.rules.keys())}")

