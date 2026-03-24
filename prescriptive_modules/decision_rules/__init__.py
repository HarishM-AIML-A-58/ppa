"""
Decision Rules Engine Package
================================
Exports the DecisionRulesEngine class and supporting data structures.
"""

from prescriptive_modules.decision_rules.engine import (
    DecisionRulesEngine,
    Rule,
    RuleSet,
    Condition,
    demo as run_decision_rules_demo,
)

__all__ = [
    "DecisionRulesEngine",
    "Rule",
    "RuleSet",
    "Condition",
    "run_decision_rules_demo",
]
