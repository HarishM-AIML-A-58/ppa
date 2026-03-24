"""
Prescriptive Modules Package
==============================
Top-level package exposing the three prescriptive analytics modules:

- recommendation_engine : Multi-strategy recommender (CF, content, association, hybrid)
- decision_rules        : Rule extraction, evaluation, and prediction engine
- optimization          : Pricing, resource allocation, inventory, and portfolio optimizers
"""

from prescriptive_modules.recommendation_engine import (
    RecommendationEngine,
    CollaborativeFilteringRecommender,
    ContentBasedRecommender,
    AssociationRuleRecommender,
    HybridRecommender,
)

from prescriptive_modules.decision_rules import (
    DecisionRulesEngine,
    Rule,
    RuleSet,
    Condition,
)

from prescriptive_modules.optimization import (
    PricingOptimizer,
    ResourceAllocationOptimizer,
    InventoryOptimizer,
    PortfolioOptimizer,
)

__all__ = [
    # Recommendation Engine
    "RecommendationEngine",
    "CollaborativeFilteringRecommender",
    "ContentBasedRecommender",
    "AssociationRuleRecommender",
    "HybridRecommender",
    # Decision Rules
    "DecisionRulesEngine",
    "Rule",
    "RuleSet",
    "Condition",
    # Optimizers
    "PricingOptimizer",
    "ResourceAllocationOptimizer",
    "InventoryOptimizer",
    "PortfolioOptimizer",
]
