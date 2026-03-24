"""
Recommendation Engine Package
==============================
Exports the main RecommendationEngine class and its component recommenders.
"""

from prescriptive_modules.recommendation_engine.engine import (
    RecommendationEngine,
    CollaborativeFilteringRecommender,
    ContentBasedRecommender,
    AssociationRuleRecommender,
    HybridRecommender,
    RecommendationResult,
    demo as run_recommendation_demo,
)

__all__ = [
    "RecommendationEngine",
    "CollaborativeFilteringRecommender",
    "ContentBasedRecommender",
    "AssociationRuleRecommender",
    "HybridRecommender",
    "RecommendationResult",
    "run_recommendation_demo",
]
