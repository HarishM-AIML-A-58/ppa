"""
Recommendation Engine Module
=============================
Production-grade recommendation engine implementing multiple strategies:
- Collaborative Filtering (user-based and item-based)
- Content-Based Filtering (TF-IDF + cosine similarity)
- Association Rule-based Recommendations
- Hybrid Recommender (weighted combination)

Author: PPA Analytics Team
"""

import os
import json
import logging
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union, Any
from dataclasses import dataclass, field, asdict

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import csr_matrix

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Output directories
# ---------------------------------------------------------------------------
OUTPUT_DIR = Path("/home/user/ppa/outputs/metrics")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PROCESSED_DIR = Path("/home/user/ppa/datasets/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class RecommendationResult:
    """Container for a single recommendation."""
    item_id: Any
    score: float
    reason: str
    strategy: str


# ---------------------------------------------------------------------------
# Collaborative Filtering Recommender
# ---------------------------------------------------------------------------
class CollaborativeFilteringRecommender:
    """
    User-based and item-based Collaborative Filtering using cosine similarity.

    Attributes
    ----------
    user_item_matrix : pd.DataFrame
        Rows = users, columns = items, values = ratings (0 = not rated).
    user_similarity : np.ndarray
        Pairwise user cosine similarity matrix.
    item_similarity : np.ndarray
        Pairwise item cosine similarity matrix.
    mode : str
        'user' for user-based CF, 'item' for item-based CF.
    """

    def __init__(self, mode: str = "user"):
        """
        Parameters
        ----------
        mode : str
            'user' (default) or 'item'.
        """
        if mode not in ("user", "item"):
            raise ValueError("mode must be 'user' or 'item'.")
        self.mode = mode
        self.user_item_matrix: Optional[pd.DataFrame] = None
        self.user_similarity: Optional[np.ndarray] = None
        self.item_similarity: Optional[np.ndarray] = None
        self._is_fitted = False

    # ------------------------------------------------------------------
    def fit(self, user_item_matrix: pd.DataFrame) -> "CollaborativeFilteringRecommender":
        """
        Compute similarity matrices from the user-item matrix.

        Parameters
        ----------
        user_item_matrix : pd.DataFrame
            Shape (n_users, n_items).  Missing ratings should be 0.

        Returns
        -------
        self
        """
        logger.info("CollaborativeFilteringRecommender.fit() — mode=%s", self.mode)
        self.user_item_matrix = user_item_matrix.copy()

        matrix = user_item_matrix.values.astype(float)

        self.user_similarity = cosine_similarity(matrix)
        self.item_similarity = cosine_similarity(matrix.T)

        self._is_fitted = True
        logger.info(
            "CF fit complete: %d users, %d items.",
            user_item_matrix.shape[0],
            user_item_matrix.shape[1],
        )
        return self

    # ------------------------------------------------------------------
    def _predict_user_based(self, user_idx: int) -> np.ndarray:
        """Return predicted ratings for all items using user-based CF."""
        sim_scores = self.user_similarity[user_idx]  # (n_users,)
        # Exclude the target user from influencing its own score
        sim_scores_copy = sim_scores.copy()
        sim_scores_copy[user_idx] = 0.0

        matrix = self.user_item_matrix.values.astype(float)
        numerator = sim_scores_copy @ matrix
        denominator = np.abs(sim_scores_copy).sum() + 1e-10
        return numerator / denominator

    def _predict_item_based(self, user_idx: int) -> np.ndarray:
        """Return predicted ratings for all items using item-based CF."""
        user_ratings = self.user_item_matrix.values[user_idx].astype(float)
        # For each item, predict as weighted average of rated items
        predicted = self.item_similarity @ user_ratings
        denominator = np.abs(self.item_similarity).sum(axis=1) + 1e-10
        return predicted / denominator

    # ------------------------------------------------------------------
    def recommend(
        self,
        user_id: Any,
        n: int = 10,
    ) -> List[Tuple[Any, float, str]]:
        """
        Recommend items for a given user.

        Parameters
        ----------
        user_id : Any
            Must be a valid index in user_item_matrix.
        n : int
            Number of recommendations to return.

        Returns
        -------
        List of (item_id, score, reason) tuples sorted by descending score.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before recommend().")

        if user_id not in self.user_item_matrix.index:
            raise ValueError(f"user_id '{user_id}' not found in training data.")

        user_idx = self.user_item_matrix.index.get_loc(user_id)
        already_rated = set(
            self.user_item_matrix.columns[
                self.user_item_matrix.values[user_idx] > 0
            ]
        )

        if self.mode == "user":
            scores = self._predict_user_based(user_idx)
            reason_tpl = "Users similar to you also liked this item (user-based CF)."
        else:
            scores = self._predict_item_based(user_idx)
            reason_tpl = "This item is similar to items you have rated highly (item-based CF)."

        items = self.user_item_matrix.columns
        results = [
            (item, float(score), reason_tpl)
            for item, score in zip(items, scores)
            if item not in already_rated
        ]
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:n]

    # ------------------------------------------------------------------
    def get_similar_items(self, item_id: Any, n: int = 5) -> List[Tuple[Any, float]]:
        """Return the n most similar items to item_id."""
        if not self._is_fitted:
            raise RuntimeError("Call fit() first.")
        if item_id not in self.user_item_matrix.columns:
            raise ValueError(f"item_id '{item_id}' not in training data.")

        item_idx = self.user_item_matrix.columns.get_loc(item_id)
        sims = self.item_similarity[item_idx]
        items = self.user_item_matrix.columns

        ranked = sorted(
            [(it, float(s)) for it, s in zip(items, sims) if it != item_id],
            key=lambda x: x[1],
            reverse=True,
        )
        return ranked[:n]


# ---------------------------------------------------------------------------
# Content-Based Recommender
# ---------------------------------------------------------------------------
class ContentBasedRecommender:
    """
    Content-based recommender using TF-IDF on item feature text + cosine similarity.

    Items are described by free-text feature strings (e.g. genre, tags, description).
    """

    def __init__(self, max_features: int = 500):
        self.max_features = max_features
        self.tfidf = TfidfVectorizer(max_features=max_features, stop_words="english")
        self.item_vectors: Optional[np.ndarray] = None
        self.item_ids: Optional[List] = None
        self.item_similarity: Optional[np.ndarray] = None
        self.user_profiles: Dict[Any, np.ndarray] = {}
        self._is_fitted = False

    # ------------------------------------------------------------------
    def fit(
        self,
        item_features: pd.DataFrame,
        user_item_matrix: Optional[pd.DataFrame] = None,
    ) -> "ContentBasedRecommender":
        """
        Fit TF-IDF model on item feature text.

        Parameters
        ----------
        item_features : pd.DataFrame
            Must contain column 'features' (str) and index = item_ids.
        user_item_matrix : pd.DataFrame, optional
            Used to build per-user content profiles.
        """
        logger.info("ContentBasedRecommender.fit() — %d items.", len(item_features))
        if "features" not in item_features.columns:
            raise ValueError("item_features must contain a 'features' column.")

        self.item_ids = list(item_features.index)
        self.item_vectors = self.tfidf.fit_transform(
            item_features["features"].fillna("")
        ).toarray()
        self.item_similarity = cosine_similarity(self.item_vectors)

        if user_item_matrix is not None:
            self._build_user_profiles(user_item_matrix, item_features)

        self._is_fitted = True
        logger.info("ContentBasedRecommender fit complete.")
        return self

    def _build_user_profiles(
        self, user_item_matrix: pd.DataFrame, item_features: pd.DataFrame
    ) -> None:
        """Build a weighted TF-IDF profile for each user."""
        common_items = [
            it for it in user_item_matrix.columns if it in item_features.index
        ]
        item_idx_map = {it: self.item_ids.index(it) for it in common_items}

        for user_id in user_item_matrix.index:
            ratings = user_item_matrix.loc[user_id, common_items].values.astype(float)
            if ratings.sum() == 0:
                self.user_profiles[user_id] = np.zeros(self.item_vectors.shape[1])
                continue
            vecs = np.array([self.item_vectors[item_idx_map[it]] for it in common_items])
            profile = (ratings[:, np.newaxis] * vecs).sum(axis=0) / (ratings.sum() + 1e-10)
            self.user_profiles[user_id] = profile

    # ------------------------------------------------------------------
    def recommend(
        self,
        user_id: Any,
        n: int = 10,
        already_rated: Optional[set] = None,
    ) -> List[Tuple[Any, float, str]]:
        """
        Recommend items to a user based on their content profile.

        Returns list of (item_id, score, reason).
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() first.")

        if user_id not in self.user_profiles:
            logger.warning("No profile for user %s — returning empty list.", user_id)
            return []

        profile = self.user_profiles[user_id].reshape(1, -1)
        scores = cosine_similarity(profile, self.item_vectors)[0]
        already_rated = already_rated or set()

        results = [
            (item, float(score), "Based on content features of items you enjoyed (content-based).")
            for item, score in zip(self.item_ids, scores)
            if item not in already_rated
        ]
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:n]

    # ------------------------------------------------------------------
    def get_similar_items(self, item_id: Any, n: int = 5) -> List[Tuple[Any, float]]:
        """Return n most similar items by content."""
        if not self._is_fitted:
            raise RuntimeError("Call fit() first.")
        if item_id not in self.item_ids:
            raise ValueError(f"item_id '{item_id}' not found.")

        idx = self.item_ids.index(item_id)
        sims = self.item_similarity[idx]
        ranked = sorted(
            [(it, float(s)) for it, s in zip(self.item_ids, sims) if it != item_id],
            key=lambda x: x[1],
            reverse=True,
        )
        return ranked[:n]


# ---------------------------------------------------------------------------
# Association Rule Recommender
# ---------------------------------------------------------------------------
class AssociationRuleRecommender:
    """
    Recommender based on pre-computed association rules.

    Rules are loaded from datasets/processed/association_rules.json (if present)
    or supplied directly via fit().

    Each rule has the format:
        {"antecedents": [item_a, item_b], "consequents": [item_c], "lift": 2.5, "confidence": 0.8}
    """

    def __init__(self):
        self.rules: List[Dict] = []
        self._is_fitted = False

    # ------------------------------------------------------------------
    def fit(
        self,
        rules: Optional[List[Dict]] = None,
        rules_path: Optional[str] = None,
    ) -> "AssociationRuleRecommender":
        """
        Load association rules.

        Parameters
        ----------
        rules : list of dict, optional
            Pre-computed rules.
        rules_path : str, optional
            Path to JSON file containing rules.
        """
        if rules is not None:
            self.rules = rules
        elif rules_path and Path(rules_path).exists():
            with open(rules_path) as f:
                self.rules = json.load(f)
            logger.info("Loaded %d association rules from %s.", len(self.rules), rules_path)
        else:
            # Attempt default path
            default = PROCESSED_DIR / "association_rules.json"
            if default.exists():
                with open(default) as f:
                    self.rules = json.load(f)
                logger.info("Loaded %d rules from default path.", len(self.rules))
            else:
                logger.warning("No association rules provided or found.  Using empty rule set.")
                self.rules = []

        self._is_fitted = True
        return self

    # ------------------------------------------------------------------
    def recommend(
        self,
        user_id: Any,
        basket: List[Any],
        n: int = 10,
    ) -> List[Tuple[Any, float, str]]:
        """
        Recommend items whose antecedents are contained in the user's basket.

        Parameters
        ----------
        user_id : Any
            Identifier (used for logging only).
        basket : list
            Items already in user's basket / purchase history.
        n : int
            Max recommendations.

        Returns
        -------
        List of (item_id, score, reason).
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() first.")

        basket_set = set(basket)
        scored: Dict[Any, float] = {}

        for rule in self.rules:
            ants = set(rule.get("antecedents", []))
            cons = rule.get("consequents", [])
            lift = float(rule.get("lift", 1.0))
            conf = float(rule.get("confidence", 0.0))

            if ants.issubset(basket_set):
                for item in cons:
                    if item not in basket_set:
                        current = scored.get(item, 0.0)
                        scored[item] = max(current, lift * conf)

        results = [
            (
                item,
                score,
                f"Frequently purchased together with {list(basket_set)[:3]} (association rules, lift={score:.2f}).",
            )
            for item, score in scored.items()
        ]
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:n]


# ---------------------------------------------------------------------------
# Hybrid Recommender
# ---------------------------------------------------------------------------
class HybridRecommender:
    """
    Weighted ensemble of CF (user-based), content-based, and association-rule recommenders.

    Weights are normalised to sum to 1.
    """

    def __init__(
        self,
        cf_weight: float = 0.5,
        cb_weight: float = 0.3,
        ar_weight: float = 0.2,
    ):
        total = cf_weight + cb_weight + ar_weight
        self.cf_weight = cf_weight / total
        self.cb_weight = cb_weight / total
        self.ar_weight = ar_weight / total

        self.cf = CollaborativeFilteringRecommender(mode="user")
        self.cb = ContentBasedRecommender()
        self.ar = AssociationRuleRecommender()

        self._is_fitted = False

    # ------------------------------------------------------------------
    def fit(
        self,
        user_item_matrix: pd.DataFrame,
        item_features: Optional[pd.DataFrame] = None,
        rules: Optional[List[Dict]] = None,
    ) -> "HybridRecommender":
        """
        Fit all sub-recommenders.

        Parameters
        ----------
        user_item_matrix : pd.DataFrame
        item_features : pd.DataFrame, optional
            Required for content-based component.
        rules : list of dict, optional
            Pre-computed association rules.
        """
        logger.info("HybridRecommender.fit() — fitting sub-recommenders.")
        self.user_item_matrix = user_item_matrix.copy()

        self.cf.fit(user_item_matrix)

        if item_features is not None:
            self.cb.fit(item_features, user_item_matrix)
            self._cb_available = True
        else:
            logger.warning("No item_features — content-based component disabled.")
            self._cb_available = False

        self.ar.fit(rules=rules)
        self._is_fitted = True
        return self

    # ------------------------------------------------------------------
    def recommend(
        self,
        user_id: Any,
        n: int = 10,
    ) -> List[Tuple[Any, float, str]]:
        """
        Blend recommendations from all three strategies.

        Returns
        -------
        List of (item_id, score, reason) sorted by descending hybrid score.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() first.")

        # --- CF scores ---
        try:
            cf_recs = {
                item: score for item, score, _ in self.cf.recommend(user_id, n=50)
            }
        except Exception as exc:
            logger.warning("CF component error: %s", exc)
            cf_recs = {}

        # --- Content-based scores ---
        if self._cb_available:
            user_idx = self.user_item_matrix.index.get_loc(user_id)
            already = set(
                self.user_item_matrix.columns[self.user_item_matrix.values[user_idx] > 0]
            )
            try:
                cb_recs = {
                    item: score
                    for item, score, _ in self.cb.recommend(user_id, n=50, already_rated=already)
                }
            except Exception as exc:
                logger.warning("CB component error: %s", exc)
                cb_recs = {}
        else:
            cb_recs = {}

        # --- Association rule scores (use full history as basket) ---
        basket = list(
            self.user_item_matrix.columns[
                self.user_item_matrix.loc[user_id].values > 0
            ]
        )
        try:
            ar_recs = {
                item: score for item, score, _ in self.ar.recommend(user_id, basket, n=50)
            }
        except Exception as exc:
            logger.warning("AR component error: %s", exc)
            ar_recs = {}

        # Normalise each source to [0, 1]
        def _normalize(d: Dict) -> Dict:
            if not d:
                return d
            max_v = max(d.values()) or 1.0
            return {k: v / max_v for k, v in d.items()}

        cf_recs = _normalize(cf_recs)
        cb_recs = _normalize(cb_recs)
        ar_recs = _normalize(ar_recs)

        # Combine
        all_items = set(cf_recs) | set(cb_recs) | set(ar_recs)
        blended: List[Tuple[Any, float, str]] = []

        for item in all_items:
            cf_s = cf_recs.get(item, 0.0) * self.cf_weight
            cb_s = cb_recs.get(item, 0.0) * self.cb_weight
            ar_s = ar_recs.get(item, 0.0) * self.ar_weight
            hybrid_score = cf_s + cb_s + ar_s

            parts = []
            if cf_s > 0:
                parts.append(f"CF({cf_s:.2f})")
            if cb_s > 0:
                parts.append(f"Content({cb_s:.2f})")
            if ar_s > 0:
                parts.append(f"Rules({ar_s:.2f})")
            reason = "Hybrid recommendation: " + " + ".join(parts)

            blended.append((item, hybrid_score, reason))

        blended.sort(key=lambda x: x[1], reverse=True)
        return blended[:n]


# ---------------------------------------------------------------------------
# Main RecommendationEngine
# ---------------------------------------------------------------------------
class RecommendationEngine:
    """
    Unified recommendation engine exposing multiple strategies through a single API.

    Strategies
    ----------
    'collaborative_user'  : User-based collaborative filtering.
    'collaborative_item'  : Item-based collaborative filtering.
    'content'             : Content-based filtering.
    'association'         : Association rule-based.
    'hybrid'              : Weighted blend of all three (default).

    Example
    -------
    >>> engine = RecommendationEngine()
    >>> engine.fit(user_item_matrix, item_features=features_df, rules=rules_list)
    >>> recs = engine.recommend(user_id='U001', n=10, strategy='hybrid')
    """

    def __init__(self):
        self._cf_user = CollaborativeFilteringRecommender(mode="user")
        self._cf_item = CollaborativeFilteringRecommender(mode="item")
        self._cb = ContentBasedRecommender()
        self._ar = AssociationRuleRecommender()
        self._hybrid = HybridRecommender()

        self._user_item_matrix: Optional[pd.DataFrame] = None
        self._item_features: Optional[pd.DataFrame] = None
        self._is_fitted = False

        logger.info("RecommendationEngine initialised.")

    # ------------------------------------------------------------------
    def fit(
        self,
        user_item_matrix: pd.DataFrame,
        item_features: Optional[pd.DataFrame] = None,
        rules: Optional[List[Dict]] = None,
    ) -> "RecommendationEngine":
        """
        Fit all recommenders.

        Parameters
        ----------
        user_item_matrix : pd.DataFrame
            Shape (n_users, n_items).  Index = user_ids, columns = item_ids.
            Values = ratings (0 if not rated).
        item_features : pd.DataFrame, optional
            Index = item_ids.  Must have a 'features' column (str).
        rules : list of dict, optional
            Pre-computed association rules.  If None, loads from processed/ dir.
        """
        logger.info(
            "RecommendationEngine.fit() — matrix shape %s.", user_item_matrix.shape
        )
        self._user_item_matrix = user_item_matrix.copy()
        self._item_features = item_features

        self._cf_user.fit(user_item_matrix)
        self._cf_item.fit(user_item_matrix)

        if item_features is not None:
            self._cb.fit(item_features, user_item_matrix)

        self._ar.fit(rules=rules)
        self._hybrid.fit(user_item_matrix, item_features=item_features, rules=rules)

        self._is_fitted = True
        logger.info("RecommendationEngine fit complete.")
        return self

    # ------------------------------------------------------------------
    def recommend(
        self,
        user_id: Any,
        n: int = 10,
        strategy: str = "hybrid",
    ) -> List[Tuple[Any, float, str]]:
        """
        Generate recommendations for a user.

        Parameters
        ----------
        user_id : Any
        n : int
            Number of recommendations.
        strategy : str
            One of: 'collaborative_user', 'collaborative_item', 'content',
            'association', 'hybrid'.

        Returns
        -------
        List of (item_id, score, reason) tuples.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before recommend().")

        strategy = strategy.lower()
        logger.info("recommend() — user=%s  n=%d  strategy=%s", user_id, n, strategy)

        if strategy == "collaborative_user":
            return self._cf_user.recommend(user_id, n)
        elif strategy == "collaborative_item":
            return self._cf_item.recommend(user_id, n)
        elif strategy == "content":
            if self._item_features is None:
                raise RuntimeError("item_features not provided — cannot use content strategy.")
            user_idx = self._user_item_matrix.index.get_loc(user_id)
            already = set(
                self._user_item_matrix.columns[self._user_item_matrix.values[user_idx] > 0]
            )
            return self._cb.recommend(user_id, n, already_rated=already)
        elif strategy == "association":
            if self._user_item_matrix is None:
                raise RuntimeError("user_item_matrix not set.")
            basket = list(
                self._user_item_matrix.columns[
                    self._user_item_matrix.loc[user_id].values > 0
                ]
            )
            return self._ar.recommend(user_id, basket, n)
        elif strategy == "hybrid":
            return self._hybrid.recommend(user_id, n)
        else:
            raise ValueError(
                f"Unknown strategy '{strategy}'.  Choose from: "
                "collaborative_user, collaborative_item, content, association, hybrid."
            )

    # ------------------------------------------------------------------
    def get_similar_items(self, item_id: Any, n: int = 5) -> List[Tuple[Any, float]]:
        """
        Return n most similar items using item-based CF similarity.

        Parameters
        ----------
        item_id : Any
        n : int

        Returns
        -------
        List of (item_id, similarity_score).
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() first.")
        return self._cf_item.get_similar_items(item_id, n)

    # ------------------------------------------------------------------
    def explain_recommendation(self, user_id: Any, item_id: Any) -> str:
        """
        Generate a human-readable explanation for why item_id is recommended to user_id.

        Parameters
        ----------
        user_id : Any
        item_id : Any

        Returns
        -------
        str  — explanation paragraph.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() first.")

        if self._user_item_matrix is None:
            return "Model not fitted."

        user_idx = self._user_item_matrix.index.get_loc(user_id)
        rated_items = list(
            self._user_item_matrix.columns[self._user_item_matrix.values[user_idx] > 0]
        )

        # CF contribution
        user_sim_scores = self._cf_user.user_similarity[user_idx]
        top_similar_users_idx = np.argsort(user_sim_scores)[::-1][1:4]
        top_similar_users = [
            self._user_item_matrix.index[i] for i in top_similar_users_idx
        ]

        # Item similarity
        if item_id in self._user_item_matrix.columns:
            item_idx = self._user_item_matrix.columns.get_loc(item_id)
            cf_item_sim = self._cf_item.item_similarity[item_idx]
            common_rated = [
                (self._user_item_matrix.columns[i], float(cf_item_sim[i]))
                for i in np.argsort(cf_item_sim)[::-1]
                if self._user_item_matrix.columns[i] in rated_items
            ][:3]
        else:
            common_rated = []

        explanation_parts = [
            f"Recommendation explanation for user '{user_id}' → item '{item_id}':",
            "",
            f"  • You have rated {len(rated_items)} items: {rated_items[:5]}{'...' if len(rated_items) > 5 else ''}.",
            f"  • Users most similar to you: {top_similar_users}.",
        ]
        if common_rated:
            explanation_parts.append(
                f"  • This item is similar to items you liked: "
                + ", ".join(f"{it} (sim={s:.2f})" for it, s in common_rated)
                + "."
            )

        # Association rule contribution
        basket_set = set(rated_items)
        matching_rules = [
            r for r in self._ar.rules
            if set(r.get("antecedents", [])).issubset(basket_set)
            and item_id in r.get("consequents", [])
        ]
        if matching_rules:
            best = max(matching_rules, key=lambda r: r.get("lift", 0))
            explanation_parts.append(
                f"  • Association rule: {best['antecedents']} → {best['consequents']} "
                f"(lift={best.get('lift', 'N/A'):.2f}, confidence={best.get('confidence', 'N/A'):.2f})."
            )

        return "\n".join(explanation_parts)

    # ------------------------------------------------------------------
    def batch_recommend(
        self,
        user_ids: List[Any],
        n: int = 10,
        strategy: str = "hybrid",
    ) -> Dict[Any, List[Tuple[Any, float, str]]]:
        """
        Vectorised recommendation for multiple users.

        Parameters
        ----------
        user_ids : list
        n : int
        strategy : str

        Returns
        -------
        dict mapping user_id → list of (item_id, score, reason).
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() first.")

        logger.info(
            "batch_recommend() — %d users, strategy=%s.", len(user_ids), strategy
        )
        results: Dict[Any, List] = {}

        for uid in user_ids:
            try:
                results[uid] = self.recommend(uid, n=n, strategy=strategy)
            except Exception as exc:
                logger.warning("Failed to recommend for user %s: %s", uid, exc)
                results[uid] = []

        return results


# ---------------------------------------------------------------------------
# Demo / entry-point
# ---------------------------------------------------------------------------
def _generate_synthetic_data(
    n_users: int = 50,
    n_items: int = 30,
    density: float = 0.3,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[Dict]]:
    """
    Generate synthetic user-item matrix, item features, and association rules.

    Returns
    -------
    user_item_matrix, item_features, rules
    """
    rng = np.random.default_rng(seed)

    user_ids = [f"U{i:03d}" for i in range(n_users)]
    item_ids = [f"I{j:03d}" for j in range(n_items)]

    # Sparse rating matrix (0 = not rated, 1-5 = rating)
    mask = rng.random((n_users, n_items)) < density
    ratings = rng.integers(1, 6, size=(n_users, n_items)).astype(float)
    ratings[~mask] = 0.0
    user_item_matrix = pd.DataFrame(ratings, index=user_ids, columns=item_ids)

    # Item features
    categories = ["electronics", "books", "clothing", "sports", "home", "music"]
    tags = ["popular", "new", "trending", "classic", "premium", "budget"]

    feature_texts = [
        f"{rng.choice(categories)} {rng.choice(tags)} {rng.choice(categories)} quality"
        for _ in item_ids
    ]
    item_features = pd.DataFrame(
        {"features": feature_texts, "category": [t.split()[0] for t in feature_texts]},
        index=item_ids,
    )

    # Synthetic association rules
    rules = []
    for _ in range(40):
        n_ant = rng.integers(1, 3)
        ants = list(rng.choice(item_ids, size=int(n_ant), replace=False))
        cons = list(rng.choice([i for i in item_ids if i not in ants], size=1, replace=False))
        rules.append(
            {
                "antecedents": ants,
                "consequents": cons,
                "support": float(rng.uniform(0.05, 0.3)),
                "confidence": float(rng.uniform(0.4, 0.9)),
                "lift": float(rng.uniform(1.1, 4.0)),
            }
        )

    return user_item_matrix, item_features, rules


def demo() -> None:
    """
    End-to-end demonstration of the RecommendationEngine.

    Generates synthetic data, fits the engine, runs all strategies,
    and saves results to outputs/metrics/recommendations.json.
    """
    logger.info("=" * 60)
    logger.info("RecommendationEngine — Demo")
    logger.info("=" * 60)

    user_item_matrix, item_features, rules = _generate_synthetic_data(
        n_users=50, n_items=30, density=0.3
    )
    logger.info(
        "Synthetic data: %d users × %d items, %d rules.",
        *user_item_matrix.shape,
        len(rules),
    )

    engine = RecommendationEngine()
    engine.fit(user_item_matrix, item_features=item_features, rules=rules)

    test_users = list(user_item_matrix.index[:5])
    output: Dict[str, Any] = {"strategies": {}, "similar_items": {}, "explanations": {}}

    strategies = [
        "collaborative_user",
        "collaborative_item",
        "content",
        "association",
        "hybrid",
    ]

    for strategy in strategies:
        logger.info("--- Strategy: %s ---", strategy)
        strategy_results: Dict[str, List] = {}
        for uid in test_users:
            try:
                recs = engine.recommend(uid, n=5, strategy=strategy)
                strategy_results[uid] = [
                    {"item_id": str(r[0]), "score": round(r[1], 4), "reason": r[2]}
                    for r in recs
                ]
                logger.info("  User %s → %s", uid, [r[0] for r in recs])
            except Exception as exc:
                logger.warning("  User %s failed: %s", uid, exc)
                strategy_results[uid] = []
        output["strategies"][strategy] = strategy_results

    # Similar items
    test_item = list(user_item_matrix.columns)[0]
    similar = engine.get_similar_items(test_item, n=5)
    output["similar_items"][str(test_item)] = [
        {"item_id": str(it), "similarity": round(s, 4)} for it, s in similar
    ]
    logger.info("Similar items to %s: %s", test_item, [s[0] for s in similar])

    # Explanation
    test_user = test_users[0]
    test_rec_item = list(user_item_matrix.columns)[5]
    explanation = engine.explain_recommendation(test_user, test_rec_item)
    output["explanations"][f"{test_user}_{test_rec_item}"] = explanation
    logger.info("Explanation:\n%s", explanation)

    # Batch recommend
    batch = engine.batch_recommend(test_users, n=5, strategy="hybrid")
    output["batch_hybrid"] = {
        uid: [
            {"item_id": str(r[0]), "score": round(r[1], 4), "reason": r[2]}
            for r in recs
        ]
        for uid, recs in batch.items()
    }

    # Save
    out_path = OUTPUT_DIR / "recommendations.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Recommendations saved to %s", out_path)


if __name__ == "__main__":
    demo()
