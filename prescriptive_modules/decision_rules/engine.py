"""
Decision Rules Engine Module
==============================
Production-grade rules engine that can:
- Extract IF-THEN rules from trained decision trees and random forests
- Support manually specified rule sets
- Evaluate rules (coverage, accuracy)
- Optimize rules (prune redundancy, rank by lift)
- Export rules to multiple formats (dict, SQL, JSON, natural language)
- Predict on new data with conflict resolution

Author: PPA Analytics Team
"""

import os
import json
import logging
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field, asdict
from copy import deepcopy

from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
OUTPUT_DIR = Path("/home/user/ppa/outputs/metrics")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class Condition:
    """A single condition within a rule (e.g. 'amount > 5000')."""
    feature: str
    operator: str        # '<', '<=', '>', '>=', '==', '!='
    threshold: float

    def evaluate(self, row: Union[pd.Series, Dict]) -> bool:
        """Return True if the condition is satisfied by the given data row."""
        if isinstance(row, dict):
            val = row.get(self.feature)
        else:
            val = row[self.feature]
        if val is None or pd.isna(val):
            return False
        ops = {
            "<": lambda a, b: a < b,
            "<=": lambda a, b: a <= b,
            ">": lambda a, b: a > b,
            ">=": lambda a, b: a >= b,
            "==": lambda a, b: a == b,
            "!=": lambda a, b: a != b,
        }
        if self.operator not in ops:
            raise ValueError(f"Unknown operator: {self.operator}")
        return ops[self.operator](float(val), float(self.threshold))

    def to_sql(self) -> str:
        return f"{self.feature} {self.operator} {self.threshold}"

    def to_natural_language(self) -> str:
        op_map = {
            ">": "greater than", ">=": "at least",
            "<": "less than", "<=": "at most",
            "==": "equal to", "!=": "not equal to",
        }
        op_str = op_map.get(self.operator, self.operator)
        return f"{self.feature} is {op_str} {self.threshold}"


@dataclass
class Rule:
    """
    An IF-THEN classification rule.

    Attributes
    ----------
    conditions : list of Condition
        Antecedent (IF part).
    prediction : Any
        Consequent class label (THEN part).
    confidence : float
        Fraction of samples satisfying conditions that belong to prediction class.
    support : float
        Fraction of total samples satisfying conditions.
    lift : float
        Confidence / (overall prevalence of prediction class).
    rule_id : str
        Unique identifier.
    """
    conditions: List[Condition]
    prediction: Any
    confidence: float = 0.0
    support: float = 0.0
    lift: float = 1.0
    rule_id: str = ""

    def evaluate(self, row: Union[pd.Series, Dict]) -> bool:
        """Return True if ALL conditions are satisfied."""
        return all(c.evaluate(row) for c in self.conditions)

    # ------------------------------------------------------------------
    def to_dict(self) -> Dict:
        return {
            "rule_id": self.rule_id,
            "conditions": [
                {"feature": c.feature, "operator": c.operator, "threshold": c.threshold}
                for c in self.conditions
            ],
            "prediction": str(self.prediction),
            "confidence": round(self.confidence, 4),
            "support": round(self.support, 4),
            "lift": round(self.lift, 4),
        }

    def to_sql(self) -> str:
        cond_str = " AND ".join(c.to_sql() for c in self.conditions)
        return f"CASE WHEN {cond_str} THEN '{self.prediction}' END"

    def to_natural_language(self) -> str:
        cond_str = " AND ".join(c.to_natural_language() for c in self.conditions)
        return (
            f"IF {cond_str} "
            f"THEN predict '{self.prediction}' "
            f"(confidence={self.confidence:.2%}, support={self.support:.2%}, lift={self.lift:.2f})."
        )


@dataclass
class RuleSet:
    """Collection of Rule objects with metadata."""
    rules: List[Rule] = field(default_factory=list)
    name: str = "RuleSet"
    target_classes: List[Any] = field(default_factory=list)
    feature_names: List[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.rules)

    def add(self, rule: Rule) -> None:
        self.rules.append(rule)

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "n_rules": len(self.rules),
            "target_classes": [str(c) for c in self.target_classes],
            "rules": [r.to_dict() for r in self.rules],
        }

    def to_json(self, path: Optional[str] = None) -> str:
        data = json.dumps(self.to_dict(), indent=2)
        if path:
            with open(path, "w") as f:
                f.write(data)
        return data

    def to_sql(self) -> str:
        lines = ["-- Auto-generated decision rules (SQL CASE expressions)"]
        for rule in self.rules:
            lines.append(rule.to_sql())
        return "\n".join(lines)

    def to_natural_language(self) -> str:
        lines = [f"# {self.name} — {len(self.rules)} rules\n"]
        for i, rule in enumerate(self.rules, 1):
            lines.append(f"Rule {i}: {rule.to_natural_language()}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# DecisionRulesEngine
# ---------------------------------------------------------------------------
class DecisionRulesEngine:
    """
    Extracts, manages, evaluates, and applies decision rules.

    Usage
    -----
    >>> engine = DecisionRulesEngine()
    >>> rules = engine.extract_from_tree(dt, feature_names)
    >>> engine.evaluate_rules(X_test, rules)
    >>> engine.predict(X_test)
    """

    def __init__(self, default_prediction: Any = "UNKNOWN"):
        self.ruleset: RuleSet = RuleSet()
        self.default_prediction = default_prediction
        self._class_prevalence: Dict[Any, float] = {}
        logger.info("DecisionRulesEngine initialised.")

    # ------------------------------------------------------------------
    # Rule extraction from Decision Tree
    # ------------------------------------------------------------------
    def extract_from_tree(
        self,
        decision_tree: DecisionTreeClassifier,
        feature_names: List[str],
        class_names: Optional[List[str]] = None,
        X_train: Optional[np.ndarray] = None,
        y_train: Optional[np.ndarray] = None,
    ) -> RuleSet:
        """
        Extract IF-THEN rules from a fitted DecisionTreeClassifier.

        Each root-to-leaf path becomes one rule.

        Parameters
        ----------
        decision_tree : DecisionTreeClassifier
        feature_names : list of str
        class_names : list of str, optional
        X_train : np.ndarray, optional  — used to compute support/confidence
        y_train : np.ndarray, optional

        Returns
        -------
        RuleSet
        """
        logger.info("Extracting rules from decision tree (depth=%d).", decision_tree.get_depth())

        tree_ = decision_tree.tree_
        if class_names is None:
            class_names = [str(c) for c in decision_tree.classes_]

        # Compute class prevalence for lift
        if y_train is not None:
            unique, counts = np.unique(y_train, return_counts=True)
            total = len(y_train)
            self._class_prevalence = {
                class_names[i]: counts[i] / total
                for i, cls in enumerate(unique)
            }

        ruleset = RuleSet(name="DecisionTreeRules", feature_names=feature_names)
        ruleset.target_classes = list(class_names)

        # Recursive path extraction
        def _traverse(node_id: int, path_conditions: List[Condition]) -> None:
            is_leaf = tree_.children_left[node_id] == _tree.TREE_LEAF

            if is_leaf:
                class_idx = int(np.argmax(tree_.value[node_id]))
                prediction = class_names[class_idx]

                n_samples_node = float(tree_.n_node_samples[node_id])
                n_samples_class = float(tree_.value[node_id][0][class_idx])
                n_total = float(tree_.n_node_samples[0])

                confidence = n_samples_class / n_samples_node if n_samples_node > 0 else 0.0
                support = n_samples_node / n_total if n_total > 0 else 0.0
                prevalence = self._class_prevalence.get(prediction, 0.5)
                lift = confidence / prevalence if prevalence > 0 else 1.0

                rule = Rule(
                    conditions=deepcopy(path_conditions),
                    prediction=prediction,
                    confidence=confidence,
                    support=support,
                    lift=lift,
                    rule_id=f"DT_R{len(ruleset.rules) + 1:04d}",
                )
                ruleset.add(rule)
                return

            feature = feature_names[tree_.feature[node_id]]
            threshold = float(tree_.threshold[node_id])

            # Left branch: feature <= threshold
            left_cond = Condition(feature=feature, operator="<=", threshold=threshold)
            _traverse(tree_.children_left[node_id], path_conditions + [left_cond])

            # Right branch: feature > threshold
            right_cond = Condition(feature=feature, operator=">", threshold=threshold)
            _traverse(tree_.children_right[node_id], path_conditions + [right_cond])

        _traverse(0, [])
        logger.info("Extracted %d rules from decision tree.", len(ruleset))
        return ruleset

    # ------------------------------------------------------------------
    # Rule extraction from Random Forest
    # ------------------------------------------------------------------
    def extract_from_rf(
        self,
        random_forest: RandomForestClassifier,
        feature_names: List[str],
        top_n: int = 20,
        class_names: Optional[List[str]] = None,
        X_train: Optional[np.ndarray] = None,
        y_train: Optional[np.ndarray] = None,
    ) -> RuleSet:
        """
        Extract the most common rules from a Random Forest.

        Collects all leaf-path rules from every tree, deduplicates them,
        and returns the top_n by support × confidence.

        Parameters
        ----------
        random_forest : RandomForestClassifier
        feature_names : list of str
        top_n : int
        class_names : list of str, optional
        X_train, y_train : optional — for support/confidence computation

        Returns
        -------
        RuleSet
        """
        logger.info(
            "Extracting top-%d rules from random forest (%d trees).",
            top_n,
            len(random_forest.estimators_),
        )
        if class_names is None:
            class_names = [str(c) for c in random_forest.classes_]

        all_rules: List[Rule] = []
        tmp_engine = DecisionRulesEngine()
        if y_train is not None:
            unique, counts = np.unique(y_train, return_counts=True)
            total = len(y_train)
            tmp_engine._class_prevalence = {
                class_names[i]: counts[i] / total for i, cls in enumerate(unique)
            }

        for est in random_forest.estimators_:
            rs = tmp_engine.extract_from_tree(est, feature_names, class_names, X_train, y_train)
            all_rules.extend(rs.rules)

        # Rank by lift × confidence × support and deduplicate by NL representation
        all_rules.sort(key=lambda r: r.lift * r.confidence * r.support, reverse=True)
        seen: set = set()
        unique_rules: List[Rule] = []
        for rule in all_rules:
            key = rule.to_natural_language()
            if key not in seen:
                seen.add(key)
                unique_rules.append(rule)
            if len(unique_rules) >= top_n:
                break

        # Re-number rule IDs
        for i, rule in enumerate(unique_rules, 1):
            rule.rule_id = f"RF_R{i:04d}"

        ruleset = RuleSet(
            rules=unique_rules,
            name="RandomForestRules",
            feature_names=feature_names,
            target_classes=list(class_names),
        )
        logger.info("Extracted %d unique rules from random forest.", len(ruleset))
        return ruleset

    # ------------------------------------------------------------------
    # Manual rule creation
    # ------------------------------------------------------------------
    def create_manual_rules(self, rules_dict: List[Dict]) -> RuleSet:
        """
        Create a RuleSet from a list of rule dictionaries.

        Expected format for each rule dict:
        {
            "rule_id": "R001",
            "conditions": [
                {"feature": "amount", "operator": ">", "threshold": 5000},
                {"feature": "hour", "operator": "<", "threshold": 6}
            ],
            "prediction": "HIGH_RISK",
            "confidence": 0.95,
            "support": 0.05,
            "lift": 3.2
        }

        Parameters
        ----------
        rules_dict : list of dict

        Returns
        -------
        RuleSet
        """
        logger.info("Creating %d manual rules.", len(rules_dict))
        ruleset = RuleSet(name="ManualRules")

        for i, rdict in enumerate(rules_dict):
            conditions = [
                Condition(
                    feature=c["feature"],
                    operator=c["operator"],
                    threshold=float(c["threshold"]),
                )
                for c in rdict.get("conditions", [])
            ]
            rule = Rule(
                conditions=conditions,
                prediction=rdict.get("prediction", "UNKNOWN"),
                confidence=float(rdict.get("confidence", 0.0)),
                support=float(rdict.get("support", 0.0)),
                lift=float(rdict.get("lift", 1.0)),
                rule_id=rdict.get("rule_id", f"M_R{i + 1:04d}"),
            )
            ruleset.add(rule)

        return ruleset

    # ------------------------------------------------------------------
    # Rule evaluation
    # ------------------------------------------------------------------
    def evaluate_rules(
        self,
        X: pd.DataFrame,
        rules: RuleSet,
        y: Optional[pd.Series] = None,
    ) -> Dict[str, Any]:
        """
        Apply rules to a dataset and compute coverage and accuracy per rule.

        Parameters
        ----------
        X : pd.DataFrame
        rules : RuleSet
        y : pd.Series, optional — true labels for accuracy computation

        Returns
        -------
        dict with keys:
            'per_rule': list of dicts (coverage, accuracy per rule)
            'overall_coverage': float
            'overall_accuracy': float (if y provided)
        """
        logger.info("Evaluating %d rules on %d samples.", len(rules), len(X))
        n_total = len(X)
        per_rule_results = []

        for rule in rules.rules:
            mask = X.apply(rule.evaluate, axis=1)
            covered = int(mask.sum())
            coverage = covered / n_total

            entry: Dict[str, Any] = {
                "rule_id": rule.rule_id,
                "prediction": str(rule.prediction),
                "coverage": round(coverage, 4),
                "n_covered": covered,
                "stated_confidence": round(rule.confidence, 4),
            }

            if y is not None and covered > 0:
                correct = int((y[mask].astype(str) == str(rule.prediction)).sum())
                entry["empirical_accuracy"] = round(correct / covered, 4)
            else:
                entry["empirical_accuracy"] = None

            per_rule_results.append(entry)

        # Overall coverage: at least one rule fires
        any_covered = X.apply(
            lambda row: any(r.evaluate(row) for r in rules.rules), axis=1
        )
        overall_coverage = float(any_covered.mean())

        result: Dict[str, Any] = {
            "per_rule": per_rule_results,
            "overall_coverage": round(overall_coverage, 4),
            "n_rules": len(rules),
        }

        if y is not None:
            preds = self._apply_ruleset(X, rules)
            valid_mask = preds != self.default_prediction
            if valid_mask.sum() > 0:
                overall_accuracy = float(
                    (preds[valid_mask].astype(str) == y[valid_mask].astype(str)).mean()
                )
            else:
                overall_accuracy = 0.0
            result["overall_accuracy"] = round(overall_accuracy, 4)

        logger.info(
            "Evaluation: coverage=%.2f%%, accuracy=%s",
            overall_coverage * 100,
            result.get("overall_accuracy", "N/A"),
        )
        return result

    # ------------------------------------------------------------------
    # Rule optimization
    # ------------------------------------------------------------------
    def prune_redundant_rules(self, rules: RuleSet, min_confidence: float = 0.5, min_support: float = 0.01) -> RuleSet:
        """
        Remove rules with insufficient confidence or support,
        and deduplicate overlapping rules (keep the one with higher lift).

        Parameters
        ----------
        rules : RuleSet
        min_confidence : float
        min_support : float

        Returns
        -------
        Pruned RuleSet.
        """
        logger.info("Pruning rules (min_confidence=%.2f, min_support=%.4f).", min_confidence, min_support)
        filtered = [
            r for r in rules.rules
            if r.confidence >= min_confidence and r.support >= min_support
        ]

        # Deduplicate by prediction + condition fingerprint
        seen: Dict[str, Rule] = {}
        for rule in sorted(filtered, key=lambda r: r.lift, reverse=True):
            key = str(rule.prediction) + "|" + "|".join(
                f"{c.feature}{c.operator}{c.threshold}" for c in sorted(rule.conditions, key=lambda c: c.feature)
            )
            if key not in seen:
                seen[key] = rule

        pruned_rules = list(seen.values())
        pruned_rules.sort(key=lambda r: r.lift, reverse=True)

        new_ruleset = RuleSet(
            rules=pruned_rules,
            name=rules.name + "_pruned",
            target_classes=rules.target_classes,
            feature_names=rules.feature_names,
        )
        logger.info("Rules after pruning: %d → %d.", len(rules), len(new_ruleset))
        return new_ruleset

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def _apply_ruleset(self, X: pd.DataFrame, rules: RuleSet) -> pd.Series:
        """Apply rules row-by-row with first-match conflict resolution."""
        # Sort by lift descending so highest-quality rules fire first
        sorted_rules = sorted(rules.rules, key=lambda r: r.lift, reverse=True)
        predictions = []
        for _, row in X.iterrows():
            pred = self.default_prediction
            for rule in sorted_rules:
                if rule.evaluate(row):
                    pred = rule.prediction
                    break
            predictions.append(pred)
        return pd.Series(predictions, index=X.index)

    def predict(
        self,
        X: pd.DataFrame,
        rules: Optional[RuleSet] = None,
    ) -> pd.Series:
        """
        Apply rules to predict labels for each row in X.

        Conflict resolution: rules are sorted by lift (descending);
        the first matching rule wins.  Rows matched by no rule receive
        self.default_prediction.

        Parameters
        ----------
        X : pd.DataFrame
        rules : RuleSet, optional
            If None, uses self.ruleset (set externally or by load_ruleset()).

        Returns
        -------
        pd.Series of predictions, same index as X.
        """
        ruleset = rules if rules is not None else self.ruleset
        if not ruleset.rules:
            logger.warning("No rules loaded — returning default prediction for all rows.")
            return pd.Series([self.default_prediction] * len(X), index=X.index)

        logger.info("predict() — applying %d rules to %d samples.", len(ruleset), len(X))
        return self._apply_ruleset(X, ruleset)

    # ------------------------------------------------------------------
    # Export helpers (delegate to RuleSet)
    # ------------------------------------------------------------------
    def to_dict(self, rules: Optional[RuleSet] = None) -> Dict:
        rs = rules or self.ruleset
        return rs.to_dict()

    def to_sql(self, rules: Optional[RuleSet] = None) -> str:
        rs = rules or self.ruleset
        return rs.to_sql()

    def to_json(self, rules: Optional[RuleSet] = None, path: Optional[str] = None) -> str:
        rs = rules or self.ruleset
        return rs.to_json(path=path)

    def to_natural_language(self, rules: Optional[RuleSet] = None) -> str:
        rs = rules or self.ruleset
        return rs.to_natural_language()


# ---------------------------------------------------------------------------
# Demo / entry-point
# ---------------------------------------------------------------------------
def _build_fraud_manual_rules() -> List[Dict]:
    """
    Return a list of hand-crafted fraud detection rules.

    Rule logic:
        - IF amount > 5000 AND hour < 6 → HIGH_RISK
        - IF location_mismatch == 1 AND new_device == 1 → MEDIUM_RISK
        - IF velocity > 5 → HIGH_RISK
        - IF amount < 100 AND velocity <= 2 → LOW_RISK
    """
    return [
        {
            "rule_id": "FRAUD_R001",
            "conditions": [
                {"feature": "amount", "operator": ">", "threshold": 5000},
                {"feature": "hour", "operator": "<", "threshold": 6},
            ],
            "prediction": "HIGH_RISK",
            "confidence": 0.92,
            "support": 0.04,
            "lift": 4.6,
        },
        {
            "rule_id": "FRAUD_R002",
            "conditions": [
                {"feature": "location_mismatch", "operator": "==", "threshold": 1},
                {"feature": "new_device", "operator": "==", "threshold": 1},
            ],
            "prediction": "MEDIUM_RISK",
            "confidence": 0.75,
            "support": 0.07,
            "lift": 2.8,
        },
        {
            "rule_id": "FRAUD_R003",
            "conditions": [
                {"feature": "velocity", "operator": ">", "threshold": 5},
            ],
            "prediction": "HIGH_RISK",
            "confidence": 0.88,
            "support": 0.06,
            "lift": 4.1,
        },
        {
            "rule_id": "FRAUD_R004",
            "conditions": [
                {"feature": "amount", "operator": "<", "threshold": 100},
                {"feature": "velocity", "operator": "<=", "threshold": 2},
            ],
            "prediction": "LOW_RISK",
            "confidence": 0.95,
            "support": 0.30,
            "lift": 1.2,
        },
    ]


def demo() -> None:
    """
    End-to-end demo:
    1. Generate synthetic classification dataset.
    2. Train DecisionTreeClassifier and RandomForestClassifier.
    3. Extract rules.
    4. Create manual fraud rules.
    5. Evaluate all rule sets.
    6. Export rules to JSON.
    """
    logger.info("=" * 60)
    logger.info("DecisionRulesEngine — Demo")
    logger.info("=" * 60)

    rng = np.random.default_rng(42)

    # ---- Synthetic dataset ----
    X_arr, y_arr = make_classification(
        n_samples=2000,
        n_features=10,
        n_informative=6,
        n_redundant=2,
        random_state=42,
    )
    feature_names = [f"feature_{i}" for i in range(X_arr.shape[1])]
    class_names = ["LOW_RISK", "HIGH_RISK"]
    y_arr_named = np.array([class_names[y] for y in y_arr])

    X_df = pd.DataFrame(X_arr, columns=feature_names)
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y_arr_named, test_size=0.3, random_state=42
    )

    engine = DecisionRulesEngine(default_prediction="LOW_RISK")

    # ---- Decision Tree ----
    dt = DecisionTreeClassifier(max_depth=4, random_state=42)
    dt.fit(X_train.values, y_train)
    dt_rules = engine.extract_from_tree(
        dt, feature_names, class_names=class_names,
        X_train=X_train.values, y_train=y_train,
    )
    logger.info("DT rules: %d", len(dt_rules))

    # ---- Random Forest ----
    rf = RandomForestClassifier(n_estimators=50, max_depth=4, random_state=42)
    rf.fit(X_train.values, y_train)
    rf_rules = engine.extract_from_rf(
        rf, feature_names, top_n=20, class_names=class_names,
        X_train=X_train.values, y_train=y_train,
    )
    logger.info("RF rules: %d", len(rf_rules))

    # ---- Manual fraud rules ----
    fraud_rule_dicts = _build_fraud_manual_rules()
    manual_rules = engine.create_manual_rules(fraud_rule_dicts)
    logger.info("Manual rules: %d", len(manual_rules))

    # ---- Prune DT rules ----
    pruned_dt = engine.prune_redundant_rules(dt_rules, min_confidence=0.6, min_support=0.005)

    # ---- Evaluate ----
    y_test_s = pd.Series(y_test, name="label")
    dt_eval = engine.evaluate_rules(X_test, dt_rules, y=y_test_s)
    rf_eval = engine.evaluate_rules(X_test, rf_rules, y=y_test_s)
    pruned_eval = engine.evaluate_rules(X_test, pruned_dt, y=y_test_s)
    logger.info(
        "DT rules eval → coverage=%.2f%%  accuracy=%.2f%%",
        dt_eval["overall_coverage"] * 100,
        dt_eval.get("overall_accuracy", 0) * 100,
    )
    logger.info(
        "RF rules eval → coverage=%.2f%%  accuracy=%.2f%%",
        rf_eval["overall_coverage"] * 100,
        rf_eval.get("overall_accuracy", 0) * 100,
    )

    # ---- Natural language export ----
    nl_output = engine.to_natural_language(manual_rules)
    logger.info("Manual rules (natural language):\n%s", nl_output)

    sql_output = engine.to_sql(manual_rules)
    logger.info("Manual rules (SQL):\n%s", sql_output)

    # ---- Predict ----
    preds = engine.predict(X_test, rules=dt_rules)
    logger.info("Sample predictions (first 10): %s", list(preds[:10]))

    # ---- Save ----
    output = {
        "decision_tree_rules": {
            "ruleset": dt_rules.to_dict(),
            "evaluation": dt_eval,
        },
        "random_forest_rules": {
            "ruleset": rf_rules.to_dict(),
            "evaluation": rf_eval,
        },
        "pruned_dt_rules": {
            "ruleset": pruned_dt.to_dict(),
            "evaluation": pruned_eval,
        },
        "manual_fraud_rules": {
            "ruleset": manual_rules.to_dict(),
            "natural_language": nl_output,
            "sql": sql_output,
        },
    }
    out_path = OUTPUT_DIR / "decision_rules.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Decision rules saved to %s", out_path)


if __name__ == "__main__":
    demo()
