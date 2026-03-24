"""
Optimization Module
=====================
Production-grade optimization module containing four optimizers:

1. PricingOptimizer        — LP-based profit maximisation with price elasticity
2. ResourceAllocationOptimizer — Budget allocation with diminishing returns
3. InventoryOptimizer      — EOQ + safety stock + reorder point
4. PortfolioOptimizer      — Markowitz mean-variance / efficient frontier

Author: PPA Analytics Team
"""

import json
import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

warnings.filterwarnings("ignore")

# Optional PuLP import (LP solver)
try:
    import pulp
    PULP_AVAILABLE = True
except ImportError:
    PULP_AVAILABLE = False
    logging.warning("PuLP not installed — PricingOptimizer will use scipy fallback.")

from scipy.optimize import minimize, linprog
from scipy.stats import norm

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
# Output directories
# ---------------------------------------------------------------------------
OUTPUT_DIR = Path("/home/user/ppa/outputs/metrics")
PLOTS_DIR = Path("/home/user/ppa/outputs/plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# ===========================================================================
# 1. Pricing Optimizer
# ===========================================================================
class PricingOptimizer:
    """
    Maximise profit across multiple products / customer segments using
    Linear Programming (PuLP when available, otherwise scipy.optimize).

    Demand model: Q(p) = base_demand * (p / reference_price) ^ (-elasticity)
    Profit: pi(p) = (p - unit_cost) * Q(p)

    Attributes
    ----------
    products : list of str
    base_demand : np.ndarray
    unit_costs : np.ndarray
    elasticities : np.ndarray
    reference_prices : np.ndarray
    price_min : np.ndarray
    price_max : np.ndarray
    results_ : dict  — populated after optimize()
    """

    def __init__(self):
        self.products: List[str] = []
        self.base_demand: Optional[np.ndarray] = None
        self.unit_costs: Optional[np.ndarray] = None
        self.elasticities: Optional[np.ndarray] = None
        self.reference_prices: Optional[np.ndarray] = None
        self.price_min: Optional[np.ndarray] = None
        self.price_max: Optional[np.ndarray] = None
        self.results_: Dict = {}
        self._is_fitted = False

    # ------------------------------------------------------------------
    def fit(self, data: pd.DataFrame) -> "PricingOptimizer":
        """
        Load product pricing data.

        Parameters
        ----------
        data : pd.DataFrame
            Required columns: product, base_demand, unit_cost, elasticity,
                              reference_price, price_min, price_max
        """
        required = ["product", "base_demand", "unit_cost", "elasticity",
                    "reference_price", "price_min", "price_max"]
        missing = [c for c in required if c not in data.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        self.products = list(data["product"])
        self.base_demand = data["base_demand"].values.astype(float)
        self.unit_costs = data["unit_cost"].values.astype(float)
        self.elasticities = data["elasticity"].values.astype(float)
        self.reference_prices = data["reference_price"].values.astype(float)
        self.price_min = data["price_min"].values.astype(float)
        self.price_max = data["price_max"].values.astype(float)

        self._is_fitted = True
        logger.info("PricingOptimizer fitted for %d products.", len(self.products))
        return self

    # ------------------------------------------------------------------
    def _demand(self, price: np.ndarray) -> np.ndarray:
        """Compute demand given price array using price elasticity model."""
        return self.base_demand * (price / self.reference_prices) ** (-self.elasticities)

    def _profit(self, price: np.ndarray) -> float:
        """Total profit (negative for minimisation)."""
        q = self._demand(price)
        margin = price - self.unit_costs
        return -float(np.sum(margin * q))

    def _profit_grad(self, price: np.ndarray) -> np.ndarray:
        """Analytical gradient of profit w.r.t. price."""
        q = self._demand(price)
        dq_dp = -self.elasticities * self.base_demand * (price / self.reference_prices) ** (-self.elasticities - 1) / self.reference_prices
        margin = price - self.unit_costs
        grad = -(q + margin * dq_dp)
        return grad

    # ------------------------------------------------------------------
    def optimize(self) -> Dict:
        """
        Run the pricing optimisation.

        Returns
        -------
        dict with optimal_prices, expected_revenue, expected_profit per product.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() first.")

        logger.info("PricingOptimizer.optimize() — %d products.", len(self.products))

        # Bounds: price in [price_min, price_max]
        bounds = list(zip(self.price_min, self.price_max))
        x0 = (self.price_min + self.price_max) / 2  # initial guess: midpoint

        result = minimize(
            self._profit,
            x0,
            jac=self._profit_grad,
            bounds=bounds,
            method="L-BFGS-B",
            options={"maxiter": 1000, "ftol": 1e-9},
        )

        optimal_prices = result.x
        optimal_demand = self._demand(optimal_prices)
        revenue = optimal_prices * optimal_demand
        profit = (optimal_prices - self.unit_costs) * optimal_demand

        self.results_ = {
            "products": self.products,
            "optimal_prices": optimal_prices.tolist(),
            "expected_demand": optimal_demand.tolist(),
            "expected_revenue": revenue.tolist(),
            "expected_profit": profit.tolist(),
            "total_revenue": float(revenue.sum()),
            "total_profit": float(profit.sum()),
            "optimisation_success": bool(result.success),
        }
        logger.info(
            "Pricing optimisation complete: total_profit=%.2f, total_revenue=%.2f",
            self.results_["total_profit"],
            self.results_["total_revenue"],
        )
        return self.results_

    # ------------------------------------------------------------------
    def get_recommendations(self) -> List[str]:
        """Return human-readable pricing recommendations."""
        if not self.results_:
            raise RuntimeError("Call optimize() first.")
        recs = []
        for prod, price, profit in zip(
            self.results_["products"],
            self.results_["optimal_prices"],
            self.results_["expected_profit"],
        ):
            recs.append(
                f"Set {prod} price to ${price:.2f} — expected profit ${profit:,.2f}."
            )
        recs.append(
            f"Total expected profit: ${self.results_['total_profit']:,.2f} "
            f"(revenue: ${self.results_['total_revenue']:,.2f})."
        )
        return recs

    # ------------------------------------------------------------------
    def plot_results(self, output_path: Optional[str] = None) -> str:
        """Generate bar chart of optimal prices and expected profits."""
        if not self.results_:
            raise RuntimeError("Call optimize() first.")

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        products = self.results_["products"]
        x = np.arange(len(products))

        axes[0].bar(x, self.results_["optimal_prices"], color="steelblue")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(products, rotation=20)
        axes[0].set_title("Optimal Prices per Product")
        axes[0].set_ylabel("Price ($)")

        axes[1].bar(x, self.results_["expected_profit"], color="seagreen")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(products, rotation=20)
        axes[1].set_title("Expected Profit per Product")
        axes[1].set_ylabel("Profit ($)")

        plt.suptitle("PricingOptimizer Results", fontsize=14, fontweight="bold")
        plt.tight_layout()

        if output_path is None:
            output_path = str(PLOTS_DIR / "pricing_optimization.png")
        plt.savefig(output_path, dpi=120, bbox_inches="tight")
        plt.close()
        logger.info("Pricing plot saved to %s", output_path)
        return output_path


# ===========================================================================
# 2. Resource Allocation Optimizer
# ===========================================================================
class ResourceAllocationOptimizer:
    """
    Allocates a marketing budget across channels using a diminishing-returns
    (logarithmic) response model to maximise total conversions.

    Model: conversions_i(b_i) = scale_i * log(1 + b_i / saturation_i)
    Constraint: sum(b_i) <= total_budget, b_i >= 0
    """

    DEFAULT_CHANNELS = ["Email", "Social", "PPC", "TV", "Direct_Mail"]
    DEFAULT_SCALE = [2000, 5000, 8000, 15000, 1200]       # max conversions
    DEFAULT_SATURATION = [5_000, 20_000, 30_000, 200_000, 8_000]  # $ at 50% sat

    def __init__(self):
        self.channels: List[str] = []
        self.scale: Optional[np.ndarray] = None
        self.saturation: Optional[np.ndarray] = None
        self.total_budget: float = 0.0
        self.results_: Dict = {}
        self._is_fitted = False

    # ------------------------------------------------------------------
    def fit(self, data: Optional[pd.DataFrame] = None, total_budget: float = 100_000) -> "ResourceAllocationOptimizer":
        """
        Load channel data and budget.

        Parameters
        ----------
        data : pd.DataFrame, optional
            Columns: channel, scale, saturation.  If None, uses defaults.
        total_budget : float
        """
        if data is not None:
            self.channels = list(data["channel"])
            self.scale = data["scale"].values.astype(float)
            self.saturation = data["saturation"].values.astype(float)
        else:
            self.channels = self.DEFAULT_CHANNELS
            self.scale = np.array(self.DEFAULT_SCALE, dtype=float)
            self.saturation = np.array(self.DEFAULT_SATURATION, dtype=float)

        self.total_budget = float(total_budget)
        self._is_fitted = True
        logger.info(
            "ResourceAllocationOptimizer fitted: %d channels, budget=$%.0f.",
            len(self.channels), self.total_budget,
        )
        return self

    # ------------------------------------------------------------------
    def _conversions(self, budget_alloc: np.ndarray) -> float:
        """Total conversions from a given budget allocation (negative for min)."""
        return -float(np.sum(self.scale * np.log1p(budget_alloc / self.saturation)))

    def _gradient(self, budget_alloc: np.ndarray) -> np.ndarray:
        return -(self.scale / (self.saturation + budget_alloc))

    # ------------------------------------------------------------------
    def optimize(self) -> Dict:
        """
        Solve the budget allocation problem.

        Returns
        -------
        dict with channel allocations, expected conversions, ROI.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() first.")

        n = len(self.channels)
        x0 = np.full(n, self.total_budget / n)

        # Constraint: sum(b) == total_budget
        constraints = {"type": "eq", "fun": lambda b: np.sum(b) - self.total_budget}
        bounds = [(0, self.total_budget) for _ in range(n)]

        result = minimize(
            self._conversions,
            x0,
            jac=self._gradient,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-9},
        )

        allocation = result.x
        conversions = self.scale * np.log1p(allocation / self.saturation)
        roi = np.where(allocation > 0, conversions / allocation * 100, 0.0)

        # Equal-split baseline for comparison
        base = np.full(n, self.total_budget / n)
        base_conv = self.scale * np.log1p(base / self.saturation)

        self.results_ = {
            "channels": self.channels,
            "optimal_allocation": allocation.tolist(),
            "expected_conversions": conversions.tolist(),
            "roi_per_dollar_pct": roi.tolist(),
            "total_conversions": float(conversions.sum()),
            "total_budget": self.total_budget,
            "baseline_conversions": float(base_conv.sum()),
            "lift_over_equal_split": float((conversions.sum() - base_conv.sum()) / base_conv.sum() * 100),
        }
        logger.info(
            "Budget allocation: total_conversions=%.0f  lift=+%.1f%%",
            self.results_["total_conversions"],
            self.results_["lift_over_equal_split"],
        )
        return self.results_

    # ------------------------------------------------------------------
    def get_recommendations(self) -> List[str]:
        if not self.results_:
            raise RuntimeError("Call optimize() first.")
        recs = []
        for ch, alloc, conv, roi in zip(
            self.results_["channels"],
            self.results_["optimal_allocation"],
            self.results_["expected_conversions"],
            self.results_["roi_per_dollar_pct"],
        ):
            recs.append(
                f"Allocate ${alloc:,.0f} to {ch}: ~{conv:,.0f} conversions (ROI {roi:.2f}% per $)."
            )
        recs.append(
            f"Total conversions: {self.results_['total_conversions']:,.0f} "
            f"(+{self.results_['lift_over_equal_split']:.1f}% vs equal split)."
        )
        return recs

    # ------------------------------------------------------------------
    def plot_results(self, output_path: Optional[str] = None) -> str:
        if not self.results_:
            raise RuntimeError("Call optimize() first.")

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        ch = self.results_["channels"]
        x = np.arange(len(ch))

        axes[0].bar(x, self.results_["optimal_allocation"], color="darkorange")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(ch, rotation=20)
        axes[0].set_title("Optimal Budget Allocation")
        axes[0].set_ylabel("Budget ($)")

        axes[1].bar(x, self.results_["expected_conversions"], color="teal")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(ch, rotation=20)
        axes[1].set_title("Expected Conversions per Channel")
        axes[1].set_ylabel("Conversions")

        plt.suptitle("Resource Allocation Optimization", fontsize=14, fontweight="bold")
        plt.tight_layout()

        if output_path is None:
            output_path = str(PLOTS_DIR / "resource_allocation.png")
        plt.savefig(output_path, dpi=120, bbox_inches="tight")
        plt.close()
        logger.info("Resource allocation plot saved to %s", output_path)
        return output_path


# ===========================================================================
# 3. Inventory Optimizer
# ===========================================================================
class InventoryOptimizer:
    """
    Classical EOQ-based inventory optimisation.

    For each SKU, computes:
    - Economic Order Quantity (EOQ)
    - Safety stock (z * sigma_demand * lead_time_std)
    - Reorder Point (ROP) = demand_during_lead_time + safety_stock
    - Total annual inventory cost (holding + ordering)
    """

    def __init__(self):
        self.skus: List[str] = []
        self.annual_demand: Optional[np.ndarray] = None
        self.order_cost: Optional[np.ndarray] = None
        self.holding_cost_rate: Optional[np.ndarray] = None
        self.unit_cost: Optional[np.ndarray] = None
        self.lead_time_days: Optional[np.ndarray] = None
        self.daily_demand_std: Optional[np.ndarray] = None
        self.service_level: float = 0.95
        self.results_: Dict = {}
        self._is_fitted = False

    # ------------------------------------------------------------------
    def fit(self, data: pd.DataFrame, service_level: float = 0.95) -> "InventoryOptimizer":
        """
        Parameters
        ----------
        data : pd.DataFrame
            Required columns: sku, annual_demand, order_cost, holding_cost_rate,
                              unit_cost, lead_time_days, daily_demand_std
        service_level : float
            Target in-stock service level (default 0.95).
        """
        required = ["sku", "annual_demand", "order_cost", "holding_cost_rate",
                    "unit_cost", "lead_time_days", "daily_demand_std"]
        missing = [c for c in required if c not in data.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        self.skus = list(data["sku"])
        self.annual_demand = data["annual_demand"].values.astype(float)
        self.order_cost = data["order_cost"].values.astype(float)
        self.holding_cost_rate = data["holding_cost_rate"].values.astype(float)
        self.unit_cost = data["unit_cost"].values.astype(float)
        self.lead_time_days = data["lead_time_days"].values.astype(float)
        self.daily_demand_std = data["daily_demand_std"].values.astype(float)
        self.service_level = service_level

        self._is_fitted = True
        logger.info("InventoryOptimizer fitted for %d SKUs.", len(self.skus))
        return self

    # ------------------------------------------------------------------
    def optimize(self) -> Dict:
        """
        Compute EOQ, safety stock, ROP, and total annual cost for each SKU.

        Returns
        -------
        dict with per-SKU results.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() first.")

        logger.info("InventoryOptimizer.optimize() — %d SKUs.", len(self.skus))

        holding_cost = self.holding_cost_rate * self.unit_cost  # $ per unit per year

        # EOQ = sqrt(2 * D * S / H)
        eoq = np.sqrt(2 * self.annual_demand * self.order_cost / holding_cost)

        # Safety stock: z * sigma_demand_LT
        z = norm.ppf(self.service_level)
        sigma_lt = self.daily_demand_std * np.sqrt(self.lead_time_days)
        safety_stock = z * sigma_lt

        # Reorder point
        daily_demand = self.annual_demand / 365
        demand_during_lt = daily_demand * self.lead_time_days
        rop = demand_during_lt + safety_stock

        # Total annual cost
        n_orders = self.annual_demand / eoq
        total_ordering_cost = n_orders * self.order_cost
        avg_inventory = eoq / 2 + safety_stock
        total_holding_cost = avg_inventory * holding_cost
        total_cost = total_ordering_cost + total_holding_cost

        self.results_ = {
            "skus": self.skus,
            "eoq": eoq.tolist(),
            "safety_stock": safety_stock.tolist(),
            "reorder_point": rop.tolist(),
            "n_orders_per_year": n_orders.tolist(),
            "total_ordering_cost": total_ordering_cost.tolist(),
            "total_holding_cost": total_holding_cost.tolist(),
            "total_annual_cost": total_cost.tolist(),
            "grand_total_cost": float(total_cost.sum()),
            "service_level": self.service_level,
        }
        logger.info(
            "Inventory optimisation complete: grand_total_cost=$%.2f",
            self.results_["grand_total_cost"],
        )
        return self.results_

    # ------------------------------------------------------------------
    def get_recommendations(self) -> List[str]:
        if not self.results_:
            raise RuntimeError("Call optimize() first.")
        recs = []
        for sku, eoq, ss, rop, n_ord, cost in zip(
            self.results_["skus"],
            self.results_["eoq"],
            self.results_["safety_stock"],
            self.results_["reorder_point"],
            self.results_["n_orders_per_year"],
            self.results_["total_annual_cost"],
        ):
            recs.append(
                f"{sku}: order {eoq:.0f} units ({n_ord:.1f}x/year), "
                f"safety stock={ss:.0f}, ROP={rop:.0f}, annual cost=${cost:,.2f}."
            )
        recs.append(f"Grand total annual inventory cost: ${self.results_['grand_total_cost']:,.2f}.")
        return recs

    # ------------------------------------------------------------------
    def plot_results(self, output_path: Optional[str] = None) -> str:
        if not self.results_:
            raise RuntimeError("Call optimize() first.")

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        skus = self.results_["skus"]
        x = np.arange(len(skus))

        axes[0].bar(x, self.results_["eoq"], label="EOQ", color="royalblue")
        axes[0].bar(x, self.results_["safety_stock"], bottom=self.results_["eoq"], label="Safety Stock", color="tomato", alpha=0.7)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(skus, rotation=20)
        axes[0].set_title("EOQ + Safety Stock per SKU")
        axes[0].set_ylabel("Units")
        axes[0].legend()

        axes[1].bar(x, self.results_["total_annual_cost"], color="goldenrod")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(skus, rotation=20)
        axes[1].set_title("Total Annual Inventory Cost per SKU")
        axes[1].set_ylabel("Cost ($)")

        plt.suptitle("Inventory Optimization Results", fontsize=14, fontweight="bold")
        plt.tight_layout()

        if output_path is None:
            output_path = str(PLOTS_DIR / "inventory_optimization.png")
        plt.savefig(output_path, dpi=120, bbox_inches="tight")
        plt.close()
        logger.info("Inventory plot saved to %s", output_path)
        return output_path


# ===========================================================================
# 4. Portfolio Optimizer
# ===========================================================================
class PortfolioOptimizer:
    """
    Markowitz Mean-Variance Portfolio Optimisation.

    Computes:
    - Efficient frontier (range of risk/return trade-offs)
    - Maximum Sharpe ratio portfolio
    - Minimum variance portfolio

    Uses historical returns data (daily or monthly returns DataFrame).
    """

    def __init__(self, risk_free_rate: float = 0.04):
        """
        Parameters
        ----------
        risk_free_rate : float
            Annual risk-free rate (default 4%).
        """
        self.risk_free_rate = risk_free_rate
        self.returns: Optional[pd.DataFrame] = None
        self.mu: Optional[np.ndarray] = None       # annualised mean returns
        self.cov: Optional[np.ndarray] = None      # annualised covariance
        self.assets: List[str] = []
        self.results_: Dict = {}
        self._is_fitted = False

    # ------------------------------------------------------------------
    def fit(self, data: pd.DataFrame, freq: str = "daily") -> "PortfolioOptimizer":
        """
        Load historical returns.

        Parameters
        ----------
        data : pd.DataFrame
            Columns = asset tickers, rows = time periods, values = returns (not prices).
        freq : str
            'daily' (annualise by 252) or 'monthly' (annualise by 12).
        """
        self.returns = data.copy()
        self.assets = list(data.columns)
        scale = 252 if freq == "daily" else 12

        self.mu = data.mean().values * scale
        self.cov = data.cov().values * scale

        self._is_fitted = True
        logger.info(
            "PortfolioOptimizer fitted: %d assets, annualisation=%dx.",
            len(self.assets), scale,
        )
        return self

    # ------------------------------------------------------------------
    def _portfolio_return(self, w: np.ndarray) -> float:
        return float(w @ self.mu)

    def _portfolio_variance(self, w: np.ndarray) -> float:
        return float(w @ self.cov @ w)

    def _portfolio_std(self, w: np.ndarray) -> float:
        return float(np.sqrt(self._portfolio_variance(w)))

    def _neg_sharpe(self, w: np.ndarray) -> float:
        ret = self._portfolio_return(w)
        std = self._portfolio_std(w)
        if std < 1e-10:
            return 0.0
        return -(ret - self.risk_free_rate) / std

    # ------------------------------------------------------------------
    def _optimize(
        self,
        objective,
        constraints: List[Dict],
        bounds=None,
        x0: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        n = len(self.assets)
        if x0 is None:
            x0 = np.ones(n) / n
        if bounds is None:
            bounds = [(0.0, 1.0)] * n

        result = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-9},
        )
        return result.x

    def _standard_constraints(self) -> List[Dict]:
        return [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    # ------------------------------------------------------------------
    def optimize(self, n_frontier: int = 50) -> Dict:
        """
        Compute efficient frontier, max-Sharpe, and min-variance portfolios.

        Parameters
        ----------
        n_frontier : int
            Number of points on the efficient frontier.

        Returns
        -------
        dict with portfolios and frontier data.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() first.")

        logger.info("PortfolioOptimizer.optimize() — %d assets.", len(self.assets))
        constraints = self._standard_constraints()

        # ---- Max Sharpe ----
        w_sharpe = self._optimize(self._neg_sharpe, constraints)
        ret_sharpe = self._portfolio_return(w_sharpe)
        std_sharpe = self._portfolio_std(w_sharpe)
        sharpe_ratio = (ret_sharpe - self.risk_free_rate) / std_sharpe

        # ---- Min Variance ----
        w_minvar = self._optimize(self._portfolio_variance, constraints)
        ret_minvar = self._portfolio_return(w_minvar)
        std_minvar = self._portfolio_std(w_minvar)

        # ---- Efficient Frontier ----
        ret_min = ret_minvar
        ret_max = float(self.mu.max())
        target_returns = np.linspace(ret_min, ret_max, n_frontier)
        frontier_std = []
        frontier_ret = []

        for target in target_returns:
            constr = constraints + [
                {"type": "eq", "fun": lambda w, t=target: self._portfolio_return(w) - t}
            ]
            w = self._optimize(self._portfolio_variance, constr)
            frontier_std.append(self._portfolio_std(w))
            frontier_ret.append(self._portfolio_return(w))

        self.results_ = {
            "assets": self.assets,
            "max_sharpe_portfolio": {
                "weights": {a: round(float(wt), 4) for a, wt in zip(self.assets, w_sharpe)},
                "expected_return": round(ret_sharpe, 4),
                "volatility": round(std_sharpe, 4),
                "sharpe_ratio": round(sharpe_ratio, 4),
            },
            "min_variance_portfolio": {
                "weights": {a: round(float(wt), 4) for a, wt in zip(self.assets, w_minvar)},
                "expected_return": round(ret_minvar, 4),
                "volatility": round(std_minvar, 4),
                "sharpe_ratio": round((ret_minvar - self.risk_free_rate) / std_minvar, 4),
            },
            "efficient_frontier": {
                "volatility": [round(s, 4) for s in frontier_std],
                "return": [round(r, 4) for r in frontier_ret],
            },
            "risk_free_rate": self.risk_free_rate,
        }
        logger.info(
            "Portfolio optimisation: max_sharpe=%.3f  min_vol=%.3f",
            sharpe_ratio, std_minvar,
        )
        return self.results_

    # ------------------------------------------------------------------
    def get_recommendations(self) -> List[str]:
        if not self.results_:
            raise RuntimeError("Call optimize() first.")
        ms = self.results_["max_sharpe_portfolio"]
        mv = self.results_["min_variance_portfolio"]
        recs = [
            "=== Maximum Sharpe Ratio Portfolio ===",
            f"  Expected Return: {ms['expected_return']:.2%}",
            f"  Volatility:      {ms['volatility']:.2%}",
            f"  Sharpe Ratio:    {ms['sharpe_ratio']:.3f}",
            "  Weights:",
        ]
        for asset, wt in ms["weights"].items():
            recs.append(f"    {asset}: {wt:.1%}")

        recs += [
            "",
            "=== Minimum Variance Portfolio ===",
            f"  Expected Return: {mv['expected_return']:.2%}",
            f"  Volatility:      {mv['volatility']:.2%}",
            f"  Sharpe Ratio:    {mv['sharpe_ratio']:.3f}",
            "  Weights:",
        ]
        for asset, wt in mv["weights"].items():
            recs.append(f"    {asset}: {wt:.1%}")
        return recs

    # ------------------------------------------------------------------
    def plot_results(self, output_path: Optional[str] = None) -> str:
        if not self.results_:
            raise RuntimeError("Call optimize() first.")

        ef = self.results_["efficient_frontier"]
        ms = self.results_["max_sharpe_portfolio"]
        mv = self.results_["min_variance_portfolio"]

        fig, ax = plt.subplots(figsize=(10, 7))

        # Efficient frontier
        ax.plot(ef["volatility"], ef["return"], "b-", linewidth=2, label="Efficient Frontier")

        # Individual assets
        for i, asset in enumerate(self.assets):
            ax.scatter(
                np.sqrt(self.cov[i, i]),
                self.mu[i],
                s=80, zorder=5, label=asset,
            )

        # Max Sharpe
        ax.scatter(
            ms["volatility"], ms["expected_return"],
            marker="*", s=250, color="gold", zorder=6, label=f"Max Sharpe ({ms['sharpe_ratio']:.2f})",
        )

        # Min Variance
        ax.scatter(
            mv["volatility"], mv["expected_return"],
            marker="D", s=100, color="red", zorder=6, label="Min Variance",
        )

        ax.set_xlabel("Annual Volatility (Std Dev)")
        ax.set_ylabel("Annual Expected Return")
        ax.set_title("Markowitz Efficient Frontier")
        ax.legend(loc="upper left", fontsize=8)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
        plt.tight_layout()

        if output_path is None:
            output_path = str(PLOTS_DIR / "portfolio_efficient_frontier.png")
        plt.savefig(output_path, dpi=120, bbox_inches="tight")
        plt.close()
        logger.info("Portfolio plot saved to %s", output_path)
        return output_path


# ===========================================================================
# Demo / entry-point
# ===========================================================================
def demo() -> None:
    """
    Run all four optimizers on synthetic data and save results.
    """
    logger.info("=" * 60)
    logger.info("Optimization Module — Demo")
    logger.info("=" * 60)

    rng = np.random.default_rng(42)
    all_results: Dict[str, Any] = {}

    # ---- 1. Pricing Optimizer ----
    logger.info("--- 1. PricingOptimizer ---")
    pricing_data = pd.DataFrame({
        "product": ["ProductA", "ProductB", "ProductC", "ProductD"],
        "base_demand": [10000, 5000, 8000, 3000],
        "unit_cost": [15.0, 25.0, 8.0, 50.0],
        "elasticity": [1.5, 1.2, 2.0, 0.9],
        "reference_price": [30.0, 50.0, 20.0, 100.0],
        "price_min": [18.0, 28.0, 10.0, 60.0],
        "price_max": [50.0, 80.0, 35.0, 150.0],
    })
    pricing_opt = PricingOptimizer()
    pricing_opt.fit(pricing_data)
    pricing_results = pricing_opt.optimize()
    pricing_opt.plot_results()
    for rec in pricing_opt.get_recommendations():
        logger.info("  %s", rec)
    all_results["pricing"] = pricing_results

    # ---- 2. Resource Allocation Optimizer ----
    logger.info("--- 2. ResourceAllocationOptimizer ---")
    ra_opt = ResourceAllocationOptimizer()
    ra_opt.fit(total_budget=150_000)
    ra_results = ra_opt.optimize()
    ra_opt.plot_results()
    for rec in ra_opt.get_recommendations():
        logger.info("  %s", rec)
    all_results["resource_allocation"] = ra_results

    # ---- 3. Inventory Optimizer ----
    logger.info("--- 3. InventoryOptimizer ---")
    inventory_data = pd.DataFrame({
        "sku": ["SKU_A", "SKU_B", "SKU_C", "SKU_D", "SKU_E"],
        "annual_demand": [12000, 5000, 8000, 2000, 20000],
        "order_cost": [100.0, 80.0, 60.0, 120.0, 90.0],
        "holding_cost_rate": [0.20, 0.25, 0.18, 0.22, 0.15],
        "unit_cost": [50.0, 120.0, 30.0, 200.0, 15.0],
        "lead_time_days": [7, 14, 5, 21, 3],
        "daily_demand_std": [8.0, 3.0, 5.0, 1.5, 15.0],
    })
    inv_opt = InventoryOptimizer()
    inv_opt.fit(inventory_data, service_level=0.95)
    inv_results = inv_opt.optimize()
    inv_opt.plot_results()
    for rec in inv_opt.get_recommendations():
        logger.info("  %s", rec)
    all_results["inventory"] = inv_results

    # ---- 4. Portfolio Optimizer ----
    logger.info("--- 4. PortfolioOptimizer ---")
    # Generate synthetic daily returns for 5 assets over 3 years
    assets = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
    n_days = 756  # ~3 years
    annual_returns = np.array([0.15, 0.12, 0.14, 0.10, 0.25])
    annual_vols = np.array([0.25, 0.22, 0.20, 0.28, 0.45])
    corr_matrix = np.array([
        [1.0, 0.6, 0.65, 0.55, 0.3],
        [0.6, 1.0, 0.70, 0.60, 0.25],
        [0.65, 0.70, 1.0, 0.58, 0.28],
        [0.55, 0.60, 0.58, 1.0, 0.22],
        [0.3, 0.25, 0.28, 0.22, 1.0],
    ])
    daily_vols = annual_vols / np.sqrt(252)
    cov_matrix = np.outer(daily_vols, daily_vols) * corr_matrix
    L = np.linalg.cholesky(cov_matrix)
    z = rng.standard_normal((n_days, len(assets)))
    daily_ret = z @ L.T + annual_returns / 252
    returns_df = pd.DataFrame(daily_ret, columns=assets)

    port_opt = PortfolioOptimizer(risk_free_rate=0.04)
    port_opt.fit(returns_df, freq="daily")
    port_results = port_opt.optimize(n_frontier=40)
    port_opt.plot_results()
    for rec in port_opt.get_recommendations():
        logger.info("  %s", rec)
    all_results["portfolio"] = port_results

    # ---- Save ----
    out_path = OUTPUT_DIR / "optimization_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("Optimization results saved to %s", out_path)


if __name__ == "__main__":
    demo()
