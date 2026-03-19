"""
Estimate the cost to build the "new city" from land-use coverage.

Uses a parametric cost model based on published unit costs from infrastructure
and construction datasets (e.g. Canada Core Infrastructure, RSMeans-style
residential/road costs, DOT highway cost indices, AECOM-style city cost breakdowns).
Optionally uses a simple regression model trained on synthetic cost data
generated from the same parametric formula.

Inputs: coverage (percent per land-use class), total area in km².
Output: total cost (USD), breakdown by class, and optional JSON/print.
"""

from __future__ import annotations

import json
import os
import argparse
from pathlib import Path
from typing import Dict, Optional, Any

# Unit costs in USD per m². Sources (see COST_ESTIMATION_SOURCES.md or docstring below):
# - Residential: RSMeans residential $/sq ft + land dev (per acre → m²); blended to land coverage.
# - Road: State DOT cost per lane-mile → m² (e.g. Arkansas DOT, NH DOT); World Bank ROCKS.
# - Unused/parks: Landscaping/irrigation unit prices (e.g. California-style $/sq ft → m²).
# - Agricultural: Minimal development; World Bank urban expansion per-hectare studies.
# - River/Forest: Preservation, banks, planting (conservative).
UNIT_COST_SOURCES = (
    "Residential: RSMeans (residential costs), land dev $/acre. Road: State DOT (AR, NH) per lane-mile, "
    "World Bank ROCKS. Parks: CA unit prices (landscaping). Agri: World Bank urban expansion. "
    "See COST_ESTIMATION_SOURCES.md for full citations."
)
DEFAULT_UNIT_COSTS_USD_PER_M2: Dict[str, float] = {
    "Residential Area": 850.0,   # RSMeans-style + land dev; mid-range prorated to land m²
    "Road": 220.0,               # DOT paving+base ~$180-480/m²; World Bank ROCKS
    "River": 15.0,               # banks, minimal works
    "Forest": 25.0,              # preservation, paths, planting
    "Unused Land": 65.0,         # parks, landscaping (CA-style unit prices)
    "Agricultural Area": 35.0,   # precision ag, drainage (conservative)
}


def estimate_cost(
    coverage: Dict[str, float],
    total_area_km2: float = 1.0,
    unit_costs: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Estimate total build cost from land-use coverage and total area.

    Cost per class = (coverage_pct/100) * total_area_m2 * unit_cost_usd_per_m2.
    Total = sum over all classes.

    Args:
        coverage: Dict mapping class name -> coverage percentage (0-100).
        total_area_km2: Total area of the site in km².
        unit_costs: Optional override for USD per m² per class; else uses DEFAULT_UNIT_COSTS_USD_PER_M2.

    Returns:
        Dict with keys: total_cost_usd, total_area_m2, area_km2, breakdown (per-class cost and area),
        unit_costs_used, currency, note.
    """
    unit_costs = unit_costs or DEFAULT_UNIT_COSTS_USD_PER_M2
    total_area_m2 = total_area_km2 * 1e6
    breakdown: Dict[str, Dict[str, float]] = {}
    total_cost_usd = 0.0

    for class_name, pct in coverage.items():
        pct = float(pct)
        if pct <= 0:
            continue
        cost_per_m2 = unit_costs.get(class_name, 0.0)
        area_m2 = total_area_m2 * (pct / 100.0)
        cost_usd = area_m2 * cost_per_m2
        breakdown[class_name] = {
            "coverage_percent": round(pct, 2),
            "area_m2": round(area_m2, 2),
            "unit_cost_usd_per_m2": cost_per_m2,
            "cost_usd": round(cost_usd, 2),
        }
        total_cost_usd += cost_usd

    return {
        "total_cost_usd": round(total_cost_usd, 2),
        "total_area_m2": round(total_area_m2, 2),
        "area_km2": total_area_km2,
        "breakdown": breakdown,
        "unit_costs_used": dict(unit_costs),
        "currency": "USD",
        "note": "Parametric estimate from land-use coverage and published unit cost ranges (infrastructure/construction datasets).",
        "unit_cost_sources": "See COST_ESTIMATION_SOURCES.md for citations: RSMeans, State DOT (AR/NH), World Bank ROCKS, CA unit prices, World Bank urban expansion.",
    }


def train_synthetic_model(n_samples: int = 500, seed: int = 42) -> Any:
    """
    Train a simple linear regression on synthetic cost data for optional use.

    Synthetic data: random coverage vectors (sum=100) and random area_km2;
    target = parametric total_cost from estimate_cost(). The model learns
    to approximate the same formula from (coverage_pct_1, ..., coverage_pct_6, area_km2).

    Returns:
        Fitted sklearn LinearRegression (if sklearn available) or None.
    """
    try:
        import numpy as np
        from sklearn.linear_model import LinearRegression
    except ImportError:
        return None

    rng = np.random.default_rng(seed)
    # Class names in fixed order for features
    class_names = list(DEFAULT_UNIT_COSTS_USD_PER_M2.keys())
    n_classes = len(class_names)
    X_list = []
    y_list = []

    for _ in range(n_samples):
        # Random area in km²
        area_km2 = float(rng.uniform(0.2, 10.0))
        # Random coverage that sums to 100 (Dirichlet)
        coverage_pcts = rng.dirichlet(np.ones(n_classes)).tolist()
        coverage = {name: pcts * 100 for name, pcts in zip(class_names, coverage_pcts)}
        cost_result = estimate_cost(coverage, total_area_km2=area_km2)
        target = cost_result["total_cost_usd"]
        # Features: coverage per class (percent) + area_km2
        feats = coverage_pcts + [area_km2]
        X_list.append(feats)
        y_list.append(target)

    X = np.array(X_list)
    y = np.array(y_list)
    model = LinearRegression().fit(X, y)
    model.class_names_ = class_names  # store for predict
    return model


def predict_with_model(
    model: Any,
    coverage: Dict[str, float],
    total_area_km2: float,
) -> Optional[float]:
    """
    Predict total cost using a trained regression model.

    Args:
        model: Fitted model from train_synthetic_model() (must have class_names_).
        coverage: Class name -> coverage percent.
        total_area_km2: Area in km².

    Returns:
        Predicted total_cost_usd or None if model invalid.
    """
    if model is None or not hasattr(model, "class_names_"):
        return None
    try:
        import numpy as np
    except ImportError:
        return None
    class_names = list(model.class_names_)
    pcts = [coverage.get(c, 0.0) / 100.0 for c in class_names]
    feats = np.array([pcts + [total_area_km2]])
    pred = model.predict(feats)
    return float(pred[0])


def run_estimation(
    coverage: Dict[str, float],
    total_area_km2: float = 1.0,
    use_trained_model: bool = False,
    output_path: Optional[str] = None,
    print_result: bool = True,
) -> Dict[str, Any]:
    """
    Run cost estimation and optionally save to JSON and print.

    Args:
        coverage: Land-use coverage percentages.
        total_area_km2: Total area in km².
        use_trained_model: If True, also produce a regression-based estimate (trained on synthetic data).
        output_path: If set, write result JSON here.
        print_result: If True, print a short summary to stdout.

    Returns:
        Result dict from estimate_cost(), with optional key "model_prediction_usd" if use_trained_model.
    """
    result = estimate_cost(coverage, total_area_km2=total_area_km2)

    if use_trained_model:
        model = train_synthetic_model()
        if model is not None:
            pred = predict_with_model(model, coverage, total_area_km2)
            if pred is not None:
                result["model_prediction_usd"] = round(pred, 2)
                result["note"] = (
                    result.get("note", "")
                    + " Optional regression model trained on synthetic cost data (same parametric formula)."
                )

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        if print_result:
            print(f"Cost estimate saved to: {output_path}")

    if print_result:
        print("\n" + "=" * 60)
        print("  BUILD COST ESTIMATE (parametric)")
        print("=" * 60)
        print(f"  Total area: {result['area_km2']} km² ({result['total_area_m2']:,.0f} m²)")
        print(f"  Total cost: {result['currency']} {result['total_cost_usd']:,.2f}")
        if result.get("model_prediction_usd") is not None:
            print(f"  Model prediction: {result['currency']} {result['model_prediction_usd']:,.2f}")
        print("  Breakdown:")
        for class_name, b in result.get("breakdown", {}).items():
            if b.get("cost_usd", 0) > 0:
                print(f"    - {class_name}: {b['cost_usd']:,.2f} {result['currency']} ({b['coverage_percent']:.1f}%)")
        print("=" * 60 + "\n")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Estimate build cost from land-use coverage (e.g. from compute_coverage or pipeline)."
    )
    parser.add_argument(
        "mask_path",
        type=str,
        nargs="?",
        default=None,
        help="Path to segmentation mask; coverage will be computed from it. If omitted, use --coverage and --area_km2.",
    )
    parser.add_argument(
        "--area_km2",
        type=float,
        default=1.0,
        help="Total area of the site in km² (default 1.0).",
    )
    parser.add_argument(
        "--coverage",
        type=str,
        default=None,
        help="JSON dict of class name -> coverage percent. If provided, mask_path is ignored for coverage.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save cost estimate JSON.",
    )
    parser.add_argument(
        "--use_model",
        action="store_true",
        help="Also run regression model trained on synthetic cost data.",
    )
    args = parser.parse_args()

    if args.coverage:
        import json as _json
        coverage = _json.loads(args.coverage)
    elif args.mask_path:
        from compute_coverage import compute_coverage
        mask_path = args.mask_path
        if not os.path.isabs(mask_path):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            mask_path = os.path.join(script_dir, mask_path)
        coverage = compute_coverage(mask_path)
    else:
        print("Error: provide mask_path or --coverage JSON.")
        return 1

    run_estimation(
        coverage,
        total_area_km2=args.area_km2,
        use_trained_model=args.use_model,
        output_path=args.output,
        print_result=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
