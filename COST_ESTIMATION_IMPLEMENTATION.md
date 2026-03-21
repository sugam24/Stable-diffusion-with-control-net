# Build Cost Estimation: What We Did and How We Implemented It

This document describes the addition of **cost-to-build estimation** for the generated smart city: what we set out to do, how we implemented it, and how to use it.

---

## 1. What We Did

### Goal

- Estimate the **cost to build the new city** shown in the pipeline output.
- Use a **model and data** informed by real construction/infrastructure cost datasets.
- Expose the result as a **final output** (total cost in USD and optional breakdown).

### Constraint

There is no single public “image → build cost” model. Real-world city cost tools (e.g. AECOM Cities Cost Model) use **parametric models**: unit costs (USD/m² or per capita) × areas, with unit costs coming from industry and government datasets.

### Approach We Took

1. **Parametric cost model**  
   - Inputs: land-use **coverage** (percent per class from the segmentation mask) and **total area** (km²).  
   - Unit costs (USD/m²) per land-use class are set from published ranges (infrastructure/construction datasets).  
   - Formula: for each class, `cost = (coverage% / 100) × total_area_m² × unit_cost_USD_per_m²`; total = sum over classes.

2. **Optional regression “model”**  
   - A small **linear regression** is trained on **synthetic cost data** generated with the same parametric formula.  
   - This gives a “model trained on cost-like data” that approximates the parametric total from (coverage, area).

3. **Integration**  
   - Cost estimation runs at the end of the pipeline (or with analysis-only).  
   - Output: **total cost (USD)** printed to console and saved in a JSON file with breakdown.

---

## 2. How We Implemented It

### 2.1 New Module: `src/estimate_build_cost.py`

- **Parametric estimation**
  - `estimate_cost(coverage, total_area_km2, unit_costs)`  
  - `coverage`: dict of class name → coverage percent (same as from `compute_coverage`).  
  - `total_area_km2`: site area in km² (default 1.0).  
  - `unit_costs`: optional override; otherwise uses `DEFAULT_UNIT_COSTS_USD_PER_M2` (see below).

- **Default unit costs (USD/m²)**  
  Sourced from published construction and infrastructure datasets. **Full citations and URLs are in [COST_ESTIMATION_SOURCES.md](COST_ESTIMATION_SOURCES.md).** Summary:

  | Class              | USD/m² | Source type (see COST_ESTIMATION_SOURCES.md)     |
  |--------------------|--------|--------------------------------------------------|
  | Residential Area   | 850    | RSMeans residential costs + land dev $/acre      |
  | Road               | 220    | State DOT $/lane-mile (AR, NH), World Bank ROCKS |
  | River              | 15     | Conservative; minimal works                      |
  | Forest             | 25     | Preservation, paths, planting                    |
  | Unused Land        | 65     | CA-style landscaping unit prices ($/sq ft→m²)  |
  | Agricultural Area  | 35     | World Bank urban expansion per-hectare         |

- **Optional regression**
  - `train_synthetic_model(n_samples, seed)`  
    - Builds synthetic data: random coverage (Dirichlet) and random `area_km2`.  
    - Target = parametric total cost from `estimate_cost()`.  
    - Fits `sklearn.linear_model.LinearRegression` on features `[coverage_pct_1, …, coverage_pct_6, area_km2]`.  
  - `predict_with_model(model, coverage, total_area_km2)`  
    - Returns model prediction for total cost (USD).

- **Convenience**
  - `run_estimation(coverage, total_area_km2, use_trained_model, output_path, print_result)`  
  - Runs parametric (and optionally regression), writes JSON, and can print a short summary.

- **CLI**
  - `python src/estimate_build_cost.py <mask_path> --area_km2 1.0 [--output path] [--use_model]`  
  - Coverage is computed from the mask; if `--coverage` JSON is provided, the mask is not used for coverage.

### 2.2 Integration in `src/generate_satellite_image.py`

- **Imports**  
  - `run_estimation` is imported as `run_build_cost_estimation` from `estimate_build_cost`.

- **When cost runs**
  - If `--cost_estimate` is set:
    - **After generation:** after saving the output image, cost is run with the same `coverage` and `args.area_km2`, and the result is saved and (if verbose) printed.
    - **With `--analysis_only`:** after writing the report, cost is run and the same JSON is written.

- **CLI arguments added**
  - `--cost_estimate`  
    - Enable cost estimation; writes `<prefix>_cost_estimate.json` and prints total (USD).
  - `--area_km2` (default `1.0`)  
    - Total site area in km² for the cost formula. Used only when cost estimation is run.
  - `--cost_use_model`  
    - Also run the regression model (requires `scikit-learn`).

### 2.3 Outputs

- **File**  
  - `<output_dir>/<prefix>_cost_estimate.json`  
  - Contents: `total_cost_usd`, `total_area_m2`, `area_km2`, `breakdown` (per-class cost and area), `unit_costs_used`, `currency`, `note`.  
  - If `--cost_use_model` is used, the JSON also includes `model_prediction_usd`.

- **Console**  
  - When verbose (default): total area, total cost (USD), optional model prediction, and a short per-class breakdown.

### 2.4 Dependencies

- **`requirements.txt`**  
  - Added `scikit-learn>=1.3.0` for the optional regression.  
  - The parametric path works without it; the regression path is optional.

---

## 3. Data Sources and “Model Trained on Those Sorts of Datasets”

- **Parametric unit costs (USD/m²)**  
  - **Detailed sources:** [COST_ESTIMATION_SOURCES.md](COST_ESTIMATION_SOURCES.md).  
  - Informed by:
    - Infrastructure cost datasets (e.g. Canada Core Infrastructure Cost, US DOT highway cost indices).
    - RSMeans-style residential and road unit costs.
    - Regional unit price guides (e.g. California-style unit prices for paving, landscaping).
  - So the **formula** is “trained on” / aligned with those kinds of datasets by choosing representative (see **COST_ESTIMATION_SOURCES.md** for per-class citations and URLs) USD/m² values.

- **Regression model**  
  - Trained on **synthetic** cost data: many (coverage, area_km2) samples with targets from the **same parametric formula**.  
  - So the “model” is trained on cost-like data that reflects the same structure as real parametric city-cost models (coverage + area → total cost).

---

## 4. Usage Summary

| Use case | Command (concept) |
|----------|-------------------|
| Generate image + cost | `python src/generate_satellite_image.py --image ... --mask ... --cost_estimate [--area_km2 2.5]` |
| Analysis only + cost   | `python src/generate_satellite_image.py --image ... --mask ... --analysis_only --cost_estimate` |
| Cost with regression  | Add `--cost_use_model` to the above. |
| Standalone from mask  | `python src/estimate_build_cost.py path/to/mask.png --area_km2 1.0 --output output/cost.json` |

**Final output:** the **cost to build the new city** in USD (parametric, and optionally regression), printed and saved in `<prefix>_cost_estimate.json` with a per-class breakdown.

---

## 5. Files Touched

| File | Change |
|------|--------|
| `src/estimate_build_cost.py` | **New.** Parametric cost, optional regression, CLI, JSON output. |
| `src/generate_satellite_image.py` | Import cost runner; `--cost_estimate`, `--area_km2`, `--cost_use_model`; run cost after generation or after analysis-only; write `<prefix>_cost_estimate.json`. |
| `requirements.txt` | Added `scikit-learn>=1.3.0`. |
| `COST_ESTIMATION_IMPLEMENTATION.md` | **New.** This documentation. |
| `COST_ESTIMATION_SOURCES.md` | **New.** Authoritative sources and citations for each USD/m² unit cost. |
