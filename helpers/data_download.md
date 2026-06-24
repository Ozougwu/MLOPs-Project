# Data — Online Shoppers Purchasing Intention

**Chosen dataset (locked):** Online Shoppers Purchasing Intention (UCI). Binary classification — predict whether a web session ends in a purchase (`Revenue` = True/False).

**Why this one (mix of creativity + reliability):**
- **Creativity:** behavioural / e-commerce angle, far less overused than churn datasets; **seasonality is built in** (`Month`, `Weekend`, `VisitorType`) so the data-drift story (req #5) is *natural*, not manufactured.
- **Reliability:** one clean CSV (~12,330 rows, 18 columns), binary target, mixed numeric/categorical → fast to a green pipeline, strong SHAP, full-sample committable.
- **DFS showcase (stretch):** derive session-aggregate features per `VisitorType` / `Month` to demonstrate Featuretools/DFS thinking without a fragile multi-table join.

## Where to get it

- **UCI:** https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset
- **File:** `online_shoppers_intention.csv` (one file, ~12,330 rows, ~1 MB)
- **Mirror:** also on Kaggle (search "Online Shoppers Purchasing Intention Dataset").

Place it at `data/01_raw/online_shoppers.csv` once the Kedro project exists. For now keep it in `Main Project/`.

## Programmatic download (optional, no Kaggle account)

```python
from ucimlrepo import fetch_ucirepo

ds = fetch_ucirepo(id=468)
X = ds.data.features
y = ds.data.targets            # 'Revenue' (bool)
df = X.join(y)
```

Or with kagglehub:
```python
import kagglehub
path = kagglehub.dataset_download("henrysue/online-shoppers-intention")
```

## Load + known gotchas

```python
import pandas as pd

df = pd.read_csv("data/01_raw/online_shoppers.csv")

# 1. Target: Revenue is bool -> map to 1/0
df["Revenue"] = df["Revenue"].astype(int)

# 2. Weekend is also bool -> 1/0
df["Weekend"] = df["Weekend"].astype(int)

# 3. Month is a 3-letter string (Feb, Mar, ...) -> categorical encode; useful for the drift story
# 4. VisitorType is categorical (Returning_Visitor / New_Visitor / Other)
# 5. Class imbalance (~15% purchase) -> report recall on purchasers / F1 / ROC-AUC, NOT accuracy
```

| Gotcha | Why it matters | Fix |
|--------|----------------|-----|
| `Revenue`, `Weekend` are bool | model needs numeric | `.astype(int)` |
| `Month` is 3-letter string | needs encoding; also the natural drift axis | categorical encode (one-hot / ordinal) |
| `VisitorType` categorical | encode for model | one-hot |
| Strong imbalance (~15% buy) | accuracy misleads | optimise **recall on purchasers / F1 / ROC-AUC** |

## Drift story (req #5 — natural, not faked)

Seasonality is real here: split sessions so the **reference** window is one set of months and the **current** batch is a different season (e.g. holiday months) → Evidently flags genuine distribution shift in `Month`, `TrafficType`, page-value features. This is a stronger narrative than synthetically perturbing a column.

## Success-metric framing for the report

- **Business metric:** value of correctly flagging a buying session (target offers/retargeting) vs. cost of false positives → optimise **recall on purchasers**, cost-weighted.
- **SHAP story:** `PageValues`, `ExitRates`, `ProductRelated_Duration`, `VisitorType` typically dominate — intuitive for non-technical graders.

## Key columns

- **Target:** `Revenue` (True/False → 1/0)
- **Numeric:** `Administrative`, `Administrative_Duration`, `Informational`, `Informational_Duration`, `ProductRelated`, `ProductRelated_Duration`, `BounceRates`, `ExitRates`, `PageValues`, `SpecialDay`
- **Categorical:** `Month`, `OperatingSystems`, `Browser`, `Region`, `TrafficType`, `VisitorType`, `Weekend`
- **No explicit ID column** — create a session index for the feature-store key.
