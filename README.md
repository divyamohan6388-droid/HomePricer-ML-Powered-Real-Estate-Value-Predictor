# 🏠 HomePricer: ML-Powered Real Estate Value Predictor

> **QSkill Python Development Internship — Slab 1, Task 2**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.4%2B-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.x-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.26%2B-013243?logo=numpy&logoColor=white)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.8%2B-orange)](https://matplotlib.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📌 Project Overview

**HomePricer** is a complete end-to-end **Machine Learning pipeline** that predicts house prices using **Linear Regression**. A 1,000-record Kaggle-style Indian real estate dataset was generated, preprocessed with feature engineering and StandardScaler, split into training/test sets, trained, and evaluated using industry-standard metrics. The project also includes a fully functional **prediction interface** where users can input custom house details and receive an instant price estimate.

---

## ✨ Features

-  **Kaggle-style Dataset** — 1,000 house records with 10 raw features
-  **Feature Engineering** — 3 derived features: `House_Age`, `Sqft_Per_Room`, `Total_Rooms`
-  **Data Preprocessing** — StandardScaler normalization, 80/20 train-test split
-  **Linear Regression Model** — Trained with scikit-learn
-  **Model Evaluation** — R², MAE, RMSE, MAPE, 5-Fold Cross Validation
-  **Feature Importance** — Ranked by absolute scaled coefficients
-  **Prediction Interface** — `predict_house_price()` function for custom predictions
-  **5 Visualizations** — Price distribution, correlation heatmap, EDA charts, actual vs predicted, residuals

---

### 📂 Project Structure

```text
HomePricer/
├── HomePricer_House_Price_Prediction.ipynb  # Main Jupyter Notebook with core ML logic
├── .gitignore                               # Specifies intentionally untracked files to ignore
├── LICENSE                                  # MIT Open-source license documentation
└── README.md                                # Project overview, architecture, and insights
```

> 📊 **Note on Dataset & Visualizations:** The dataset (`house_prices.csv`) and exploratory plots (`.png`) are omitted from the root directory to maintain a clean production-ready repository. All charts, graphs, and model metrics are fully rendered and viewable directly inside the Jupyter Notebook.


---

## 📦 Dataset

A synthetic **House Price Dataset** with **1,000 transaction records** simulating Indian real estate market conditions.

| Feature | Type | Description | Range |
|---------|------|-------------|-------|
| `Bedrooms` | Integer | Number of bedrooms | 1 – 6 |
| `Bathrooms` | Integer | Number of bathrooms | 1 – 5 |
| `Sqft_Living` | Integer | Living area (sq. ft.) | 500 – 5,000 |
| `Sqft_Lot` | Integer | Total lot size (sq. ft.) | 700 – 8,000 |
| `Floors` | Float | Number of floors | 1, 1.5, 2, 2.5, 3 |
| `Condition` | Integer | House condition score | 1 (Poor) – 5 (Excellent) |
| `Location_Grade` | Integer | Neighbourhood desirability | 1 (Poor) – 10 (Prime) |
| `Year_Built` | Integer | Year of construction | 1970 – 2022 |
| `Has_Garage` | Binary | Garage available | 0 / 1 |
| `Has_Pool` | Binary | Pool available | 0 / 1 |
| `Price` *(target)* | Integer | House price in Rupees | ₹5L – ₹5Cr |

### Engineered Features (created during preprocessing)

| Feature | Formula | Purpose |
|---------|---------|---------|
| `House_Age` | `2024 − Year_Built` | Captures property age effect |
| `Sqft_Per_Room` | `Sqft_Living / (Beds + Baths)` | Space efficiency ratio |
| `Total_Rooms` | `Bedrooms + Bathrooms` | Combined room count |

---

## 🧮 ML Pipeline

```
Raw Dataset (1000 rows)
        │
        ▼
Feature Engineering         ← House_Age, Sqft_Per_Room, Total_Rooms
        │
        ▼
Train / Test Split (80/20)  ← random_state=42
        │
        ▼
StandardScaler              ← Zero mean, unit variance
        │
        ▼
LinearRegression.fit()      ← scikit-learn
        │
        ▼
Evaluation                  ← R², MAE, RMSE, 5-Fold CV
        │
        ▼
predict_house_price()       ← Custom prediction interface
```

---

## 📈 Model Performance

| Metric | Description |
|--------|-------------|
| **R² Score** | Proportion of price variance explained by the model |
| **MAE** | Mean Absolute Error — average prediction error in ₹ |
| **RMSE** | Root Mean Squared Error — penalises large errors more |
| **MAPE** | Mean Absolute Percentage Error |
| **5-Fold CV** | Cross-validation R² mean ± std deviation |

---

## 🧮 Prediction Interface

Use the built-in `predict_house_price()` function to predict house prices instantly:

```python
predicted_price = predict_house_price(
    bedrooms       = 3,       # Number of bedrooms (1–6)
    bathrooms      = 2,       # Number of bathrooms (1–5)
    sqft_living    = 1800,    # Living area in sq. ft.
    sqft_lot       = 3000,    # Lot size in sq. ft.
    floors         = 2.0,     # Number of floors (1/1.5/2/2.5/3)
    condition      = 3,       # House condition (1=Poor → 5=Excellent)
    location_grade = 6,       # Location grade (1=Poor → 10=Prime)
    year_built     = 2005,    # Year built
    has_garage     = 1,       # 1 = Yes, 0 = No
    has_pool       = 0        # 1 = Yes, 0 = No
)
```

### Example Predictions

| Property Type | Beds | Baths | Sqft | Grade | Pool | Garage |
|---------------|------|-------|------|-------|------|--------|
| Budget Starter | 2 | 1 | 750 | 3 | No | No |
| Mid-range Family | 3 | 2 | 1800 | 6 | No | Yes |
| Luxury Villa | 5 | 4 | 4200 | 10 | Yes | Yes |

---

## 📊 Visualizations

| Chart | Description |
|-------|-------------|
| `price_distribution.png` | Raw price histogram + log-price normalized distribution |
| `correlation_heatmap.png` | Pearson correlation matrix of all features |
| `eda_charts.png` | Sqft vs Price, Bedrooms vs Price, Location Grade vs Price, Amenity combos |
| `model_evaluation.png` | Actual vs Predicted scatter + Residuals vs Predicted |
| `feature_importance.png` | Horizontal bar chart of absolute scaled coefficients |

---

## 💡 Key Insights

- **Living area** (Sqft_Living) is the strongest price driver — size matters most.
- **Location Grade** shows a near-linear price relationship — prime areas command significant premiums.
- **Pool** adds approx. ₹3.5L on average; **Garage** adds approx. ₹2L.
- **House_Age** (engineered feature) contributes meaningfully — newer homes are priced higher.
- Residuals are randomly distributed — confirming the linear regression assumptions hold reasonably well.

---

## 🧑‍💻 Tech Stack

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.10+ | Core programming language |
| Pandas | 2.x | Data loading and manipulation |
| NumPy | 1.26+ | Numerical operations & dataset generation |
| Matplotlib | 3.8+ | Base plotting framework |
| Seaborn | 0.13+ | Statistical visualizations |
| Scikit-learn | 1.4+ | Linear regression, scaling, metrics, cross-validation |
| Jupyter Notebook | 7.x | Interactive development environment |

---

## 📁 Output Files

| File | Description |
|------|-------------|
| `house_prices.csv` | Generated dataset (1,000 rows × 11 columns) |
| `price_distribution.png` | Price histogram and log-normalized distribution |
| `correlation_heatmap.png` | Feature correlation matrix |
| `eda_charts.png` | Exploratory analysis charts |
| `model_evaluation.png` | Model performance visualization |
| `feature_importance.png` | Ranked feature importance |

---

## 🎓 Internship Details

| Field | Details |
|-------|---------|
| Organization | QSkill |
| Domain | Python Development |
| Slab | Slab 1 (Beginners) |
| Task | Task 2 of 3 |
| Duration | 1st April 2026 – 1st May 2026 |

---

## 🙌 Acknowledgements

- Dataset structure inspired by the Kaggle House Prices competition
- Built as part of the **QSkill Python Development Internship — Slab 1**

---

*Made with ❤️ using Python, Scikit-learn, Pandas & Matplotlib*
