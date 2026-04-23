"""
train.py  —  Run ONCE locally before pushing to GitHub.
    python train.py
Reads cleaned_data.csv → engineers 13 features → trains 5 models → saves models/
"""
import os, pickle
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

DATA_PATH = "cleaned_data.csv"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Tier maps ──────────────────────────────────────────────────
TIER1 = {"Bangalore","Bengaluru","Mumbai","Delhi","New Delhi","Chennai","Hyderabad",
         "Kolkata","Gurgaon","Gurugram","Noida","Andheri","Chembur","Kormangala",
         "Bengaluru and Gurugram","Mumbai/Bengaluru"}
TIER2 = {"Pune","Ahmedabad","Jaipur","Chandigarh","Indore","Lucknow","Nagpur","Kochi",
         "Coimbatore","Bhopal","Vadodara","Surat","Trivandrum","Bhubneswar",
         "Goa","Panaji","Faridabad","Haryana","Karnataka","Kerala","Taramani"}
TIER3 = {"Amritsar","Gwalior","Varanasi","Kanpur","Rourkela","Jodhpur","Udaipur",
         "Gaya","Udupi","Belgaum","Missourie","Burnsville","Tulangan","Nairobi"}

HIGH_FUND_INDUSTRIES = {"FinTech","E-commerce","E-Commerce","Finance",
                         "Consumer Internet","Technology","Logistics"}

def get_tier(city):
    if city in TIER1: return "Tier 1"
    if city in TIER2: return "Tier 2"
    if city in TIER3: return "Tier 3"
    if any(x in city for x in ["USA","US","SFO","California","Boston",
                                "New York","Singapore","Palo Alto"]): return "International"
    return "Other"

def tier_num(city):
    t = get_tier(city)
    return {"Tier 1":1,"Tier 2":2,"Tier 3":3,"International":0,"Other":4}[t]

# ── Feature engineering ────────────────────────────────────────
def engineer_features(df_raw):
    """
    Takes cleaned_data.csv (one-hot encoded) and adds 13 interpretable features.
    Returns a DataFrame with ONLY the 13 engineered features + Success + Amoount.
    """
    city_cols = [c for c in df_raw.columns if c.startswith("City  Location_")]
    ind_cols  = [c for c in df_raw.columns if c.startswith("Industry Vertical_")]

    city_matrix = df_raw[city_cols].astype(int)
    ind_matrix  = df_raw[ind_cols].astype(int)

    city_name = city_matrix.idxmax(axis=1).str.replace("City  Location_","",regex=False)
    ind_name  = ind_matrix.idxmax(axis=1).str.replace("Industry Vertical_","",regex=False)

    # City-level aggregates
    city_avg_fund   = df_raw.groupby(city_name)["Amoount"].transform("mean")
    city_count      = city_name.map(city_name.value_counts())
    city_success_rt = df_raw.groupby(city_name)["Success"].transform("mean")

    # Industry-level aggregates
    ind_avg_fund    = df_raw.groupby(ind_name)["Amoount"].transform("mean")
    ind_count       = ind_name.map(ind_name.value_counts())

    feat = pd.DataFrame({
        # ── Feature 1: Raw funding amount
        "funding_amount":         df_raw["Amoount"],
        # ── Feature 2: Log-transformed amount (handles skew)
        "log_funding":            np.log1p(df_raw["Amoount"]),
        # ── Feature 3: Funding bucket (0=<5L … 5=50Cr+)
        "funding_bucket":         pd.cut(df_raw["Amoount"],
                                    bins=[0,500_000,2_000_000,10_000_000,
                                          50_000_000,500_000_000,float("inf")],
                                    labels=[0,1,2,3,4,5]).astype(float),
        # ── Feature 4: City tier number (1=best, 3=small, 0=international)
        "city_tier":              city_name.map(tier_num),
        # ── Feature 5: Is Tier 1 city (binary)
        "is_tier1_city":          city_name.isin(TIER1).astype(int),
        # ── Feature 6: Is international city (binary)
        "is_international":       city_name.apply(lambda c:
                                    1 if any(x in c for x in ["USA","US","SFO",
                                    "California","Boston","New York",
                                    "Singapore","Palo Alto"]) else 0),
        # ── Feature 7: City startup density (how active the city is)
        "city_startup_density":   city_count,
        # ── Feature 8: City average funding (ecosystem strength)
        "city_avg_funding":       city_avg_fund,
        # ── Feature 9: City historical success rate
        "city_success_rate":      city_success_rt,
        # ── Feature 10: Is high-funding industry
        "is_hot_industry":        ind_name.isin(HIGH_FUND_INDUSTRIES).astype(int),
        # ── Feature 11: Industry startup density
        "industry_density":       ind_count,
        # ── Feature 12: Industry average funding
        "industry_avg_funding":   ind_avg_fund,
        # ── Feature 13: Funding vs. industry average ratio
        "fund_vs_industry_avg":   df_raw["Amoount"] / (ind_avg_fund + 1),
    })

    feat["Success"] = df_raw["Success"].values
    feat = feat.fillna(0)
    return feat, city_name, ind_name

# ── Load data ──────────────────────────────────────────────────
print("📂 Loading data…")
df_raw = pd.read_csv(DATA_PATH)
print(f"   Raw shape: {df_raw.shape}")

feat_df, city_series, ind_series = engineer_features(df_raw)
print(f"   Feature shape: {feat_df.shape}")
print(f"   Features: {[c for c in feat_df.columns if c != 'Success']}")

# ── Train / test split ─────────────────────────────────────────
FEATURE_COLS = [c for c in feat_df.columns if c not in ("Success","funding_amount")]
X = feat_df[FEATURE_COLS]
y = feat_df["Success"]
X_reg = feat_df[FEATURE_COLS + ["Success"]]  # regression uses success too
y_reg = feat_df["funding_amount"]

print(f"\n   X shape: {X.shape}, y distribution: {y.value_counts().to_dict()}")

# ── Five classifiers ───────────────────────────────────────────
print("\n🧠 Training classifiers…")

models = {}

# 1. Logistic Regression
lr = Pipeline([("scaler", StandardScaler()),
               ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5))])
lr.fit(X, y)
models["Logistic Regression"] = lr
print(f"   LR accuracy: {lr.score(X,y):.4f}")

# 2. Random Forest
rf = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42,
                             class_weight="balanced", min_samples_leaf=3)
rf.fit(X, y)
models["Random Forest"] = rf
print(f"   RF accuracy: {rf.score(X,y):.4f}")

# 3. Gradient Boosting
gb = GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                                  random_state=42)
gb.fit(X, y)
models["Gradient Boosting"] = gb
print(f"   GB accuracy: {gb.score(X,y):.4f}")

# 4. SVM
svm = Pipeline([("scaler", StandardScaler()),
                ("clf", SVC(probability=True, class_weight="balanced",
                            kernel="rbf", C=1.0, random_state=42))])
svm.fit(X, y)
models["SVM"] = svm
print(f"   SVM accuracy: {svm.score(X,y):.4f}")

# 5. KNN
knn = Pipeline([("scaler", StandardScaler()),
                ("clf", KNeighborsClassifier(n_neighbors=15))])
knn.fit(X, y)
models["KNN"] = knn
print(f"   KNN accuracy: {knn.score(X,y):.4f}")

# ── Regressor ──────────────────────────────────────────────────
print("\n📈 Training funding regressor (Ridge)…")
ridge_reg = Pipeline([("scaler", StandardScaler()),
                       ("reg", Ridge(alpha=10.0))])
ridge_reg.fit(X_reg, y_reg)
print(f"   Ridge R²: {ridge_reg.score(X_reg, y_reg):.4f}")

# ── Chart stats ────────────────────────────────────────────────
print("\n📊 Computing chart stats…")

df_raw2 = df_raw.copy()
df_raw2["city_name"]     = city_series.values
df_raw2["industry_name"] = ind_series.values

city_cols = [c for c in df_raw.columns if c.startswith("City  Location_")]
ind_cols_r = [c for c in df_raw.columns if c.startswith("Industry Vertical_")]

city_stats = (df_raw2.groupby("city_name")
              .agg(count=("Success","count"),
                   success_rate=("Success","mean"),
                   avg_fund=("Amoount","mean"),
                   median_fund=("Amoount","median"))
              .reset_index()
              .query("count >= 5"))
city_stats["tier"] = city_stats["city_name"].apply(get_tier)

ind_stats = (df_raw2.groupby("industry_name")
             .agg(count=("Success","count"),
                  success_rate=("Success","mean"),
                  avg_fund=("Amoount","mean"),
                  median_fund=("Amoount","median"))
             .reset_index()
             .query("count >= 8"))

tier_stats = (city_stats.groupby("tier")
              .apply(lambda g: pd.Series({
                  "total_startups": int(g["count"].sum()),
                  "avg_success_rate": float((g["success_rate"]*g["count"]).sum()/g["count"].sum()),
                  "avg_fund": float((g["avg_fund"]*g["count"]).sum()/g["count"].sum()),
              }), include_groups=False)
              .reset_index())

bins   = [0,500_000,2_000_000,10_000_000,50_000_000,500_000_000,float("inf")]
labels = ["<5L","5L–20L","20L–1Cr","1Cr–5Cr","5Cr–50Cr","50Cr+"]
df_raw2["fund_bucket"] = pd.cut(df_raw2["Amoount"], bins=bins, labels=labels)
bucket_stats = (df_raw2.groupby("fund_bucket", observed=True)["Success"]
                .agg(["count","mean"])
                .reset_index()
                .rename(columns={"mean":"success_rate"}))

# Feature importance from RF
feat_importance = pd.Series(rf.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)

raw_stats = {
    "city_df":    city_stats,
    "ind_df":     ind_stats,
    "tier_df":    tier_stats,
    "bucket_df":  bucket_stats,
    "feat_imp":   feat_importance,
    "total":      len(df_raw2),
    "overall_success":  float(df_raw2["Success"].mean()),
    "avg_fund_overall": float(df_raw2["Amoount"].mean()),
}

# ── Save ──────────────────────────────────────────────────────
saves = {
    "models.pkl":          models,
    "ridge_reg.pkl":       ridge_reg,
    "feature_cols.pkl":    FEATURE_COLS,
    "raw_stats.pkl":       raw_stats,
    "city_avg_fund.pkl":   dict(zip(df_raw2["city_name"], city_series.map(df_raw2.groupby("city_name")["Amoount"].mean()))),
    "ind_avg_fund.pkl":    dict(zip(df_raw2["industry_name"], ind_series.map(df_raw2.groupby("industry_name")["Amoount"].mean()))),
    "city_counts.pkl":     df_raw2["city_name"].value_counts().to_dict(),
    "ind_counts.pkl":      df_raw2["industry_name"].value_counts().to_dict(),
    "city_success_rt.pkl": df_raw2.groupby("city_name")["Success"].mean().to_dict(),
}

for fname, obj in saves.items():
    with open(os.path.join(MODEL_DIR, fname),"wb") as f:
        pickle.dump(obj, f)

print(f"\n✅ Saved {len(saves)} files to ./{MODEL_DIR}/")
for fname in saves: print(f"   {fname}")
print("\n🚀 Run: streamlit run app.py")
