import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

FILE_PATH = r"C:\Users\Kotha Yashwanth\OneDrive\Desktop\AMAT593_Project\LLCP2024.XPT"

MODEL_SAVE_PATH = r"C:\Users\Kotha Yashwanth\OneDrive\Desktop\AMAT593_Project\arthritis-app\backend\xgb_model.pkl"

columns_needed = [
    "_RFHLTH", "_PHYS14D", "_MENT14D", "_TOTINDA", "_EXTETH3",
    "_MICHD",  "_LTASTH1", "_DRDXAR2", "_SEX",    "_AGE_G",
    "HTIN4",   "WTKG3",    "_BMI5",    "_CHLDCNT", "_URBSTAT",
    "_SMOKER3", "DRNKANY6", "PNEUVAC4", "_STATE"
]

column_mapping = {
    "_RFHLTH"  : "Health Status",
    "_PHYS14D" : "Physical Health Status",
    "_MENT14D"  : "Mental Health",
    "_TOTINDA"  : "Physical Activity",
    "_EXTETH3"  : "Oral Health",
    "_MICHD"    : "CHD or Heart Disease",
    "_LTASTH1"  : "Asthma",
    "_DRDXAR2"  : "Arthritis",
    "_SEX"      : "Sex",
    "_AGE_G"    : "Age",
    "HTIN4"     : "Height (inches)",
    "WTKG3"     : "Weight (kg)",
    "_BMI5"     : "BMI",
    "_CHLDCNT"  : "Child Count",
    "_URBSTAT"  : "Urban/Rural",
    "_SMOKER3"  : "Tobacco Use",
    "DRNKANY6"  : "Alcohol Consumption",
    "PNEUVAC4"  : "Pneumonia Vaccination",
    "_STATE"    : "State"
}

print("Loading dataset in chunks...")
chunks = []
for chunk in pd.read_sas(FILE_PATH, format="xport", chunksize=10000):
    cols_available = [c for c in columns_needed if c in chunk.columns]
    chunks.append(chunk[cols_available])

df = pd.concat(chunks, ignore_index=True)
df_selected = df[list(column_mapping.keys())].rename(columns=column_mapping)

cols_with_79 = [
    "Health Status", "Physical Health Status", "Mental Health",
    "Physical Activity", "Oral Health", "Asthma",
    "Child Count", "Tobacco Use", "Alcohol Consumption",
    "Pneumonia Vaccination"
]
df_selected[cols_with_79] = df_selected[cols_with_79].replace({7: pd.NA, 9: pd.NA})
df_selected = df_selected.convert_dtypes()
df_selected["Weight (kg)"] = df_selected["Weight (kg)"] / 100
df_selected["BMI"]         = df_selected["BMI"] / 100

drop_row_cols = ["Physical Activity", "Alcohol Consumption", "Urban/Rural"]
df_imputed = df_selected.dropna(subset=drop_row_cols)

careful_cols = ["Pneumonia Vaccination", "Physical Health Status", "Oral Health", "Health Status"]
for col in careful_cols:
    df_imputed[col] = df_imputed[col].fillna(df_imputed[col].mode(dropna=True)[0])

simple_mode_cols = ["Child Count", "Tobacco Use", "CHD or Heart Disease", "Mental Health", "Asthma", "State"]
for col in simple_mode_cols:
    if df_imputed[col].isna().sum() > 0:
        df_imputed[col] = df_imputed[col].fillna(df_imputed[col].mode(dropna=True)[0])

simple_median_cols = ["BMI", "Weight (kg)", "Height (inches)"]
for col in simple_median_cols:
    if df_imputed[col].isna().sum() > 0:
        df_imputed[col] = df_imputed[col].fillna(df_imputed[col].median())

df_model = df_imputed.dropna(subset=["Arthritis"]).copy()
df_model = df_model.drop(columns=["Physical Activity", "Alcohol Consumption", "Urban/Rural"])

le = LabelEncoder()
for col in df_model.select_dtypes(include=["object", "category"]).columns:
    df_model[col] = le.fit_transform(df_model[col].astype(str))

df_sample = df_model.sample(n=min(100000, len(df_model)), random_state=42)

X = df_sample.drop(columns=["Arthritis"])
y = df_sample["Arthritis"].astype(int)
y = (y == 1).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
spw = neg / pos

print("Training XGBoost model...")
xgb = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=spw,
    use_label_encoder=False,
    eval_metric="auc",
    random_state=42,
    n_jobs=-1
)
xgb.fit(X_train, y_train)

from sklearn.metrics import roc_auc_score, accuracy_score
y_pred = xgb.predict(X_test)
y_proba = xgb.predict_proba(X_test)[:, 1]
print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Test AUC: {roc_auc_score(y_test, y_proba):.4f}")

with open(MODEL_SAVE_PATH, "wb") as f:
    pickle.dump(xgb, f)

print("Model saved to xgb_model.pkl")
print(f"Feature order: {list(X.columns)}")
