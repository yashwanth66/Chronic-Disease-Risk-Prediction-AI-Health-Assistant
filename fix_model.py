import pickle
import json
import tempfile
import os
from xgboost import Booster as XGBBooster

with open("xgb_model.pkl", "rb") as f:
    model = pickle.load(f)

booster = model.get_booster()

with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
    tmp_path = f.name

booster.save_model(tmp_path)

with open(tmp_path, "r") as f:
    model_json = json.load(f)

param = model_json["learner"]["learner_model_param"]
bs = param.get("base_score", "0.5")
print(f"Original base_score: {bs}")

bs_clean = bs.strip("[]").strip()
try:
    bs_float = float(bs_clean)
except Exception:
    bs_float = 0.5

param["base_score"] = str(bs_float)
print(f"Fixed base_score: {param['base_score']}")

with open(tmp_path, "w") as f:
    json.dump(model_json, f)

clean_booster = XGBBooster()
clean_booster.load_model(tmp_path)
os.unlink(tmp_path)

model.get_booster().load_model(tmp_path if os.path.exists(tmp_path) else tmp_path)

with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
    fixed_path = f.name

clean_booster.save_model(fixed_path)
with open(fixed_path, "r") as f:
    check = json.load(f)
bs_check = check["learner"]["learner_model_param"]["base_score"]
print(f"Verified base_score in clean booster: {bs_check}")
os.unlink(fixed_path)

model._Booster = clean_booster

with open("xgb_model_fixed.pkl", "wb") as f:
    pickle.dump(model, f)

print("Done! Saved as xgb_model_fixed.pkl")
print("Now rename it: copy xgb_model_fixed.pkl xgb_model.pkl")
