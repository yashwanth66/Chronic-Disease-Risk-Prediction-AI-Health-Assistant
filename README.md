# ArthriSense - Arthritis Risk Assessment App

Interactive UI for arthritis risk prediction using XGBoost + FastAPI + React + Claude AI.

---

## Project Structure

```
arthritis-app/
├── backend/
│   ├── main.py           ← FastAPI server
│   ├── train_model.py    ← Train & save XGBoost model
│   ├── requirements.txt
│   └── xgb_model.pkl     ← Generated after training
└── frontend/
    ├── src/
    │   ├── App.jsx
    │   ├── steps/
    │   │   ├── NameStep.jsx
    │   │   ├── GreetingStep.jsx
    │   │   ├── HealthStep.jsx
    │   │   ├── LifestyleStep.jsx
    │   │   ├── DemographicsStep.jsx
    │   │   └── ResultStep.jsx
    │   └── ...
    └── package.json
```

---

## Setup

### Step 1 - Train the Model

Open `backend/train_model.py` and update `FILE_PATH` to point to your `LLCP2024.XPT` file.

```bash
cd backend
pip install -r requirements.txt
python train_model.py
```

This saves `xgb_model.pkl` in the backend folder.

### Step 2 - Set Your Anthropic API Key

The backend uses Claude to generate personalized greeting and result text.

```bash
# On Windows
set ANTHROPIC_API_KEY=your_key_here

# On Mac/Linux
export ANTHROPIC_API_KEY=your_key_here
```

### Step 3 - Start the Backend

```bash
cd backend
uvicorn main:app --reload --port 8000
```

Backend runs at: http://localhost:8000

### Step 4 - Start the Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend runs at: http://localhost:5173

---

## How It Works

1. User enters their name → Claude generates a warm personalized greeting
2. User fills in Health History (Step 1 of 3)
3. User fills in Lifestyle Habits (Step 2 of 3)
4. User fills in Demographics with auto-calculated BMI (Step 3 of 3)
5. XGBoost model predicts arthritis risk
6. Claude generates a personalized summary of the result

---

## Feature Mapping (BRFSS → Model Input)

| UI Field              | BRFSS Code   | Model Feature            |
|-----------------------|--------------|--------------------------|
| General Health        | _RFHLTH      | Health Status            |
| Physical Health Days  | _PHYS14D     | Physical Health Status   |
| Mental Health Days    | _MENT14D     | Mental Health            |
| Physical Activity     | _TOTINDA     | Physical Activity        |
| Oral Health           | _EXTETH3     | Oral Health              |
| Heart Disease         | _MICHD       | CHD or Heart Disease     |
| Asthma                | _LTASTH1     | Asthma                   |
| Sex                   | _SEX         | Sex                      |
| Age Group             | _AGE_G       | Age                      |
| Height (inches)       | HTIN4        | Height (inches)          |
| Weight (converted)    | WTKG3        | Weight (kg)              |
| BMI (auto-calc)       | _BMI5        | BMI                      |
| Children              | _CHLDCNT     | Child Count              |
| Tobacco               | _SMOKER3     | Tobacco Use              |
| Alcohol               | DRNKANY6     | Alcohol Consumption      |
| Pneumonia Vax         | PNEUVAC4     | Pneumonia Vaccination    |
| State                 | _STATE       | State                    |

---

## Deploying to Render / Vercel

**Backend (Render):**
- Create a new Web Service
- Root directory: `backend`
- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- Add `ANTHROPIC_API_KEY` as an environment variable
- Upload `xgb_model.pkl` or retrain via a startup script

**Frontend (Vercel):**
- Root directory: `frontend`
- Build command: `npm run build`
- Output directory: `dist`
- Add env variable `VITE_API_URL` pointing to your Render backend URL
- Update API calls in App.jsx to use `import.meta.env.VITE_API_URL`
