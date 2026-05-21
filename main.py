from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import pickle
import os
import anthropic

app = FastAPI()

prediction_cache = {}  

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "xgb_model.pkl"
model = None

if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

GREETING_STYLES = [
    "warm and encouraging, like a trusted healthcare companion",
    "calm and reassuring, like a knowledgeable friend who happens to be a doctor",
    "professional yet approachable, like a wellness coach",
    "gentle and empathetic, acknowledging that health topics can feel personal",
    "optimistic and motivating, emphasizing that awareness is the first step to better health",
    "grounded and straightforward, respecting the user's intelligence and time",
]

FALLBACK_GREETINGS = [
    "Welcome, {name}! We're glad you're taking this step toward understanding your health. This tool uses a machine learning model trained on real BRFSS 2024 survey data to assess your arthritis risk.",
    "Hello, {name}! It's great to have you here. Let's walk through a few questions together - our model, trained on over 450,000 health survey responses, will provide your personalized risk assessment.",
    "Hi {name}, welcome to NexHealth! You're about to complete a research-grade arthritis risk assessment powered by the BRFSS 2024 dataset and an XGBoost prediction model.",
    "Good to see you, {name}! This assessment is designed to be straightforward and informative. Our machine learning model will analyze your responses and provide a personalized risk estimate.",
    "Welcome, {name}! Taking a few minutes to understand your health risk is a meaningful step. Our model has been trained on hundreds of thousands of real health survey responses to give you an accurate picture.",
    "Hi there, {name}! We've built this tool to make health risk assessment approachable and clear. Your answers will be processed by an XGBoost model trained on the BRFSS 2024 national health dataset.",
]

FALLBACK_SUMMARIES_HIGH = [
    "{name}, your responses indicate a higher likelihood of arthritis based on patterns in our training data. This doesn't mean a diagnosis - it means being more proactive about your joint health could be beneficial. Consider incorporating low-impact exercise like swimming or cycling into your routine, which has been shown to support joint health.",
    "Based on your health profile, {name}, the model has flagged a higher arthritis risk compared to the general population. We encourage you to discuss this with a healthcare provider who can give you personalized guidance. In the meantime, maintaining a healthy weight and staying active can make a meaningful difference.",
    "{name}, the model's assessment suggests elevated arthritis risk based on your health and lifestyle inputs. This is a valuable signal worth paying attention to. One practical step you can take today is to reduce processed food intake and increase anti-inflammatory foods like leafy greens, nuts, and fatty fish.",
]

FALLBACK_SUMMARIES_LOW = [
    "{name}, great news - your responses suggest a lower arthritis risk based on the patterns our model has learned. Keep up your current habits, as they appear to be working in your favor. Staying physically active and maintaining regular health checkups will help you continue on this positive path.",
    "Your health profile looks encouraging, {name}! The model places you in a lower arthritis risk category based on your inputs. Continue prioritizing regular exercise and a balanced diet - these are among the strongest protective factors against joint conditions.",
    "{name}, the model's assessment is reassuring - your responses align with lower arthritis risk patterns in the BRFSS dataset. That said, joint health is worth maintaining proactively at any age. Staying hydrated and incorporating strength training can help protect your joints long-term.",
]


class PredictionInput(BaseModel):
    name: str
    health_status: int
    physical_health: int
    mental_health: int
    physical_activity: int
    oral_health: int
    heart_disease: int
    asthma: int
    sex: int
    age: int
    height: float
    weight: float
    bmi: float
    child_count: int
    tobacco_use: int
    alcohol: int
    pneumonia_vax: int
    state: int


class GreetingInput(BaseModel):
    name: str


def encode_features(data: PredictionInput):
    return np.array([[
        data.health_status,
        data.physical_health,
        data.mental_health,
        data.oral_health,
        data.heart_disease,
        data.asthma,
        data.sex,
        data.age,
        data.height,
        data.weight,
        data.bmi,
        data.child_count,
        data.tobacco_use,
        data.pneumonia_vax,
        data.state,
    ]], dtype=float)


def has_api_key():
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    return bool(key and key.strip())


@app.post("/greet")
async def greet(body: GreetingInput):
    name_seed = sum(ord(c) for c in body.name.lower())

    if not has_api_key():
        greeting = FALLBACK_GREETINGS[name_seed % len(FALLBACK_GREETINGS)]
        return {"greeting": greeting.format(name=body.name.split()[0])}

    try:
        client = anthropic.Anthropic()
        style = GREETING_STYLES[name_seed % len(GREETING_STYLES)]
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=200,
            messages=[{
                "role": "user",
                "content": (
                    f"The user's name is {body.name}. Write a warm, 2-sentence greeting "
                    f"welcoming them to NexHealth, an Arthritis Risk Assessment tool powered by the BRFSS 2024 dataset. "
                    f"Address them by first name. Tone: {style}. "
                    f"Briefly mention that the tool uses a machine learning model trained on real health survey data. "
                    f"Do not repeat the same phrasing you might use for others - make it feel unique and personal. "
                    f"Do not use emojis. Do not use bullet points."
                )
            }]
        )
        return {"greeting": message.content[0].text}
    except Exception:
        greeting = FALLBACK_GREETINGS[name_seed % len(FALLBACK_GREETINGS)]
        return {"greeting": greeting.format(name=body.name.split()[0])}


@app.post("/predict")
async def predict(data: PredictionInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please run train_model.py first.")

    X = encode_features(data)
    print(f"INPUT FEATURES: {X.tolist()}")
    print(f"BMI: {data.bmi}, Weight: {data.weight}, Height: {data.height}")
    proba = model.predict_proba(X)[0]
    prediction = int(model.predict(X)[0])
    print(f"PROBA: {proba}, PREDICTION: {prediction}")

    risk_label = "High Risk" if prediction == 1 else "Low Risk"
    confidence = float(proba[prediction]) * 100
    name_seed = sum(ord(c) for c in data.name.lower())

    cache_key = f"{data.health_status}_{data.physical_health}_{data.mental_health}_{data.oral_health}_{data.heart_disease}_{data.asthma}_{data.sex}_{data.age}_{data.height}_{data.weight}_{data.bmi}_{data.child_count}_{data.tobacco_use}_{data.pneumonia_vax}_{data.state}_{data.name.lower()}"
    if cache_key in prediction_cache:
        ai_summary = prediction_cache[cache_key]
    else:

        if not has_api_key():
            if prediction == 1:
                summary = FALLBACK_SUMMARIES_HIGH[name_seed % len(FALLBACK_SUMMARIES_HIGH)]
            else:
                summary = FALLBACK_SUMMARIES_LOW[name_seed % len(FALLBACK_SUMMARIES_LOW)]
            ai_summary = summary.format(name=data.name.split()[0])
        else:
            try:
                client = anthropic.Anthropic()
                message = client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=300,
                    messages=[{
                        "role": "user",
                        "content": (
                            f"A user named {data.name} just completed an arthritis risk assessment. "
                            f"Result: {risk_label} with {confidence:.1f}% model confidence. "
                            f"Their age group: {data.age} (1=18-24, 2=25-34, 3=35-44, 4=45-54, 5=55-64, 6=65+). "
                            f"BMI: {data.bmi:.1f}. "
                            f"Write a compassionate, professional 3-sentence response explaining this result. "
                            f"Include one specific, actionable lifestyle tip relevant to arthritis prevention. "
                            f"Address them by first name. Do not give medical advice or diagnoses. "
                            f"Keep the tone warm and encouraging. Do not use emojis, bullet points, or em dashes. Write in plain flowing sentences only."
                        )
                    }]
                )
                ai_summary = message.content[0].text.replace("-", ",").replace("–", "to")
            except Exception:
                if prediction == 1:
                    summary = FALLBACK_SUMMARIES_HIGH[name_seed % len(FALLBACK_SUMMARIES_HIGH)]
                else:
                    summary = FALLBACK_SUMMARIES_LOW[name_seed % len(FALLBACK_SUMMARIES_LOW)]
                ai_summary = summary.format(name=data.name.split()[0])
        prediction_cache[cache_key] = ai_summary

    return {
        "prediction": prediction,
        "risk_label": risk_label,
        "confidence": round(confidence, 1),
        "proba_arthritis": round(float(proba[1]) * 100, 1),
        "proba_no_arthritis": round(float(proba[0]) * 100, 1),
        "ai_summary": ai_summary
    }


@app.get("/model-status")
async def model_status():
    return {"loaded": model is not None}


FEATURE_LABELS = {
    "health_status":    "General Health Status",
    "physical_health":  "Physical Health (past 30 days)",
    "mental_health":    "Mental Health (past 30 days)",
    "oral_health":      "Permanent Teeth Extracted",
    "heart_disease":    "Heart Disease History",
    "asthma":           "Asthma Diagnosis",
    "sex":              "Biological Sex",
    "age":              "Age Group",
    "height":           "Height (inches)",
    "weight":           "Weight (lbs)",
    "bmi":              "Body Mass Index",
    "child_count":      "Children in Household",
    "tobacco_use":      "Tobacco Use",
    "pneumonia_vax":    "Pneumonia Vaccination",
    "state":            "State of Residence",
}

DISPLAY_VALUES = {
    "health_status":   {1: "Good or Better", 2: "Fair or Poor"},
    "physical_health": {1: "0 days not good", 2: "1 to 13 days", 3: "14 or more days"},
    "mental_health":   {1: "0 days not good", 2: "1 to 13 days", 3: "14 or more days"},
    "oral_health":     {1: "Yes, teeth removed", 2: "No teeth removed"},
    "heart_disease":   {1: "Yes, diagnosed", 2: "No"},
    "asthma":          {1: "Yes, diagnosed", 2: "No"},
    "sex":             {1: "Male", 2: "Female"},
    "age":             {1: "18 to 24", 2: "25 to 34", 3: "35 to 44", 4: "45 to 54", 5: "55 to 64", 6: "65 or older"},
    "tobacco_use":     {1: "Daily smoker", 2: "Occasional smoker", 3: "Former smoker", 4: "Never smoked"},
    "pneumonia_vax":   {1: "Yes, vaccinated", 2: "Not vaccinated"},
    "child_count":     {1: "No children", 2: "One child", 3: "Two children", 4: "Three children", 5: "Four children", 6: "Five or more"},
}


@app.post("/shap")
async def shap_analysis(data: PredictionInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    import numpy as np

    feature_keys = [
        "health_status", "physical_health", "mental_health", "oral_health",
        "heart_disease", "asthma", "sex", "age", "height", "weight",
        "bmi", "child_count", "tobacco_use", "pneumonia_vax", "state"
    ]

    raw_values = [
        data.health_status, data.physical_health, data.mental_health, data.oral_health,
        data.heart_disease, data.asthma, data.sex, data.age, data.height, data.weight,
        data.bmi, data.child_count, data.tobacco_use, data.pneumonia_vax, data.state
    ]

    X = encode_features(data)
    base_proba = float(model.predict_proba(X)[0][1])

    contributions = []
    for i in range(len(feature_keys)):
        X_mod = X.copy()
        median_val = np.median([1, 2, 3])
        X_mod[0, i] = median_val
        perturbed = float(model.predict_proba(X_mod)[0][1])
        contribution = base_proba - perturbed
        contributions.append(contribution)

    features = []
    for i, key in enumerate(feature_keys):
        raw = raw_values[i]
        display_map = DISPLAY_VALUES.get(key, {})
        if key == "weight":
            lbs = round(raw * 2.205, 1)
            display_val = f"{lbs} lbs"
        elif key == "height":
            inches = int(raw)
            feet = inches // 12
            rem = inches % 12
            display_val = f"{feet}ft {rem}in"
        elif key == "bmi":
            display_val = f"{raw:.1f}"
        elif key == "state":
            display_val = f"State {int(raw)}"
        else:
            display_val = display_map.get(int(raw), str(raw))

        features.append({
            "name": FEATURE_LABELS.get(key, key),
            "shap_value": round(float(contributions[i]), 4),
            "display_value": display_val
        })

    features.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
    return {"features": features[:10]}
