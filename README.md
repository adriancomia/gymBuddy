# GymBuddy

A fitness chatbot built with a self-trained intent classification model. No external AI APIs are used — all responses come from a TF-IDF + Logistic Regression model trained on a custom dataset, combined with calculation logic for personalized workout schedules, diet plans, and BMI results.

## Live Demo

https://gymbuddyph.vercel.app/

## API Health Check

https://gymbuddy-api-1wit.onrender.com/health


## Stack

- **Model:** scikit-learn (TF-IDF vectorizer + Logistic Regression classifier)
- **Backend:** Python, Flask
- **Frontend:** React, Vite
- **NLP preprocessing:** NLTK (lemmatization), pyspellchecker (spelling correction)

## How it works

User input is cleaned and corrected, then classified into one of 22 intent categories by the trained model. If the predicted intent needs personal data (workout schedule, diet plan, protein target, BMI), the backend pulls the user's profile — weight, height, goal, experience, training days — and calculates the response. Everything else returns a pre-written answer tied to that category.

```
intents.json   → training dataset (22 categories, ~24 patterns each)
train.py       → trains the model, saves it to .pkl files
app.py         → Flask API, loads the model and serves chat requests
```

## Project structure

```
gymbuddy/
├── backend/
│   ├── intents.json
│   ├── train.py
│   ├── app.py
│   └── requirements.txt
└── frontend/
    ├── src/
    │   ├── App.jsx
    │   ├── App.css
    │   ├── Onboarding.jsx
    │   └── Onboarding.css
    └── package.json
```

## Setup

### Backend

```bash
cd backend
python -m venv venv
venv\Scripts\activate        # Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
python train.py              # trains the model, run once (or after editing intents.json)
python app.py                # starts the API on localhost:5000
```

### Frontend

```bash
cd frontend
npm install
npm run dev                  # runs on localhost:3000
```

## Retraining

Any change to `intents.json` requires retraining before it takes effect:

```bash
python train.py
```

`train.py` prints cross-validation accuracy and a held-out classification report (precision/recall/F1 per category) so you can check whether a category needs more training patterns.

## API

| Method | Route | Purpose |
|---|---|---|
| GET | `/health` | Confirms the server and model are running |
| POST | `/chat` | Send a message + profile, get a classified response |
| POST | `/profile/validate` | Returns BMI and recommended macros for a profile |

`/chat` request body:
```json
{
  "message": "give me a workout schedule",
  "user_profile": {
    "weight": 70,
    "height": 175,
    "goal": "muscle gain",
    "experience": "beginner",
    "daysPerWeek": 3
  }
}
```

---

**Author:** Adrian Comia
Built solo for IPT102 final project.
