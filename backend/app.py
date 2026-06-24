"""
GymBuddy AI - Flask API Server
Run: python app.py
Endpoint: POST /chat   Body: { "message": "...", "user_profile": {...} }

Intent classification is performed entirely by a trained TF-IDF +
Logistic Regression model. Code after classification only formats
responses using the user's profile data (math, not classification).
"""

import json
import pickle
import random
import re
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import nltk
from nltk.stem import WordNetLemmatizer
from spellchecker import SpellChecker

nltk.download('wordnet', quiet=True)

app = Flask(__name__)
CORS(app)

lemmatizer = WordNetLemmatizer()
spell = SpellChecker()

# ─── Load Trained Model ───────────────────────────────────────────────────────
MODEL_PATH = 'gymbuddy_model.pkl'
INTENTS_PATH = 'intents_data.pkl'

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Model not found! Run train.py first.")

with open(MODEL_PATH, 'rb') as f:
    pipeline = pickle.load(f)

with open(INTENTS_PATH, 'rb') as f:
    intents = pickle.load(f)

print("✅ Trained model loaded successfully")


# ─── Text Preprocessing (must match train.py exactly) ────────────────────────
def preprocess(text):
    number_map = {'0': 'o', '1': 'i', '3': 'e', '4': 'a', '5': 's', '7': 't'}
    text = ''.join(number_map.get(c, c) for c in text)

    words = text.split()
    corrected = [spell.correction(w) or w for w in words]
    text = ' '.join(corrected)

    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return ' '.join(tokens)


def get_static_response(tag):
    for intent in intents:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])
    return "I'm not sure about that. Try asking about workouts, diet, or your schedule!"


# ─── Business Logic Helpers (math/formatting, not classification) ───────────
def calculate_bmi(weight_kg, height_cm):
    h = height_cm / 100
    bmi = weight_kg / (h * h)
    if bmi < 18.5:
        category = "Underweight"
    elif bmi < 25:
        category = "Normal weight"
    elif bmi < 30:
        category = "Overweight"
    else:
        category = "Obese"
    return round(bmi, 1), category


def extract_days_from_message(msg_lower):
    day_words = {
        'one': 1, 'whole': 1, 'once': 1,
        'two': 2, 'twice': 2,
        'three': 3, 'four': 4,
        'five': 5, 'six': 6, 'seven': 7, 'everyday': 7, 'every day': 7,
        '1': 1, '2': 2, '3': 3, '4': 4,
        '5': 5, '6': 6, '7': 7
    }
    for word, num in day_words.items():
        if word in msg_lower:
            return num
    return None


def generate_schedule(profile, override_days=None):
    goal = profile.get('goal', 'general fitness').lower()
    days_per_week = override_days if override_days else profile.get('daysPerWeek', 3)

    if days_per_week <= 2:
        plan = {
            "type": "Full Body Program",
            "schedule": [
                {"day": "Monday", "focus": "Full Body A", "exercises": ["Squat 3x8", "Bench Press 3x8", "Bent-over Row 3x8", "OHP 3x10", "Plank 3x30s"]},
                {"day": "Thursday", "focus": "Full Body B", "exercises": ["Deadlift 3x5", "Incline Press 3x10", "Pull-ups 3x max", "Lunges 3x12", "Core circuit"]},
            ]
        }
    elif days_per_week == 3:
        plan = {
            "type": "Full Body Program",
            "schedule": [
                {"day": "Monday", "focus": "Full Body A", "exercises": ["Squat 3x8", "Bench Press 3x8", "Bent-over Row 3x8", "OHP 3x10", "Plank 3x30s"]},
                {"day": "Wednesday", "focus": "Full Body B", "exercises": ["Deadlift 3x5", "Incline Press 3x10", "Pull-ups 3x max", "Lunges 3x12", "Core circuit"]},
                {"day": "Friday", "focus": "Full Body C", "exercises": ["Front Squat 3x8", "Dips 3x10", "Cable Row 3x12", "Lateral Raises 3x15", "Ab wheel 3x10"]}
            ]
        }
    elif days_per_week == 4:
        plan = {
            "type": "Upper/Lower Split",
            "schedule": [
                {"day": "Monday", "focus": "Upper Body (Strength)", "exercises": ["Bench Press 4x5", "Barbell Row 4x5", "OHP 3x8", "Pull-ups 3x8"]},
                {"day": "Tuesday", "focus": "Lower Body (Strength)", "exercises": ["Squat 4x5", "Romanian Deadlift 3x8", "Leg Press 3x10", "Calf Raises 4x15"]},
                {"day": "Thursday", "focus": "Upper Body (Hypertrophy)", "exercises": ["Incline DB Press 4x10", "Cable Row 4x10", "Lateral Raises 4x15", "Bicep Curl 3x12", "Tricep Pushdown 3x12"]},
                {"day": "Saturday", "focus": "Lower Body (Hypertrophy)", "exercises": ["Hack Squat 4x10", "Leg Curl 4x12", "Walking Lunges 3x20", "Glute Bridge 3x15", "Calf Raises 4x20"]}
            ]
        }
    else:
        plan = {
            "type": "Push/Pull/Legs (PPL)",
            "schedule": [
                {"day": "Monday", "focus": "Push (Chest/Shoulders/Triceps)", "exercises": ["Bench Press 4x8", "OHP 3x10", "Incline DB Press 3x12", "Lateral Raises 3x15", "Tricep Dips 3x12"]},
                {"day": "Tuesday", "focus": "Pull (Back/Biceps)", "exercises": ["Deadlift 4x5", "Pull-ups 4x max", "Barbell Row 3x8", "Face Pulls 3x15", "Hammer Curls 3x12"]},
                {"day": "Wednesday", "focus": "Legs", "exercises": ["Squat 4x8", "Romanian Deadlift 3x10", "Leg Press 3x12", "Leg Curl 3x12", "Calf Raises 4x20"]},
                {"day": "Thursday", "focus": "Push (Volume)", "exercises": ["Incline Bench 4x10", "DB Shoulder Press 3x12", "Cable Flyes 3x15", "Skull Crushers 3x12"]},
                {"day": "Friday", "focus": "Pull (Volume)", "exercises": ["Lat Pulldown 4x12", "Seated Cable Row 3x12", "Single-arm Row 3x12", "Preacher Curl 3x12"]},
                {"day": "Saturday", "focus": "Legs + Core", "exercises": ["Front Squat 3x8", "Hack Squat 3x10", "Walking Lunges 3x20", "Ab Circuit 3 rounds"]}
            ]
        }

    tip = {
        'weight loss': "Do 20-30 min cardio after each session for maximum fat burn 🔥",
        'muscle gain': "Make sure you're eating in a 300-500 calorie surplus 🍗",
        'strength': "Rest 3-5 minutes between heavy sets for full recovery ⚡",
        'general fitness': "Consistency beats intensity — show up every scheduled day 💪"
    }.get(goal, "Stay hydrated and get 7-9 hours of sleep 💤")

    return plan, tip


def generate_diet_plan(profile):
    weight_kg = profile.get('weight', 70)
    goal = profile.get('goal', 'general fitness').lower()
    weight_lbs = weight_kg * 2.205

    if 'weight loss' in goal or 'cut' in goal:
        calories = int(weight_lbs * 13)
        protein = int(weight_lbs * 1.0)
        carbs = int((calories * 0.35) / 4)
        fats = int((calories * 0.25) / 9)
        meal_focus = "High protein, calorie deficit"
    elif 'muscle' in goal or 'bulk' in goal:
        calories = int(weight_lbs * 17)
        protein = int(weight_lbs * 1.0)
        carbs = int((calories * 0.45) / 4)
        fats = int((calories * 0.30) / 9)
        meal_focus = "High carb, calorie surplus"
    else:
        calories = int(weight_lbs * 15)
        protein = int(weight_lbs * 0.85)
        carbs = int((calories * 0.40) / 4)
        fats = int((calories * 0.30) / 9)
        meal_focus = "Balanced macros for maintenance"

    return {
        "daily_calories": calories,
        "macros": {"protein_g": protein, "carbs_g": carbs, "fats_g": fats},
        "meal_focus": meal_focus,
        "meals": [
            {"name": "Breakfast", "example": "3 eggs + oats with banana + black coffee"},
            {"name": "Lunch", "example": "Grilled chicken breast + brown rice + veggies"},
            {"name": "Snack", "example": "Greek yogurt + handful of almonds"},
            {"name": "Dinner", "example": "Salmon or beef + sweet potato + salad"},
            {"name": "Post-Workout", "example": "Whey protein shake + fruit"}
        ]
    }


# ─── Routes ───────────────────────────────────────────────────────────────────
@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "model": "GymBuddy TF-IDF + Logistic Regression Classifier"})


@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    if not data or 'message' not in data:
        return jsonify({"error": "Missing 'message' field"}), 400

    user_message = data['message'].strip()
    user_profile = data.get('user_profile', {})

    if not user_message:
        return jsonify({"error": "Empty message"}), 400

    msg_lower = user_message.lower()

    # ─── Intent classification — 100% model, no shortcuts ───────────────────
    processed = preprocess(user_message)
    predicted_tag = pipeline.predict([processed])[0]
    confidence = pipeline.predict_proba([processed]).max()

    if confidence < 0.15:
        response = "I'm not quite sure I understood that 🤔 Try asking about workouts, diet tips, your schedule, or exercises like squats or bench press!"
        return jsonify({"response": response, "tag": "unknown", "confidence": round(float(confidence), 2)})

    # ─── Response formatting (uses predicted_tag, not classification logic) ──
    if predicted_tag == 'workout_schedule':
        if not user_profile.get('weight'):
            response = "I'd love to build you a schedule! Please **set up your profile first** so I can personalize it for your goals and experience level. Click the profile icon 👆"
        else:
            override_days = extract_days_from_message(msg_lower)
            plan, tip = generate_schedule(user_profile, override_days)
            schedule_text = f"🗓️ **Your {plan['type']} Schedule**\n\n"
            for session in plan['schedule']:
                schedule_text += f"**{session['day']} — {session['focus']}**\n"
                for ex in session['exercises']:
                    schedule_text += f"  • {ex}\n"
                schedule_text += "\n"
            schedule_text += f"💡 **Pro Tip:** {tip}"
            response = schedule_text

    elif predicted_tag == 'diet_general':
        if not user_profile.get('weight'):
            response = "To give you a personalized diet plan, please **set up your profile** first! I need your weight and goal to calculate your macros. Click the profile icon 👆"
        else:
            diet = generate_diet_plan(user_profile)
            response = f"🥗 **Your Personalized Diet Plan**\n\n"
            response += f"**Goal:** {diet['meal_focus']}\n"
            response += f"**Daily Calories:** {diet['daily_calories']} kcal\n\n"
            response += f"**Macros:**\n"
            response += f"  • Protein: {diet['macros']['protein_g']}g\n"
            response += f"  • Carbs: {diet['macros']['carbs_g']}g\n"
            response += f"  • Fats: {diet['macros']['fats_g']}g\n\n"
            response += "**Sample Meals:**\n"
            for meal in diet['meals']:
                response += f"  🍽️ **{meal['name']}:** {meal['example']}\n"

    elif predicted_tag == 'protein':
        if not user_profile.get('weight'):
            response = "Please **set up your profile first** so I can calculate your exact protein needs! Click the profile icon 👆"
        else:
            weight_lbs = user_profile['weight'] * 2.205
            min_protein = int(weight_lbs * 0.8)
            max_protein = int(weight_lbs * 1.0)
            response = f"🥩 **Your Daily Protein Target**\n\n"
            response += f"**Your Weight:** {user_profile['weight']}kg\n"
            response += f"**Recommended:** {min_protein}g – {max_protein}g per day\n\n"
            response += f"**Best Sources:**\n"
            response += f"  • Chicken breast — 31g per 100g\n"
            response += f"  • Eggs — 6g per egg\n"
            response += f"  • Greek yogurt — 17g per cup\n"
            response += f"  • Whey protein shake — 25g per scoop\n"
            response += f"  • Tuna — 30g per can\n"
            response += f"  • Lentils — 18g per cup\n\n"
            response += f"💡 **Tip:** Spread protein across 4-5 meals for maximum absorption!"

    elif predicted_tag == 'bmi':
        if not user_profile.get('weight') or not user_profile.get('height'):
            response = "To calculate your BMI, I need your **weight and height**. Please set up your profile first! Click the profile icon 👆"
        else:
            bmi, category = calculate_bmi(user_profile['weight'], user_profile['height'])
            response = f"📊 **Your BMI Results**\n\n"
            response += f"**BMI:** {bmi}\n"
            response += f"**Category:** {category}\n\n"
            if category == "Underweight":
                response += "💡 Focus on increasing calorie intake and building muscle. A bulking program would work well for you!"
            elif category == "Normal weight":
                response += "💡 Great! Maintain your weight while improving strength and endurance."
            elif category == "Overweight":
                response += "💡 A calorie deficit diet combined with cardio and strength training will help you reach a healthy weight."
            else:
                response += "💡 Prioritize fat loss through diet (80%) and exercise (20%). Start with 3 days/week of mixed cardio and weights."

    elif predicted_tag == 'hydration':
        response = "💧 **How Much Water Should You Drink?**\n\n"
        response += "**General Rule:** 35ml per kg of bodyweight per day\n\n"
        if user_profile.get('weight'):
            daily = round(user_profile['weight'] * 0.035, 1)
            response += f"**Your target:** ~{daily}L per day based on your {user_profile['weight']}kg\n\n"
        response += "**Drink more when:**\n"
        response += "  • Training sessions (add 500ml-1L per hour of exercise)\n"
        response += "  • Hot weather\n"
        response += "  • High intensity workouts\n\n"
        response += "**Signs of dehydration:**\n"
        response += "  • Dark yellow urine\n"
        response += "  • Headaches during workout\n"
        response += "  • Muscle cramps\n\n"
        response += "💡 **Tip:** Drink a glass of water first thing in the morning before anything else!"

    elif predicted_tag == 'weight_management':
        response = "⚖️ **Weight Management Guide**\n\n"
        response += "**To Lose Weight:**\n"
        response += "  • Eat in a calorie deficit (300-500 kcal below maintenance)\n"
        response += "  • High protein diet to preserve muscle\n"
        response += "  • Combine cardio + strength training\n\n"
        response += "**To Gain Weight:**\n"
        response += "  • Eat in a calorie surplus (300-500 kcal above maintenance)\n"
        response += "  • Focus on resistance training\n"
        response += "  • Eat every 3-4 hours\n\n"
        response += "**To Maintain Weight:**\n"
        response += "  • Eat at maintenance calories\n"
        response += "  • Stay consistent with exercise\n\n"
        response += "💡 **Tip:** Set up your profile and ask for a diet plan — I'll calculate your exact calorie target!"

    elif predicted_tag == 'profile':
        response = "To update your profile, click the **profile icon** in the bottom left corner. You can set your age, weight, height, fitness goal, and experience level there!"

    else:
        response = get_static_response(predicted_tag)

    return jsonify({
        "response": response,
        "tag": predicted_tag,
        "confidence": round(float(confidence), 2)
    })


@app.route('/profile/validate', methods=['POST'])
def validate_profile():
    profile = request.get_json()
    if not profile:
        return jsonify({"error": "No profile data"}), 400
    result = {}
    if profile.get('weight') and profile.get('height'):
        bmi, category = calculate_bmi(profile['weight'], profile['height'])
        result['bmi'] = bmi
        result['bmi_category'] = category
    if profile.get('weight') and profile.get('goal'):
        diet = generate_diet_plan(profile)
        result['recommended_calories'] = diet['daily_calories']
        result['recommended_macros'] = diet['macros']
    return jsonify(result)


# ─── Run ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("\n🏋️  GymBuddy AI Backend Running")
    print("API: http://localhost:5000")
    print("Health check: http://localhost:5000/health\n")
    app.run(host='0.0.0.0', port=5000, debug=False)