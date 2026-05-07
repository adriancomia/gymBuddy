"""
GymBuddy AI - Flask API Server
Run: python app.py
Endpoint: POST /chat   Body: { "message": "...", "user_profile": {...} }
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
from textblob import TextBlob

nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)

app = Flask(__name__)
CORS(app)

lemmatizer = WordNetLemmatizer()

# ─── Load Model ──────────────────────────────────────────────────────────────
MODEL_PATH = 'gymbuddy_model.pkl'
INTENTS_PATH = 'intents_data.pkl'

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Model not found! Run train.py first.")

with open(MODEL_PATH, 'rb') as f:
    pipeline = pickle.load(f)

with open(INTENTS_PATH, 'rb') as f:
    intents = pickle.load(f)

print("✅ Model loaded successfully")

# ─── Helpers ─────────────────────────────────────────────────────────────────
def preprocess(text):
    number_map = {'0': 'o', '1': 'i', '3': 'e', '4': 'a', '5': 's', '7': 't'}
    text = ''.join(number_map.get(c, c) for c in text)
    text = str(TextBlob(text).correct())
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return ' '.join(tokens)

def get_response_for_tag(tag):
    for intent in intents:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])
    return "I'm not sure about that. Try asking about workouts, diet, or your schedule!"

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

# ─── Routes ──────────────────────────────────────────────────────────────────
@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "model": "GymBuddy TF-IDF Classifier"})

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
    override_days = None

    # ─── Keyword Shortcuts ────────────────────────────────────────────────────
    if any(w in msg_lower for w in ['bmi', 'body mass']):
        predicted_tag = 'bmi'
        confidence = 1.0

    elif any(w in msg_lower for w in ['creatine', 'supplement', 'preworkout', 'whey', 'bcaa', 'protein powder', 'mass gainer']):
        predicted_tag = 'supplements'
        confidence = 1.0

    elif any(w in msg_lower for w in ['fasting while', 'fasted workout', 'fasted training', 'empty stomach', 'workout without eating', 'is it okay to workout on']):
        predicted_tag = 'fasted_workout'
        confidence = 1.0

    elif any(w in msg_lower for w in ['sweet treat', 'cheat meal', 'cheat day', 'junk food', 'can i eat', 'is it okay to eat', 'unhealthy food']):
        predicted_tag = 'cheat_meal'
        confidence = 1.0

    elif any(w in msg_lower for w in ['how long until', 'when will i see results', 'visible results', 'how long to see']):
        predicted_tag = 'results_timeline'
        confidence = 1.0

    elif any(w in msg_lower for w in ['safe exercise', 'after having a baby', 'postpartum', 'after pregnancy']):
        predicted_tag = 'postpartum'
        confidence = 1.0

    elif any(w in msg_lower for w in ['progressive overload', 'what is progressive overload', 'how to progress']):
        predicted_tag = 'progressive_overload'
        confidence = 1.0

    elif any(w in msg_lower for w in ['slim and strong', 'slim strong', 'lean and strong', 'toned body', 'slim body']):
        predicted_tag = 'slim_strong'
        confidence = 1.0

    elif any(w in msg_lower for w in ['lower body fat', 'exercise to lower body fat', 'reduce body fat', 'lose body fat']):
        predicted_tag = 'lower_body_fat'
        confidence = 1.0

    elif any(w in msg_lower for w in ['first time in gym', 'tips for gym', 'first day gym', 'gym for first time', 'never been to gym']):
        predicted_tag = 'first_time_gym'
        confidence = 1.0

    elif any(w in msg_lower for w in ['lose weight fast', 'fastest way to lose', 'quick weight loss', 'how to lose weight']):
        predicted_tag = 'weight_loss_tips'
        confidence = 1.0

    elif any(w in msg_lower for w in ['gain weight', 'how to gain weight', 'too skinny', 'underweight', 'cant gain weight']):
        predicted_tag = 'gain_weight'
        confidence = 1.0

    elif any(w in msg_lower for w in ['abs', 'six pack', 'core workout', 'how to get abs', 'flat stomach', 'stomach exercise']):
        predicted_tag = 'abs_workout'
        confidence = 1.0

    elif any(w in msg_lower for w in ['how much sleep', 'sleep for muscle', 'sleep and gym', 'sleep recovery', 'sleep and workout']):
        predicted_tag = 'sleep'
        confidence = 1.0

    elif any(w in msg_lower for w in ['injury', 'hurt', 'pain during workout', 'knee pain', 'back pain', 'shoulder pain', 'wrist pain']):
        predicted_tag = 'injury'
        confidence = 1.0

    elif any(w in msg_lower for w in ['gym bag', 'what to bring to gym', 'gym essentials', 'what do i need for gym']):
        predicted_tag = 'gym_essentials'
        confidence = 1.0

    elif any(w in msg_lower for w in ['home workout', 'workout at home', 'no gym', 'without gym', 'no equipment']):
        predicted_tag = 'home_workout'
        confidence = 1.0

    elif any(w in msg_lower for w in ['plateau', 'not seeing progress', 'stuck', 'no progress', 'stopped losing weight', 'not gaining muscle']):
        predicted_tag = 'plateau'
        confidence = 1.0

    elif any(w in msg_lower for w in ['how to breathe', 'breathing during', 'breathing while lifting', 'when to breathe']):
        predicted_tag = 'breathing'
        confidence = 1.0

    elif any(w in msg_lower for w in ['gym anxiety', 'scared of gym', 'nervous at gym', 'intimidated by gym', 'shy at gym']):
        predicted_tag = 'gym_anxiety'
        confidence = 1.0

    elif any(w in msg_lower for w in ['what to eat after', 'post workout food', 'after workout meal', 'eat after gym', 'best post workout', 'post workout meal', 'after training meal']):
        predicted_tag = 'post_workout_meal'
        confidence = 1.0

    elif any(w in msg_lower for w in ['how to bulk', 'bulking tips', 'dirty bulk', 'clean bulk', 'lean bulk']):
        predicted_tag = 'bulking'
        confidence = 1.0

    elif any(w in msg_lower for w in ['how to cut', 'cutting tips', 'cutting phase', 'shredding', 'get shredded', 'get lean']):
        predicted_tag = 'cutting'
        confidence = 1.0

    elif any(w in msg_lower for w in ['how to track', 'track calories', 'count calories', 'calorie counting', 'track macros', 'track food']):
        predicted_tag = 'calorie_tracking'
        confidence = 1.0

    elif any(w in msg_lower for w in ['intermittent fasting', '16 8', 'eating window', 'skip breakfast', 'if diet']):
        predicted_tag = 'intermittent_fasting'
        confidence = 1.0

    elif any(w in msg_lower for w in ['increase stamina', 'increase endurance', 'get more energy', 'always tired', 'low energy', 'fatigue at gym']):
        predicted_tag = 'stamina'
        confidence = 1.0

    elif any(w in msg_lower for w in ['vegetarian', 'vegan', 'plant based', 'no meat', 'meatless']):
        predicted_tag = 'vegan_diet'
        confidence = 1.0

    elif any(w in msg_lower for w in ['arm fat', 'flabby arms', 'bat wings', 'tone arms', 'lose arm fat']):
        predicted_tag = 'arm_fat'
        confidence = 1.0

    elif any(w in msg_lower for w in ['thigh fat', 'inner thigh', 'slim thighs', 'tone legs', 'lose thigh fat']):
        predicted_tag = 'thigh_fat'
        confidence = 1.0

    elif any(w in msg_lower for w in ['mental health', 'stress and exercise', 'anxiety and gym', 'depression and exercise', 'exercise for mental health']):
        predicted_tag = 'mental_health'
        confidence = 1.0

    elif any(w in msg_lower for w in ['stay consistent', 'how to stay consistent', 'keep going', 'build habit', 'gym habit', 'how to not quit']):
        predicted_tag = 'consistency'
        confidence = 1.0

    elif any(w in msg_lower for w in ['bro split', 'arnold split', 'training split', 'what split should i do', 'what is bro split']):
        predicted_tag = 'training_splits'
        confidence = 1.0

    elif any(w in msg_lower for w in ['healthy snack', 'gym snack', 'snack ideas', 'what to snack', 'pre gym snack']):
        predicted_tag = 'healthy_snacks'
        confidence = 1.0

    elif any(w in msg_lower for w in ['should i do walks', 'mix running', 'running and lifting', 'cardio and weights', 'combine cardio']):
        predicted_tag = 'cardio_lifting'
        confidence = 1.0

    elif any(w in msg_lower for w in ['diet plan', 'meal plan', 'what should i eat', 'give me a diet', 'food plan']):
        predicted_tag = 'diet_general'
        confidence = 1.0

    elif any(w in msg_lower for w in ['workout schedule', 'training plan', 'workout plan', 'make me a schedule', 'training schedule', 'times a week', 'days a week']):
        predicted_tag = 'workout_schedule'
        confidence = 1.0
        override_days = extract_days_from_message(msg_lower)

    elif any(w in msg_lower for w in ['protein intake', 'how much protein', 'protein need', 'my protein', 'protein requirement']):
        predicted_tag = 'protein'
        confidence = 1.0

    elif any(w in msg_lower for w in ['pre meal', 'premeal', 'before gym', 'what to eat before', 'pre workout meal', 'meal before', 'are pre meals']):
        predicted_tag = 'pre_workout_meal'
        confidence = 1.0

    elif any(w in msg_lower for w in ['how many sets', 'how many reps', 'sets and reps', 'rep range', 'how much volume', 'sets of squat', 'sets per']):
        predicted_tag = 'sets_reps'
        confidence = 1.0

    elif any(w in msg_lower for w in ['stretch', 'stretching', 'flexibility', 'warm up', 'warmup', 'cool down', 'cooldown', 'post workout stretch', 'before workout stretch']):
        predicted_tag = 'stretching'
        confidence = 1.0

    elif any(w in msg_lower for w in ['good form', 'proper form', 'how to do', 'technique', 'correct form', 'form for', 'form on']):
        has_squat = any(w in msg_lower for w in ['squat', 'lunge'])
        has_bench = any(w in msg_lower for w in ['bench', 'chest', 'push up', 'pushup'])
        has_deadlift = any(w in msg_lower for w in ['deadlift', 'dead lift'])
        if has_squat and has_bench:
            predicted_tag = 'combined_form'
        elif has_deadlift:
            predicted_tag = 'deadlift_form'
        elif has_bench:
            predicted_tag = 'bench_form'
        elif has_squat:
            predicted_tag = 'squat_form'
        elif any(w in msg_lower for w in ['shoulder', 'overhead', 'ohp']):
            predicted_tag = 'shoulder_workout'
        elif any(w in msg_lower for w in ['curl', 'bicep', 'tricep', 'arm']):
            predicted_tag = 'arm_workout'
        else:
            predicted_tag = 'squat_form'
        confidence = 1.0

    elif any(w in msg_lower for w in ['motivate me', 'no motivation', 'feel like giving up', 'i dont want to', "don't feel like", 'lazy', 'demotivated']):
        predicted_tag = 'motivation'
        confidence = 1.0

    elif any(w in msg_lower for w in ['okay to work out', 'ok to work out', 'workout when sore', 'train when sore', 'exercise when sore', 'if im sore', 'rest day', 'recovery', 'sore muscle', 'overtraining', 'doms']):
        predicted_tag = 'rest_recovery'
        confidence = 1.0

    elif any(w in msg_lower for w in ['how often should i work out', 'how often workout', 'how many times a week should i', 'how frequent']):
        predicted_tag = 'workout_frequency'
        confidence = 1.0

    elif any(w in msg_lower for w in ['how much water', 'water intake', 'hydration', 'drink water']):
        predicted_tag = 'hydration'
        confidence = 1.0

    elif any(w in msg_lower for w in ['manage weight', 'weight management', 'maintain weight', 'control weight']):
        predicted_tag = 'weight_management'
        confidence = 1.0

    elif any(w in msg_lower for w in ['tough workout', 'get through workout', 'hard workout', 'push through', 'belly fat', 'lose belly', 'what to do after cardio', 'after cardio']):
        predicted_tag = 'workout_tips'
        confidence = 1.0

    elif any(w in msg_lower for w in ['calories burn', 'calorie burn', 'how many calories', 'how much calories', 'burn in a run', 'calories running', 'calories walking']):
        predicted_tag = 'calorie_burn'
        confidence = 1.0

    elif any(w in msg_lower for w in ['beginner', 'just started', 'new to gym', 'never worked out', 'starting out', 'where do i start', 'best exercises to start', 'what exercises should i do']):
        predicted_tag = 'beginner_advice'
        confidence = 1.0

    elif any(w in msg_lower for w in ['nutrition', 'what to eat', 'healthy food', 'clean eating', 'eating healthy']):
        predicted_tag = 'diet_general'
        confidence = 1.0

    elif any(w in msg_lower for w in ['chest', 'bench press', 'pec']):
        predicted_tag = 'chest_workout'
        confidence = 1.0

    elif any(w in msg_lower for w in ['back workout', 'lat', 'deadlift', 'pull up', 'pullup', 'pull-up']):
        predicted_tag = 'back_workout'
        confidence = 1.0

    elif any(w in msg_lower for w in ['leg workout', 'squat', 'leg day', 'lunges', 'hamstring', 'glute']):
        predicted_tag = 'leg_workout'
        confidence = 1.0

    elif any(w in msg_lower for w in ['shoulder workout', 'deltoid', 'lateral raise', 'ohp']):
        predicted_tag = 'shoulder_workout'
        confidence = 1.0

    elif any(w in msg_lower for w in ['arm workout', 'bicep', 'tricep', 'curl']):
        predicted_tag = 'arm_workout'
        confidence = 1.0

    elif any(w in msg_lower for w in ['cardio', 'running', 'fat burn', 'hiit', 'endurance', 'stamina']):
        predicted_tag = 'cardio'
        confidence = 1.0

    elif any(w in msg_lower for w in ['walking']):
        predicted_tag = 'cardio_lifting'
        confidence = 1.0

    else:
        processed = preprocess(user_message)
        print(f"Input: '{user_message}' → Processed: '{processed}'")
        predicted_tag = pipeline.predict([processed])[0]
        confidence = pipeline.predict_proba([processed]).max()

    # ─── Low Confidence Fallback ──────────────────────────────────────────────
    if confidence < 0.20:
        response = "I'm not quite sure I understood that 🤔 Try asking about workouts, diet tips, your schedule, or exercises like squats or bench press!"
        return jsonify({"response": response, "tag": "unknown", "confidence": round(float(confidence), 2)})

    # ─── Dynamic Responses ────────────────────────────────────────────────────
    if predicted_tag == 'stretching':
        is_post = any(w in msg_lower for w in ['post', 'after', 'cool down', 'cooldown'])
        is_pre = any(w in msg_lower for w in ['pre', 'before', 'warm up', 'warmup'])
        if is_post:
            response  = "🧘 **Post-Workout Stretches**\n\n"
            response += "Hold each stretch for **30-60 seconds**\n\n"
            response += "**Lower Body:**\n"
            response += "  • Hip flexor stretch — kneel on one knee, push hips forward\n"
            response += "  • Hamstring stretch — sit and reach for your toes\n"
            response += "  • Pigeon pose — great for glutes and hips\n"
            response += "  • Quad stretch — stand, pull foot to glute\n\n"
            response += "**Upper Body:**\n"
            response += "  • Cross-arm shoulder stretch\n"
            response += "  • Overhead tricep stretch\n"
            response += "  • Chest opener — clasp hands behind back\n"
            response += "  • Cat-cow for spine mobility\n\n"
            response += "💡 **Tip:** Post-workout stretching reduces soreness and improves flexibility. Never skip it!"
        elif is_pre:
            response  = "🔥 **Pre-Workout Warm-Up**\n\n"
            response += "Do **dynamic stretches** before training — never static!\n\n"
            response += "**5-Minute Warm-Up:**\n"
            response += "  • Jumping jacks — 30 seconds\n"
            response += "  • Arm circles — 20 reps each direction\n"
            response += "  • Leg swings — 15 reps each leg\n"
            response += "  • Hip circles — 10 reps each direction\n"
            response += "  • Bodyweight squats — 15 reps\n"
            response += "  • Inchworms — 5 reps\n\n"
            response += "💡 **Tip:** Warm-up increases blood flow and reduces injury risk. Always do it before lifting heavy!"
        else:
            response  = "🧘 **Stretching Guide**\n\n"
            response += "**Pre-Workout (Dynamic):** Move through the range of motion\n"
            response += "  • Leg swings, arm circles, hip rotations\n"
            response += "  • Never hold static stretches before lifting!\n\n"
            response += "**Post-Workout (Static):** Hold for 30-60 seconds\n"
            response += "  • Hip flexors, hamstrings, chest, shoulders\n"
            response += "  • This is when you improve flexibility\n\n"
            response += "💡 **Tip:** Ask me about 'pre workout stretches' or 'post workout stretches' for a full routine!"

    elif predicted_tag == 'supplements':
        response  = "💊 **Supplements Guide**\n\n"
        response += "**The Basics (Worth It):**\n"
        response += "  • **Creatine Monohydrate** — increases strength and power. 5g/day ✅\n"
        response += "  • **Whey Protein** — convenient post-workout protein. 25g per scoop ✅\n"
        response += "  • **Caffeine** — improves focus and endurance. Coffee works fine ✅\n\n"
        response += "**Optional:**\n"
        response += "  • **BCAA** — only useful if you train fasted\n"
        response += "  • **Fish Oil** — good for joint health\n"
        response += "  • **Vitamin D** — important if you don't get much sunlight\n\n"
        response += "**Not Worth It:**\n"
        response += "  • Fat burners, testosterone boosters, most proprietary blends\n\n"
        response += "💡 **Tip:** Food first, supplements second. Creatine is the only one with strong evidence for muscle and strength gains!"

    elif predicted_tag == 'fasted_workout':
        response  = "🌅 **Working Out on an Empty Stomach**\n\n"
        response += "**For Fat Loss:** Fasted cardio can burn slightly more fat — best for walking or light jog\n\n"
        response += "**For Strength Training:** NOT recommended — you need fuel to lift heavy\n\n"
        response += "**Best approach:**\n"
        response += "  • Light snack 30-60 mins before (banana, oats, yogurt)\n"
        response += "  • Never do heavy lifting completely fasted\n\n"
        response += "💡 **Tip:** At minimum, have a banana and coffee before morning weight training!"

    elif predicted_tag == 'cheat_meal':
        response  = "🍕 **Cheat Meals & Treats**\n\n"
        response += "**Short answer: Yes, it's okay! 🎉**\n\n"
        response += "**The 80/20 Rule:**\n"
        response += "  • Eat clean 80% of the time\n"
        response += "  • The other 20% won't ruin your progress\n\n"
        response += "**Benefits:**\n"
        response += "  • Refills glycogen stores\n"
        response += "  • Boosts leptin — helps fat burning long term\n"
        response += "  • Improves mental sustainability of your diet\n\n"
        response += "💡 **Tip:** Consistency over weeks matters more than perfection on any single day!"

    elif predicted_tag == 'results_timeline':
        response  = "⏱️ **When Will You See Results?**\n\n"
        response += "  • **Week 1-2:** More energy, better sleep\n"
        response += "  • **Week 3-4:** Strength increases, clothes fit better\n"
        response += "  • **Month 2-3:** Visible muscle tone, noticeable fat loss\n"
        response += "  • **Month 3-6:** Significant body change\n"
        response += "  • **6-12 months:** Dramatic transformation\n\n"
        response += "💡 **Tip:** Take progress photos every 2-4 weeks — the mirror lies but photos don't. Trust the process!"

    elif predicted_tag == 'postpartum':
        response  = "👶 **Safe Exercise After Having a Baby**\n\n"
        response += "**Always consult your doctor first!**\n\n"
        response += "**Timeline:**\n"
        response += "  • 0-6 weeks: Rest, gentle walking only\n"
        response += "  • 6-12 weeks: Light exercise after doctor clearance\n"
        response += "  • 3+ months: Gradually return to normal training\n\n"
        response += "**Safe to start:** Walking, pelvic floor exercises, gentle yoga\n\n"
        response += "**Avoid initially:** Running, jumping, heavy lifting, core exercises\n\n"
        response += "💡 **Tip:** Listen to your body — recovery varies for everyone!"

    elif predicted_tag == 'progressive_overload':
        response  = "📈 **What is Progressive Overload?**\n\n"
        response += "Gradually increasing demands on your muscles so they keep growing.\n\n"
        response += "**Ways to Apply It:**\n"
        response += "  • Add more weight each week\n"
        response += "  • Do more reps (8 → 10)\n"
        response += "  • Do more sets (3 → 4)\n"
        response += "  • Rest less between sets\n\n"
        response += "**Example:**\n"
        response += "  Week 1: Bench 60kg x 3x8\n"
        response += "  Week 2: Bench 60kg x 3x10\n"
        response += "  Week 3: Bench 62.5kg x 3x8\n\n"
        response += "💡 **Tip:** Log your workouts every session — you can't progress what you don't track!"

    elif predicted_tag == 'slim_strong':
        response  = "💪 **Slim & Strong Body**\n\n"
        response += "**Training:**\n"
        response += "  • Resistance training 3-4x/week\n"
        response += "  • Compound lifts: squat, deadlift, bench, rows\n"
        response += "  • Add 2-3 cardio sessions per week\n"
        response += "  • Don't only do cardio — you'll lose muscle too\n\n"
        response += "**Diet:**\n"
        response += "  • Slight calorie deficit (200-300 below maintenance)\n"
        response += "  • High protein: 1g per lb of bodyweight\n"
        response += "  • Don't crash diet — you'll lose muscle not fat\n\n"
        response += "💡 **Tip:** Lifting + calorie deficit + high protein = the formula for a slim strong body!"

    elif predicted_tag == 'lower_body_fat':
        response  = "🔥 **How to Lower Body Fat**\n\n"
        response += "**The Truth:** You can't spot reduce — fat loss happens all over\n\n"
        response += "**Best Exercises:**\n"
        response += "  • HIIT — burns most calories in least time\n"
        response += "  • Compound lifts (squat, deadlift) — burns fat + builds muscle\n"
        response += "  • Steady state cardio — 30-45 min jog\n"
        response += "  • Walking — underrated, great daily calorie burn\n\n"
        response += "**Diet is 80% of the result:**\n"
        response += "  • 300-500 kcal deficit per day\n"
        response += "  • High protein to preserve muscle\n\n"
        response += "💡 **Tip:** Ask for a diet plan — I'll calculate your exact calorie target!"

    elif predicted_tag == 'first_time_gym':
        response  = "🌟 **Tips for Your First Time at the Gym**\n\n"
        response += "**Before You Go:**\n"
        response += "  • Wear comfortable clothes and proper shoes\n"
        response += "  • Bring water, a towel, and headphones\n"
        response += "  • Go during off-peak hours (early morning or midday)\n\n"
        response += "**At the Gym:**\n"
        response += "  • Start with machines — safer for beginners\n"
        response += "  • Ask staff for help — that's what they're there for\n"
        response += "  • Wipe equipment after use\n"
        response += "  • Rest 60-90 seconds between sets\n\n"
        response += "**Focus on:** Form first, light weights, full body 3x/week\n\n"
        response += "💡 **Tip:** Everyone was a beginner once. Nobody is watching you — they're focused on themselves!"

    elif predicted_tag == 'weight_loss_tips':
        response  = "⚡ **How to Lose Weight Effectively**\n\n"
        response += "**The Formula:** Calories in < Calories out\n\n"
        response += "**Top Tips:**\n"
        response += "  • 300-500 kcal deficit per day\n"
        response += "  • Lose 0.5-1kg per week (sustainable pace)\n"
        response += "  • High protein diet — keeps you full longer\n"
        response += "  • Strength train — muscle burns calories at rest\n"
        response += "  • Walk 8,000-10,000 steps daily\n"
        response += "  • Cut liquid calories (soda, juice, alcohol)\n"
        response += "  • Sleep 7-9 hours — poor sleep = more hunger\n\n"
        response += "💡 **Tip:** Set up your profile and ask for a diet plan — I'll calculate your exact calories!"

    elif predicted_tag == 'gain_weight':
        response  = "🍗 **How to Gain Weight the Right Way**\n\n"
        response += "**Eat 300-500 kcal above maintenance**\n"
        response += "Aim for 0.25-0.5kg gained per week\n\n"
        response += "**Best Foods:**\n"
        response += "  • Rice, oats, bread, pasta (carbs = calories)\n"
        response += "  • Chicken, beef, eggs, fish (protein = muscle)\n"
        response += "  • Peanut butter, avocado, nuts (healthy fats)\n\n"
        response += "**Training:** Compound lifts + progressive overload every week\n\n"
        response += "💡 **Tip:** Eat every 3-4 hours. Add a peanut butter smoothie if you struggle to eat enough!"

    elif predicted_tag == 'abs_workout':
        response  = "🔥 **Abs & Core Workout**\n\n"
        response += "**The Truth:** Abs are made in the kitchen — diet is key!\n"
        response += "You need low body fat to see them (12-15% men, 18-22% women)\n\n"
        response += "**Best Ab Exercises:**\n"
        response += "  • Plank — 3x30-60 seconds\n"
        response += "  • Hanging Leg Raises — 3x12\n"
        response += "  • Cable Crunches — 3x15\n"
        response += "  • Ab Wheel Rollout — 3x10\n"
        response += "  • Dead Bug — 3x10 each side\n\n"
        response += "💡 **Tip:** Train abs 2-3x/week. Squat and deadlift are also great for core strength!"

    elif predicted_tag == 'sleep':
        response  = "😴 **Sleep & Muscle Growth**\n\n"
        response += "**How much:** Minimum 7 hours, optimal 8-9 for athletes\n\n"
        response += "**Why it matters:**\n"
        response += "  • Growth hormone released during deep sleep\n"
        response += "  • Muscle repair happens while you rest\n"
        response += "  • Poor sleep increases cortisol (kills gains)\n"
        response += "  • Bad sleep = more hunger and cravings\n\n"
        response += "**Better Sleep Tips:**\n"
        response += "  • Same sleep/wake time daily\n"
        response += "  • No screens 30 mins before bed\n"
        response += "  • Cool, dark room\n\n"
        response += "💡 **Tip:** Sleep is when you GROW. Training breaks muscle down, sleep builds it back!"

    elif predicted_tag == 'injury':
        response  = "🩹 **Dealing with Workout Injuries**\n\n"
        response += "**Use R.I.C.E:**\n"
        response += "  • **Rest** — stop immediately\n"
        response += "  • **Ice** — 15-20 mins every 2-3 hours\n"
        response += "  • **Compression** — wrap to reduce swelling\n"
        response += "  • **Elevation** — raise above heart level\n\n"
        response += "**Common Causes:** Poor form, too much weight, skipping warmup, overtraining\n\n"
        response += "**See a doctor if:** Sharp pain, swelling that doesn't go down, pain lasting 1+ week\n\n"
        response += "💡 **Tip:** Never train through sharp pain. Work other muscle groups while you heal!"

    elif predicted_tag == 'gym_essentials':
        response  = "🎒 **Gym Bag Essentials**\n\n"
        response += "**Must Haves:** Water bottle (1L+), towel, proper shoes, comfortable clothes, headphones\n\n"
        response += "**Nice to Have:**\n"
        response += "  • Lifting belt (heavy compound lifts)\n"
        response += "  • Wrist wraps (bench/OHP)\n"
        response += "  • Knee sleeves (squats)\n"
        response += "  • Post-workout protein shake\n\n"
        response += "💡 **Tip:** You don't need fancy gear to start. Water, towel, and shoes is all you really need!"

    elif predicted_tag == 'home_workout':
        response  = "🏠 **Home Workout — No Equipment Needed**\n\n"
        response += "**Full Body Routine (3x/week):**\n\n"
        response += "**Push:** Push-ups 3x max · Pike Push-ups 3x10 · Chair Dips 3x12\n\n"
        response += "**Pull:** Superman holds 3x15 · Doorframe rows (if available) 3x10\n\n"
        response += "**Legs:** Squats 3x20 · Lunges 3x15 each · Glute bridges 3x20\n\n"
        response += "**Core:** Plank 3x45s · Crunches 3x20\n\n"
        response += "💡 **Tip:** Add a backpack with books for extra resistance as you get stronger!"

    elif predicted_tag == 'plateau':
        response  = "📊 **Breaking Through a Plateau**\n\n"
        response += "**Why it happens:** Your body adapted to the same routine\n\n"
        response += "**How to Break It:**\n"
        response += "  • Change your exercises — new stimulus = new growth\n"
        response += "  • Increase weight, reps, or sets\n"
        response += "  • Take a deload week (lighter weights)\n"
        response += "  • Recalculate your calories\n"
        response += "  • Check sleep and stress levels\n\n"
        response += "💡 **Tip:** A plateau means your body got efficient — that's progress. Time to level up!"

    elif predicted_tag == 'breathing':
        response  = "💨 **How to Breathe When Lifting**\n\n"
        response += "**The Rule:** Exhale on effort, inhale on the easier part\n\n"
        response += "**Examples:**\n"
        response += "  • Bench Press: Inhale down → exhale pushing up\n"
        response += "  • Squat: Inhale down → exhale coming up\n"
        response += "  • Deadlift: Big breath before → exhale at top\n\n"
        response += "**For heavy lifts — Valsalva Maneuver:**\n"
        response += "  • Big breath into belly, brace core tight, hold through the lift\n\n"
        response += "💡 **Tip:** Never hold your breath for multiple reps — only for max effort singles!"

    elif predicted_tag == 'gym_anxiety':
        response  = "😰 **Dealing with Gym Anxiety**\n\n"
        response += "**You're not alone — most beginners feel this!**\n\n"
        response += "**Tips:**\n"
        response += "  • Go during off-peak hours (early morning, midday)\n"
        response += "  • Have a plan — know your workout before you go\n"
        response += "  • Use headphones — creates your own bubble\n"
        response += "  • Start with machines, not free weights\n"
        response += "  • Bring a friend\n\n"
        response += "**Remember:** Everyone is focused on themselves. Even the biggest guy was once a beginner!\n\n"
        response += "💡 **Tip:** After 2-3 weeks of going regularly, the anxiety disappears completely!"

    elif predicted_tag == 'post_workout_meal':
        response  = "🍗 **What to Eat After a Workout**\n\n"
        response += "**Eat within 30-60 minutes after training**\n\n"
        response += "**You need:** Protein (repair muscle) + Carbs (replenish glycogen)\n\n"
        response += "**Best Meals:**\n"
        response += "  • Chicken + rice + vegetables\n"
        response += "  • Whey protein shake + banana\n"
        response += "  • Eggs + toast\n"
        response += "  • Greek yogurt + granola + berries\n"
        response += "  • Tuna + rice cakes\n\n"
        response += "💡 **Tip:** Don't skip this meal — it's when your muscles absorb nutrients best!"

    elif predicted_tag == 'bulking':
        response  = "📈 **How to Bulk Properly**\n\n"
        response += "**Eat 300-500 kcal above maintenance**\n"
        response += "Aim for 0.25-0.5kg gained per week (more = mostly fat)\n\n"
        response += "**Macros:** Protein 1g/lb · Carbs 40-50% · Fats 25-35%\n\n"
        response += "**Training:** Heavy compound lifts + progressive overload\n"
        response += "Limit cardio to 1-2 light sessions/week\n\n"
        response += "💡 **Tip:** Dirty bulking just makes you fat. Clean bulk = slow and steady muscle gain!"

    elif predicted_tag == 'cutting':
        response  = "✂️ **How to Cut Properly**\n\n"
        response += "**Eat 300-500 kcal below maintenance**\n"
        response += "Lose 0.5-1kg per week max (faster = muscle loss)\n\n"
        response += "**Macros:** Protein 1-1.2g/lb (protect muscle!) · Carbs 30-40% · Fats 20-30%\n\n"
        response += "**Training:** Keep lifting heavy! Add 2-3 cardio sessions/week\n\n"
        response += "💡 **Tip:** The biggest cutting mistake is dropping calories too fast. Slow and steady!"

    elif predicted_tag == 'calorie_tracking':
        response  = "📱 **How to Track Calories & Macros**\n\n"
        response += "**Best Free Apps:** MyFitnessPal · Cronometer · Lose It\n\n"
        response += "**How to Track:**\n"
        response += "  1. Weigh food with a kitchen scale (most accurate)\n"
        response += "  2. Log everything — even small snacks count\n"
        response += "  3. Pre-log meals the night before\n"
        response += "  4. Track for at least 2 weeks consistently\n\n"
        response += "💡 **Tip:** Track for 2-4 weeks to learn your portions. After that you'll intuitively know!"

    elif predicted_tag == 'intermittent_fasting':
        response  = "⏰ **Intermittent Fasting Guide**\n\n"
        response += "**Most Popular: 16/8 Method**\n"
        response += "  • Fast 16 hours, eat within 8 hour window\n"
        response += "  • Example: Eat 12pm-8pm, fast 8pm-12pm\n\n"
        response += "**Benefits:** Easier calorie control · Improved insulin sensitivity · Aids fat loss\n\n"
        response += "**Downsides:** Not ideal for muscle building · Hurts performance if you train in the morning\n\n"
        response += "💡 **Tip:** IF is just a tool — it works if it fits your lifestyle. Calories still matter!"

    elif predicted_tag == 'stamina':
        response  = "⚡ **How to Build Stamina & Energy**\n\n"
        response += "**Training:**\n"
        response += "  • Add cardio 2-3x/week — start low, build up\n"
        response += "  • HIIT is fastest way to improve cardio fitness\n"
        response += "  • Zone 2 cardio (conversational pace) builds aerobic base\n\n"
        response += "**Lifestyle:**\n"
        response += "  • Sleep 7-9 hours — biggest energy booster\n"
        response += "  • Stay hydrated — dehydration kills energy\n"
        response += "  • Eat enough carbs — primary fuel source\n\n"
        response += "💡 **Tip:** Stamina improves fast — 2-3 weeks of consistent cardio and you'll notice a big difference!"

    elif predicted_tag == 'vegan_diet':
        response  = "🌱 **Vegan/Vegetarian Fitness Diet**\n\n"
        response += "**Can you build muscle without meat? YES! ✅**\n\n"
        response += "**Best Plant Protein Sources:**\n"
        response += "  • Tofu — 17g per 100g\n"
        response += "  • Tempeh — 19g per 100g\n"
        response += "  • Lentils — 18g per cup\n"
        response += "  • Chickpeas — 15g per cup\n"
        response += "  • Seitan — 25g per 100g\n"
        response += "  • Pea protein powder — 20-25g per scoop\n\n"
        response += "**Watch out for:** B12 (supplement), Iron (leafy greens + vitamin C), Omega 3 (flaxseed, chia)\n\n"
        response += "💡 **Tip:** Combine proteins (rice + beans) to get all essential amino acids!"

    elif predicted_tag == 'arm_fat':
        response  = "💪 **How to Lose Arm Fat**\n\n"
        response += "**The Truth:** You cannot spot reduce — fat loss happens all over\n\n"
        response += "**What Works:** Calorie deficit + full body training + cardio\n\n"
        response += "**Best Arm Toning Exercises:**\n"
        response += "  • Tricep dips — 3x15\n"
        response += "  • Overhead tricep extension — 3x12\n"
        response += "  • Bicep curls — 3x12\n"
        response += "  • Push-ups — 3x max\n"
        response += "  • Diamond push-ups — 3x10\n\n"
        response += "💡 **Tip:** Lose fat through diet, build muscle through training — that's how you get toned arms!"

    elif predicted_tag == 'thigh_fat':
        response  = "🦵 **How to Lose Thigh Fat**\n\n"
        response += "**The Truth:** Spot reduction isn't possible — lose fat all over\n\n"
        response += "**Best Exercises:**\n"
        response += "  • Squats — targets quads, glutes, hamstrings\n"
        response += "  • Sumo squats — great for inner thighs\n"
        response += "  • Lunges — 3x15 each leg\n"
        response += "  • Hip abduction machine\n"
        response += "  • Cycling — great cardio for legs\n\n"
        response += "**Diet:** Calorie deficit + high protein + reduce sodium (reduces water retention)\n\n"
        response += "💡 **Tip:** Building leg muscle while losing fat creates the toned look — don't skip leg day!"

    elif predicted_tag == 'mental_health':
        response  = "🧠 **Exercise & Mental Health**\n\n"
        response += "**Exercise is one of the most powerful mood boosters:**\n\n"
        response += "**What it does for your brain:**\n"
        response += "  • Releases endorphins — natural mood boost\n"
        response += "  • Reduces cortisol (stress hormone)\n"
        response += "  • Improves sleep quality\n"
        response += "  • Boosts self-confidence\n"
        response += "  • Reduces anxiety and depression symptoms\n\n"
        response += "**Best for mental health:** Any consistent exercise helps\n"
        response += "Cardio = immediate mood boost · Strength = long term confidence · Yoga = reduces anxiety\n\n"
        response += "💡 **Tip:** Even a 20 minute walk can significantly improve your mood. Start there!"

    elif predicted_tag == 'consistency':
        response  = "🔒 **How to Stay Consistent at the Gym**\n\n"
        response += "**Build the Habit:**\n"
        response += "  • Schedule gym like a meeting — non-negotiable\n"
        response += "  • Same time every day (morning works best for most)\n"
        response += "  • Start with 3 days/week — don't overwhelm yourself\n"
        response += "  • Prepare gym bag the night before\n\n"
        response += "**Stay Motivated:**\n"
        response += "  • Track progress — photos, measurements, strength logs\n"
        response += "  • Find a workout partner\n"
        response += "  • Set short term goals (4 week challenges)\n\n"
        response += "**When You Miss a Day:** Don't try to make up for it. Just continue.\n"
        response += "Missing one day is fine, missing two is a habit.\n\n"
        response += "💡 **Tip:** Motivation gets you started, discipline keeps you going!"

    elif predicted_tag == 'training_splits':
        response  = "📋 **Training Splits Explained**\n\n"
        response += "**Full Body (3x/week) — Best for Beginners**\n"
        response += "  Train all muscles every session. More frequency = faster learning.\n\n"
        response += "**Upper/Lower (4x/week) — Intermediate**\n"
        response += "  2 upper days + 2 lower days. Good balance of volume and recovery.\n\n"
        response += "**Push/Pull/Legs (6x/week) — Intermediate/Advanced**\n"
        response += "  Push: Chest/shoulders/triceps · Pull: Back/biceps · Legs: everything lower\n\n"
        response += "**Bro Split (5x/week) — Advanced**\n"
        response += "  One muscle group per day. High volume per muscle.\n\n"
        response += "💡 **Tip:** Full Body or Upper/Lower gives best results for most people. PPL once you're past beginner stage!"

    elif predicted_tag == 'healthy_snacks':
        response  = "🍎 **Healthy Gym Snacks**\n\n"
        response += "**Pre-Workout (30-60 min before):**\n"
        response += "  • Banana — quick carbs + potassium\n"
        response += "  • Rice cakes + peanut butter\n"
        response += "  • Oats with honey\n"
        response += "  • Apple + almond butter\n\n"
        response += "**Post-Workout:**\n"
        response += "  • Whey protein shake + fruit\n"
        response += "  • Greek yogurt + berries\n"
        response += "  • Eggs on toast\n\n"
        response += "**Anytime:**\n"
        response += "  • Almonds or walnuts · Hard boiled eggs · Edamame · Hummus + veggies\n\n"
        response += "💡 **Tip:** Prep snacks in advance — when hungry you'll grab whatever is convenient!"

    elif predicted_tag == 'cardio_lifting':
        response  = "🏃 **Should You Mix Cardio & Lifting?**\n\n"
        response += "**Yes — but order matters!**\n\n"
        response += "  1. Lift weights FIRST\n"
        response += "  2. Do cardio AFTER\n\n"
        response += "**Why?** Lifting needs max energy. Cardio after burns more fat since glycogen is depleted.\n\n"
        response += "**Walking specifically:**\n"
        response += "  • Low impact, great for recovery\n"
        response += "  • 8,000-10,000 steps/day burns significant calories\n"
        response += "  • Won't interfere with muscle growth\n"
        response += "  • Perfect for rest days\n\n"
        response += "💡 **Tip:** Don't do intense cardio the day before heavy leg day — your performance will suffer!"

    elif predicted_tag == 'calorie_burn':
        if user_profile.get('weight'):
            weight_kg = user_profile['weight']
            run_cals = int(weight_kg * 0.0175 * 9.8 * 60)
            walk_cals = int(weight_kg * 0.0175 * 3.8 * 60)
            response  = f"🔥 **Calorie Burn Estimates** (based on your {weight_kg}kg)\n\n"
            response += f"  • Running (1 hour): ~{run_cals} kcal\n"
            response += f"  • Walking (1 hour): ~{walk_cals} kcal\n"
            response += f"  • Cycling (1 hour): ~{int(weight_kg * 0.0175 * 7.5 * 60)} kcal\n"
            response += f"  • Swimming (1 hour): ~{int(weight_kg * 0.0175 * 8.0 * 60)} kcal\n"
            response += f"  • Weight Training (1 hour): ~{int(weight_kg * 0.0175 * 5.0 * 60)} kcal\n\n"
            response += "💡 **Tip:** These are estimates. Actual burn depends on intensity, age, and fitness level."
        else:
            response  = "🔥 **General Calorie Burn (per hour, 70kg person)**\n\n"
            response += "  • Running — ~700 kcal\n"
            response += "  • Cycling — ~550 kcal\n"
            response += "  • Swimming — ~600 kcal\n"
            response += "  • Walking — ~280 kcal\n"
            response += "  • Weight Training — ~370 kcal\n\n"
            response += "💡 **Tip:** Set up your profile for a personalized calculation based on your weight!"

    elif predicted_tag == 'workout_frequency':
        response  = "📅 **How Often Should You Work Out?**\n\n"
        response += "**By Experience:**\n"
        response += "  • Beginner: 3 days/week\n"
        response += "  • Intermediate: 4 days/week\n"
        response += "  • Advanced: 5-6 days/week\n\n"
        response += "**By Goal:**\n"
        response += "  • Weight Loss: 4-5 days (mix cardio + weights)\n"
        response += "  • Muscle Gain: 3-5 days (resistance training)\n"
        response += "  • General Fitness: 3 days is enough to start\n\n"
        response += "💡 **Tip:** More is not always better. Recovery is when you grow — never skip rest days!"

    elif predicted_tag == 'hydration':
        response  = "💧 **How Much Water Should You Drink?**\n\n"
        response += "**General Rule:** 35ml per kg of bodyweight per day\n\n"
        if user_profile.get('weight'):
            daily = round(user_profile['weight'] * 0.035, 1)
            response += f"**Your target:** ~{daily}L per day based on your {user_profile['weight']}kg\n\n"
        response += "**Drink more when:** Training (add 500ml-1L/hour) · Hot weather · Intense workouts\n\n"
        response += "**Signs of dehydration:** Dark yellow urine · Headaches · Muscle cramps\n\n"
        response += "💡 **Tip:** Drink a glass of water first thing every morning!"

    elif predicted_tag == 'weight_management':
        response  = "⚖️ **Weight Management Guide**\n\n"
        response += "**To Lose Weight:** Calorie deficit (300-500 below) · High protein · Cardio + weights\n\n"
        response += "**To Gain Weight:** Calorie surplus (300-500 above) · Resistance training · Eat every 3-4 hours\n\n"
        response += "**To Maintain:** Eat at maintenance · Stay consistent with exercise\n\n"
        response += "💡 **Tip:** Set up your profile and ask for a diet plan — I'll calculate your exact calorie target!"

    elif predicted_tag == 'workout_tips':
        response  = "💪 **Workout Tips**\n\n"
        response += "**To Push Through a Tough Workout:**\n"
        response += "  • Break it into chunks — 'just 5 more reps'\n"
        response += "  • Music helps a lot\n"
        response += "  • Remember your WHY\n\n"
        response += "**To Lose Belly Fat:** Can't spot reduce — full body fat loss through diet + cardio\n\n"
        response += "**After Cardio:** Stretch 5-10 mins · Rehydrate · Eat protein within 30-60 mins\n\n"
        response += "💡 **Tip:** Consistency beats perfection every single time!"

    elif predicted_tag == 'protein':
        if not user_profile.get('weight'):
            response = "Please **set up your profile first** so I can calculate your exact protein needs! Click the profile icon 👆"
        else:
            weight_lbs = user_profile['weight'] * 2.205
            min_protein = int(weight_lbs * 0.8)
            max_protein = int(weight_lbs * 1.0)
            response  = f"🥩 **Your Daily Protein Target**\n\n"
            response += f"**Your Weight:** {user_profile['weight']}kg\n"
            response += f"**Recommended:** {min_protein}g – {max_protein}g per day\n\n"
            response += f"**Best Sources:**\n"
            response += f"  • Chicken breast — 31g per 100g\n"
            response += f"  • Eggs — 6g per egg\n"
            response += f"  • Greek yogurt — 17g per cup\n"
            response += f"  • Whey protein shake — 25g per scoop\n"
            response += f"  • Tuna — 30g per can\n\n"
            response += f"💡 **Tip:** Spread protein across 4-5 meals for maximum absorption!"

    elif predicted_tag == 'pre_workout_meal':
        response  = "🍌 **Pre-Workout Nutrition**\n\n"
        response += "**Eat before training — YES, it fuels performance!**\n\n"
        response += "**Timing:** Large meal 2-3 hours before · Snack 30-60 mins before\n\n"
        response += "**Best Foods:** Banana + peanut butter · Oats with honey · Rice + chicken · Greek yogurt + berries\n\n"
        response += "**Avoid:** High fat foods · High fiber foods · Eating right before training\n\n"
        response += "💡 **Tip:** Carbs = fuel. Don't train fasted unless you're used to it!"

    elif predicted_tag == 'sets_reps':
        response  = "📊 **Sets & Reps Guide**\n\n"
        response += "**Strength:** 3-5 sets × 3-6 reps · Rest 3-5 mins\n"
        response += "**Muscle Growth:** 3-4 sets × 8-12 reps · Rest 60-90 secs\n"
        response += "**Endurance:** 2-3 sets × 15-20 reps · Rest 30-60 secs\n\n"
        response += "**Per workout:**\n"
        response += "  • Compound lifts (squat, bench): 4-5 sets\n"
        response += "  • Isolation exercises (curls): 3 sets\n\n"
        response += "💡 **Tip:** Beginners do best starting with 3×10 on all exercises!"

    elif predicted_tag == 'workout_schedule':
        if not user_profile.get('weight'):
            response = "I'd love to build you a schedule! Please **set up your profile first** so I can personalize it. Click the profile icon 👆"
        else:
            plan, tip = generate_schedule(user_profile, override_days)
            schedule_text  = f"🗓️ **Your {plan['type']} Schedule**\n\n"
            for session in plan['schedule']:
                schedule_text += f"**{session['day']} — {session['focus']}**\n"
                for ex in session['exercises']:
                    schedule_text += f"  • {ex}\n"
                schedule_text += "\n"
            schedule_text += f"💡 **Pro Tip:** {tip}"
            response = schedule_text

    elif predicted_tag == 'diet_general':
        if not user_profile.get('weight'):
            response = "To give you a personalized diet plan, please **set up your profile** first! Click the profile icon 👆"
        else:
            diet = generate_diet_plan(user_profile)
            response  = f"🥗 **Your Personalized Diet Plan**\n\n"
            response += f"**Goal:** {diet['meal_focus']}\n"
            response += f"**Daily Calories:** {diet['daily_calories']} kcal\n\n"
            response += f"**Macros:**\n"
            response += f"  • Protein: {diet['macros']['protein_g']}g\n"
            response += f"  • Carbs: {diet['macros']['carbs_g']}g\n"
            response += f"  • Fats: {diet['macros']['fats_g']}g\n\n"
            response += "**Sample Meals:**\n"
            for meal in diet['meals']:
                response += f"  🍽️ **{meal['name']}:** {meal['example']}\n"

    elif predicted_tag == 'squat_form':
        response  = "🦵 **Proper Squat Form**\n\n"
        response += "**Setup:** Bar on upper traps · Feet shoulder-width · Toes slightly out (30°)\n\n"
        response += "**The Movement:**\n"
        response += "  1. Deep breath, brace your core\n"
        response += "  2. Push knees out in line with toes\n"
        response += "  3. Sit back and down — break parallel\n"
        response += "  4. Keep chest up\n"
        response += "  5. Drive through heels to stand\n\n"
        response += "**Common Mistakes:** ❌ Knees caving · ❌ Heels lifting · ❌ Not going deep enough\n\n"
        response += "💡 **Tip:** Film yourself from the side — chest dropping before hips rise = too heavy!"

    elif predicted_tag == 'bench_form':
        response  = "🏋️ **Proper Bench Press Form**\n\n"
        response += "**Setup:** Retract shoulder blades · Slight arch · Grip wider than shoulder-width\n\n"
        response += "**The Movement:**\n"
        response += "  1. Unrack with straight arms\n"
        response += "  2. Lower bar slowly to mid-chest\n"
        response += "  3. Elbows at 45° — not flared\n"
        response += "  4. Drive up and slightly back\n"
        response += "  5. Lock out at top\n\n"
        response += "**Common Mistakes:** ❌ Bouncing bar · ❌ Elbows flaring · ❌ Butt lifting\n\n"
        response += "💡 **Tip:** Think 'push yourself away from the bar' not 'push the bar up'!"

    elif predicted_tag == 'deadlift_form':
        response  = "🏋️ **Proper Deadlift Form**\n\n"
        response += "**Setup:** Bar over mid-foot · Hip-width stance · Grip outside legs · Chest up\n\n"
        response += "**The Movement:**\n"
        response += "  1. Big breath, brace core hard\n"
        response += "  2. Push the floor away — leg drive first\n"
        response += "  3. Bar drags up your shins\n"
        response += "  4. Hips and shoulders rise together\n"
        response += "  5. Squeeze glutes at the top\n\n"
        response += "**Common Mistakes:** ❌ Rounding back · ❌ Bar drifting away · ❌ Jerking the bar\n\n"
        response += "💡 **Tip:** Deadlift is a PUSH not a pull — push the ground away with your legs!"

    elif predicted_tag == 'combined_form':
        response  = "🏋️ **Squat & Bench Press Form**\n\n"
        response += "**Squat:** Feet shoulder-width · Chest up · Break parallel · Drive through heels\n\n"
        response += "**Bench Press:** Retract shoulder blades · Lower to mid-chest · Elbows 45° · Drive through chest\n\n"
        response += "💡 **Tip:** Film yourself from the side to check form on both lifts!"

    elif predicted_tag == 'bmi':
        if not user_profile.get('weight') or not user_profile.get('height'):
            response = "To calculate your BMI, I need your **weight and height**. Please set up your profile first! Click the profile icon 👆"
        else:
            bmi, category = calculate_bmi(user_profile['weight'], user_profile['height'])
            response  = f"📊 **Your BMI Results**\n\n"
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

    elif predicted_tag == 'profile':
        response = "To update your profile, click the **profile icon** in the bottom left corner. You can set your age, weight, height, fitness goal, and experience level there!"

    else:
        response = get_response_for_tag(predicted_tag)

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

if __name__ == '__main__':
    print("\n🏋️  GymBuddy AI Backend Running")
    print("API: http://localhost:5000")
    print("Health check: http://localhost:5000/health\n")
    app.run(host='0.0.0.0', port=5000, debug=False)