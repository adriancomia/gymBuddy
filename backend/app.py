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

    elif any(w in msg_lower for w in ['creatine', 'supplement', 'pre workout', 'preworkout', 'whey', 'bcaa', 'protein powder', 'mass gainer']):
        predicted_tag = 'supplements'
        confidence = 1.0

    elif any(w in msg_lower for w in ['should i do walks', 'walking', 'should i walk', 'mix running', 'running and lifting', 'cardio and weights', 'combine cardio']):
        predicted_tag = 'cardio_lifting'
        confidence = 1.0

    elif any(w in msg_lower for w in ['sweet treat', 'cheat meal', 'cheat day', 'junk food', 'can i eat', 'is it okay to eat', 'is it ok to eat', 'unhealthy food']):
        predicted_tag = 'cheat_meal'
        confidence = 1.0

    elif any(w in msg_lower for w in ['diet plan', 'meal plan', 'what should i eat', 'give me a diet', 'food plan']):
        predicted_tag = 'diet_general'
        confidence = 1.0

    elif any(w in msg_lower for w in ['workout schedule', 'training plan', 'workout plan', 'make me a schedule', 'training schedule', 'times a week', 'days a week']):
        predicted_tag = 'workout_schedule'
        confidence = 1.0
        override_days = extract_days_from_message(msg_lower)

    elif any(w in msg_lower for w in ['protein intake', 'how much protein', 'protein need', 'my protein', 'protein requirement', 'protein use my weight']):
        predicted_tag = 'protein'
        confidence = 1.0

    elif any(w in msg_lower for w in ['pre meal', 'premeal', 'pre-meal', 'before gym', 'what to eat before', 'pre workout meal', 'meal before', 'are pre meals']):
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

    elif any(w in msg_lower for w in ['rest day', 'recovery', 'sore muscle', 'how many rest', 'overtraining', 'doms']):
        predicted_tag = 'rest_recovery'
        confidence = 1.0

    elif any(w in msg_lower for w in ['beginner', 'just started', 'new to gym', 'never worked out', 'starting out', 'where do i start']):
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

    elif any(w in msg_lower for w in ['calories burn', 'calorie burn', 'how many calories', 'how much calories', 'burn in a run', 'calories running', 'calories walking', 'calories workout']):
        predicted_tag = 'calorie_burn'
        confidence = 1.0

    elif any(w in msg_lower for w in ['okay to work out', 'ok to work out', 'workout when sore', 'train when sore', 'exercise when sore', 'if im sore']):
        predicted_tag = 'rest_recovery'
        confidence = 1.0

    elif any(w in msg_lower for w in ['best exercises to start', 'exercises to start', 'what exercises should i do', 'exercises to lose fat', 'exercises to build muscle', 'best type of exercise', 'exercises for heart']):
        predicted_tag = 'beginner_advice'
        confidence = 1.0

    elif any(w in msg_lower for w in ['how often should i work out', 'how often workout', 'how many times a week should i', 'how frequent']):
        predicted_tag = 'workout_frequency'
        confidence = 1.0

    elif any(w in msg_lower for w in ['how much water', 'water intake', 'how much water should', 'hydration', 'drink water']):
        predicted_tag = 'hydration'
        confidence = 1.0

    elif any(w in msg_lower for w in ['manage weight', 'weight management', 'maintain weight', 'control weight']):
        predicted_tag = 'weight_management'
        confidence = 1.0

    elif any(w in msg_lower for w in ['tough workout', 'get through workout', 'hard workout', 'push through', 'belly fat', 'lose belly', 'what to do after cardio', 'after cardio']):
        predicted_tag = 'workout_tips'
        confidence = 1.0    

    elif any(w in msg_lower for w in ['empty stomach', 'fasted workout', 'fasted training', 'workout without eating']):
        predicted_tag = 'fasted_workout'
        confidence = 1.0

    elif any(w in msg_lower for w in ['how long until', 'when will i see results', 'visible results', 'how long to see']):
        predicted_tag = 'results_timeline'
        confidence = 1.0

    elif any(w in msg_lower for w in ['safe exercise', 'after having a baby', 'postpartum', 'after pregnancy']):
        predicted_tag = 'postpartum'
        confidence = 1.0

    elif any(w in msg_lower for w in ['progressive overload', 'what is progressive', 'how to progress']):
        predicted_tag = 'progressive_overload'
        confidence = 1.0

    elif any(w in msg_lower for w in ['slim and strong', 'slim strong', 'lean and strong', 'toned body', 'slim body']):
        predicted_tag = 'slim_strong'
        confidence = 1.0

    elif any(w in msg_lower for w in ['lower body fat', 'exercise to lower body fat', 'reduce body fat', 'lose body fat']):
        predicted_tag = 'lower_body_fat'
        confidence = 1.0

    elif any(w in msg_lower for w in ['how often should i work out', 'how often workout', 'how frequent']):
        predicted_tag = 'workout_frequency'
        confidence = 1.0

    elif any(w in msg_lower for w in ['how much water', 'water intake', 'hydration', 'drink water']):
        predicted_tag = 'hydration'
        confidence = 1.0      

    elif any(w in msg_lower for w in ['fasting while', 'fasted workout', 'fasted training', 'empty stomach', 'workout without eating']):
        predicted_tag = 'fasted_workout'
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

    elif any(w in msg_lower for w in ['sleep', 'how much sleep', 'sleep for muscle', 'sleep and gym', 'sleep recovery']):
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

    elif any(w in msg_lower for w in ['what to eat after', 'post workout food', 'after workout meal', 'eat after gym']):
        predicted_tag = 'post_workout_meal'
        confidence = 1.0

    elif any(w in msg_lower for w in ['how to bulk', 'bulking tips', 'dirty bulk', 'clean bulk', 'lean bulk']):
        predicted_tag = 'bulking'
        confidence = 1.0

    elif any(w in msg_lower for w in ['how to cut', 'cutting tips', 'cutting phase', 'shredding', 'get shredded', 'get lean']):
        predicted_tag = 'cutting'
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
        response += "  • **Creatine Monohydrate** — increases strength and power output. 5g/day, most researched supplement ✅\n"
        response += "  • **Whey Protein** — convenient protein source post-workout. 25g per scoop ✅\n"
        response += "  • **Caffeine** — improves focus and endurance. Coffee works fine ✅\n\n"
        response += "**Optional:**\n"
        response += "  • **BCAA** — only useful if you train fasted\n"
        response += "  • **Fish Oil** — good for joint health and inflammation\n"
        response += "  • **Vitamin D** — important if you don't get much sunlight\n\n"
        response += "**Not Worth It:**\n"
        response += "  • Fat burners, testosterone boosters, most 'proprietary blends'\n\n"
        response += "💡 **Tip:** Food first, supplements second. Creatine is the only supplement with strong evidence for muscle and strength gains!"

    elif predicted_tag == 'cardio_lifting':
        response  = "🏃 **Should You Mix Running & Lifting?**\n\n"
        response += "**Yes — but order matters!**\n\n"
        response += "**Best Order:**\n"
        response += "  1. Lift weights FIRST\n"
        response += "  2. Do cardio AFTER\n\n"
        response += "**Why?** Lifting requires maximum energy and focus. Cardio after burns more fat since glycogen is depleted.\n\n"
        response += "**Walking specifically:**\n"
        response += "  • Walking is low impact and great for recovery\n"
        response += "  • 8,000-10,000 steps/day burns significant calories\n"
        response += "  • Won't interfere with muscle growth at all\n"
        response += "  • Great to do on rest days\n\n"
        response += "💡 **Tip:** Don't do intense cardio the day before a heavy leg day — your performance will suffer!"

    elif predicted_tag == 'cheat_meal':
        response  = "🍕 **Cheat Meals & Treats**\n\n"
        response += "**Short answer: Yes, it's okay! 🎉**\n\n"
        response += "**The 80/20 Rule:**\n"
        response += "  • Eat clean 80% of the time\n"
        response += "  • The other 20% won't ruin your progress\n\n"
        response += "**Benefits of cheat meals:**\n"
        response += "  • Refills glycogen stores (great before a big workout)\n"
        response += "  • Boosts leptin levels — helps fat burning long term\n"
        response += "  • Improves mental sustainability of your diet\n\n"
        response += "**Tips:**\n"
        response += "  • Plan it — don't let it become a cheat weekend\n"
        response += "  • Enjoy it guilt-free, then get back on track\n"
        response += "  • Once a week is fine for most people\n\n"
        response += "💡 **Tip:** Consistency over weeks matters more than perfection on any single day!"

    elif predicted_tag == 'squat_form':
        response  = "🦵 **Proper Squat Form**\n\n"
        response += "**Setup:**\n"
        response += "  • Bar on upper traps, feet shoulder-width apart\n"
        response += "  • Toes pointed slightly outward (30°)\n\n"
        response += "**The Movement:**\n"
        response += "  1. Take a deep breath, brace your core\n"
        response += "  2. Push knees out in line with toes\n"
        response += "  3. Sit back and down — break parallel\n"
        response += "  4. Keep chest up, don't let it cave forward\n"
        response += "  5. Drive through heels to stand up\n\n"
        response += "**Common Mistakes:**\n"
        response += "  ❌ Knees caving inward\n"
        response += "  ❌ Heels coming off the floor\n"
        response += "  ❌ Not going deep enough\n\n"
        response += "💡 **Tip:** Film yourself from the side — if your chest drops before hips rise, the weight is too heavy!"

    elif predicted_tag == 'bench_form':
        response  = "🏋️ **Proper Bench Press Form**\n\n"
        response += "**Setup:**\n"
        response += "  • Retract shoulder blades — pinch them together\n"
        response += "  • Slight arch in lower back, feet flat on floor\n"
        response += "  • Grip slightly wider than shoulder-width\n\n"
        response += "**The Movement:**\n"
        response += "  1. Unrack with straight arms\n"
        response += "  2. Lower bar slowly to mid-chest\n"
        response += "  3. Keep elbows at 45° — not flared out\n"
        response += "  4. Drive the bar up and slightly back\n"
        response += "  5. Lock out at the top\n\n"
        response += "**Common Mistakes:**\n"
        response += "  ❌ Bouncing the bar off your chest\n"
        response += "  ❌ Elbows flaring at 90°\n"
        response += "  ❌ Lifting your butt off the bench\n\n"
        response += "💡 **Tip:** Think 'push yourself away from the bar' not 'push the bar up'!"

    elif predicted_tag == 'deadlift_form':
        response  = "🏋️ **Proper Deadlift Form**\n\n"
        response += "**Setup:**\n"
        response += "  • Bar over mid-foot, hip-width stance\n"
        response += "  • Grip just outside your legs\n"
        response += "  • Hips higher than knees, chest up\n\n"
        response += "**The Movement:**\n"
        response += "  1. Take a big breath, brace your core hard\n"
        response += "  2. Push the floor away — leg drive first\n"
        response += "  3. Keep the bar dragging up your shins\n"
        response += "  4. Hips and shoulders rise at the same rate\n"
        response += "  5. Lock out by squeezing glutes at the top\n\n"
        response += "**Common Mistakes:**\n"
        response += "  ❌ Rounding your lower back\n"
        response += "  ❌ Bar drifting away from your body\n"
        response += "  ❌ Jerking the bar off the floor\n\n"
        response += "💡 **Tip:** The deadlift is a PUSH not a pull — push the ground away with your legs!"

    elif predicted_tag == 'combined_form':
        response  = "🏋️ **Squat & Bench Press Form Guide**\n\n"
        response += "**Squat:**\n"
        response += "  • Feet shoulder-width, toes slightly out\n"
        response += "  • Keep chest up, core braced\n"
        response += "  • Break parallel for full glute activation\n"
        response += "  • Drive through heels on the way up\n\n"
        response += "**Bench Press:**\n"
        response += "  • Retract shoulder blades into the bench\n"
        response += "  • Grip slightly wider than shoulder-width\n"
        response += "  • Lower bar to mid-chest, elbows at 45°\n"
        response += "  • Drive through chest, not just arms\n\n"
        response += "💡 **Tip:** Film yourself from the side to check your form on both lifts!"

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
            response += f"  • Tuna — 30g per can\n"
            response += f"  • Lentils — 18g per cup\n\n"
            response += f"💡 **Tip:** Spread protein across 4-5 meals for maximum absorption!"

    elif predicted_tag == 'pre_workout_meal':
        response  = "🍌 **Pre-Workout Nutrition**\n\n"
        response += "**Should you eat before training?** YES — it fuels performance!\n\n"
        response += "**Timing:**\n"
        response += "  • Large meal: 2-3 hours before\n"
        response += "  • Small snack: 30-60 mins before\n\n"
        response += "**Best Pre-Workout Foods:**\n"
        response += "  • Banana + peanut butter 🍌\n"
        response += "  • Oats with honey\n"
        response += "  • Rice + chicken (2hrs before)\n"
        response += "  • Greek yogurt + berries\n\n"
        response += "**What to avoid:**\n"
        response += "  • High fat foods (slow digestion)\n"
        response += "  • High fiber foods (bloating)\n"
        response += "  • Eating right before training\n\n"
        response += "💡 **Tip:** Carbs = fuel for your workout. Don't train fasted unless you're used to it!"

    elif predicted_tag == 'sets_reps':
        response  = "📊 **Sets & Reps Guide**\n\n"
        response += "**For Strength (heavy weight):**\n"
        response += "  • 3-5 sets × 3-6 reps · Rest: 3-5 mins\n\n"
        response += "**For Muscle Growth (hypertrophy):**\n"
        response += "  • 3-4 sets × 8-12 reps · Rest: 60-90 secs\n\n"
        response += "**For Endurance (light weight):**\n"
        response += "  • 2-3 sets × 15-20 reps · Rest: 30-60 secs\n\n"
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

    elif predicted_tag == 'workout_frequency':
        response  = "📅 **How Often Should You Work Out?**\n\n"
        response += "**By Goal:**\n"
        response += "  • Weight Loss: 4-5 days/week (mix cardio + weights)\n"
        response += "  • Muscle Gain: 3-5 days/week (resistance training)\n"
        response += "  • General Fitness: 3 days/week is enough to start\n\n"
        response += "**By Experience:**\n"
        response += "  • Beginner: 3 days/week — your body needs recovery time\n"
        response += "  • Intermediate: 4 days/week\n"
        response += "  • Advanced: 5-6 days/week\n\n"
        response += "💡 **Tip:** More is not always better. Recovery is when you actually grow — never skip rest days!"

    elif predicted_tag == 'hydration':
        response  = "💧 **How Much Water Should You Drink?**\n\n"
        response += "**General Rule:** 35ml per kg of bodyweight per day\n\n"
        if user_profile.get('weight'):
            daily = round(user_profile['weight'] * 0.035, 1)
            response += f"**Your target:** ~{daily}L per day based on your weight\n\n"
        response += "**For Athletes — drink more when:**\n"
        response += "  • Training sessions (add 500ml-1L per hour of exercise)\n"
        response += "  • Hot weather\n"
        response += "  • High intensity workouts\n\n"
        response += "**Signs of dehydration:**\n"
        response += "  • Dark yellow urine\n"
        response += "  • Headaches during workout\n"
        response += "  • Muscle cramps\n\n"
        response += "💡 **Tip:** Drink a glass of water first thing in the morning before anything else!"

    elif predicted_tag == 'weight_management':
        response  = "⚖️ **Weight Management Guide**\n\n"
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

    elif predicted_tag == 'workout_tips':
        response  = "💪 **Workout Tips & Advice**\n\n"
        response += "**To Push Through a Tough Workout:**\n"
        response += "  • Break it into smaller chunks — 'just 5 more reps'\n"
        response += "  • Use music to boost energy\n"
        response += "  • Train with a partner for accountability\n"
        response += "  • Remember your WHY\n\n"
        response += "**To Lose Belly Fat:**\n"
        response += "  • You can't spot reduce — full body fat loss is the way\n"
        response += "  • Best exercises: HIIT, compound lifts, cardio\n"
        response += "  • Diet is 80% of the result — calorie deficit is key\n\n"
        response += "**After Cardio:**\n"
        response += "  • Stretch for 5-10 minutes\n"
        response += "  • Rehydrate — drink water immediately\n"
        response += "  • Eat protein within 30-60 mins\n"
        response += "  • Rest or do light activity next day\n\n"
        response += "💡 **Tip:** Consistency beats perfection every single time!" 

    elif predicted_tag == 'fasted_workout':
        response  = "🌅 **Working Out on an Empty Stomach**\n\n"
        response += "**For Fat Loss:** Fasted cardio can burn slightly more fat — best for walking or light jog\n\n"
        response += "**For Strength Training:** NOT recommended — you need fuel to lift heavy\n\n"
        response += "**Best approach:**\n"
        response += "  • Light snack 30-60 mins before (banana, oats, yogurt)\n"
        response += "  • Never do heavy lifting completely fasted\n\n"
        response += "💡 **Tip:** At minimum, have a banana and coffee before morning weight training!"

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
        response += "**Safe to start:**\n"
        response += "  • Walking, pelvic floor exercises, gentle yoga\n\n"
        response += "**Avoid initially:**\n"
        response += "  • Running, jumping, heavy lifting, core exercises\n\n"
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
        response += "💡 **Tip:** The combo of lifting + calorie deficit + high protein is the formula for a slim strong body!"

    elif predicted_tag == 'lower_body_fat':
        response  = "🔥 **How to Lower Body Fat**\n\n"
        response += "**The Truth:** You can't spot reduce fat — you lose it all over\n\n"
        response += "**Best Exercises:**\n"
        response += "  • HIIT — burns the most calories in least time\n"
        response += "  • Compound lifts (squat, deadlift) — burns fat + builds muscle\n"
        response += "  • Steady state cardio — 30-45 min jog\n"
        response += "  • Walking — underrated, great for daily calorie burn\n\n"
        response += "**Diet is 80% of the result:**\n"
        response += "  • Calorie deficit of 300-500 kcal/day\n"
        response += "  • High protein to preserve muscle while losing fat\n\n"
        response += "💡 **Tip:** Ask me for a personalized diet plan — I'll calculate your exact calorie target based on your weight!"

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
        response += "**Drink more when:**\n"
        response += "  • Training (add 500ml-1L per hour of exercise)\n"
        response += "  • Hot weather or sweating heavily\n\n"
        response += "**Signs of dehydration:**\n"
        response += "  • Dark yellow urine\n"
        response += "  • Headaches during workout\n"
        response += "  • Muscle cramps\n\n"
        response += "💡 **Tip:** Drink a glass of water first thing every morning!"  

    elif predicted_tag == 'first_time_gym':
        response  = "🌟 **Tips for Your First Time at the Gym**\n\n"
        response += "**Before You Go:**\n"
        response += "  • Wear comfortable clothes and proper shoes\n"
        response += "  • Bring water, a towel, and headphones\n"
        response += "  • Go during off-peak hours (early morning or midday)\n\n"
        response += "**At the Gym:**\n"
        response += "  • Start with machines — they're safer for beginners\n"
        response += "  • Don't be afraid to ask staff for help\n"
        response += "  • Wipe equipment after use\n"
        response += "  • Rest 60-90 seconds between sets\n\n"
        response += "**Focus on:**\n"
        response += "  • Learning proper form first\n"
        response += "  • Light weights — ego check at the door\n"
        response += "  • Full body workout 3x/week to start\n\n"
        response += "💡 **Tip:** Everyone was a beginner once. Nobody is watching you — they're all focused on themselves!"

    elif predicted_tag == 'weight_loss_tips':
        response  = "⚡ **How to Lose Weight Effectively**\n\n"
        response += "**The Formula:**\n"
        response += "  • Calorie deficit = calories in < calories out\n"
        response += "  • Aim for 300-500 kcal deficit per day\n"
        response += "  • Lose 0.5-1kg per week (sustainable pace)\n\n"
        response += "**Top Tips:**\n"
        response += "  • High protein diet — keeps you full longer\n"
        response += "  • Strength train — muscle burns more calories at rest\n"
        response += "  • Walk 8,000-10,000 steps daily\n"
        response += "  • Cut liquid calories (soda, juice, alcohol)\n"
        response += "  • Sleep 7-9 hours — poor sleep = more hunger\n\n"
        response += "💡 **Tip:** Set up your profile and ask for a diet plan — I'll calculate your exact calories!"

    elif predicted_tag == 'gain_weight':
        response  = "🍗 **How to Gain Weight the Right Way**\n\n"
        response += "**Calorie Surplus:**\n"
        response += "  • Eat 300-500 kcal above your maintenance\n"
        response += "  • Aim for 0.25-0.5kg gained per week\n\n"
        response += "**Best Foods for Weight Gain:**\n"
        response += "  • Rice, oats, bread, pasta (carbs = calories)\n"
        response += "  • Chicken, beef, eggs, fish (protein = muscle)\n"
        response += "  • Peanut butter, avocado, nuts (healthy fats)\n"
        response += "  • Whole milk, Greek yogurt\n\n"
        response += "**Training:**\n"
        response += "  • Focus on compound lifts — squat, bench, deadlift\n"
        response += "  • Progressive overload every week\n\n"
        response += "💡 **Tip:** Eat every 3-4 hours. If you struggle to eat enough, add a peanut butter smoothie to your day!"

    elif predicted_tag == 'abs_workout':
        response  = "🔥 **Abs & Core Workout**\n\n"
        response += "**The Truth About Abs:**\n"
        response += "  • Abs are made in the kitchen — diet is key\n"
        response += "  • You need low body fat to see them (12-15% for men, 18-22% for women)\n\n"
        response += "**Best Ab Exercises:**\n"
        response += "  • Plank — 3x30-60 seconds\n"
        response += "  • Hanging Leg Raises — 3x12\n"
        response += "  • Cable Crunches — 3x15\n"
        response += "  • Ab Wheel Rollout — 3x10\n"
        response += "  • Dead Bug — 3x10 each side\n\n"
        response += "**Core Compound Lifts (best for abs):**\n"
        response += "  • Squat, Deadlift, Overhead Press\n\n"
        response += "💡 **Tip:** Train abs 2-3x/week. More than that won't help — they need recovery too!"

    elif predicted_tag == 'sleep':
        response  = "😴 **Sleep & Muscle Growth**\n\n"
        response += "**How much do you need?**\n"
        response += "  • Minimum: 7 hours\n"
        response += "  • Optimal for athletes: 8-9 hours\n\n"
        response += "**Why sleep matters:**\n"
        response += "  • Growth hormone is released during deep sleep\n"
        response += "  • Muscle repair happens while you rest\n"
        response += "  • Poor sleep increases cortisol (kills gains)\n"
        response += "  • Bad sleep = more hunger and cravings\n\n"
        response += "**Better Sleep Tips:**\n"
        response += "  • Sleep and wake at the same time daily\n"
        response += "  • No screens 30 mins before bed\n"
        response += "  • Keep room cool and dark\n\n"
        response += "💡 **Tip:** Sleep is when you actually GROW. Training breaks muscle down, sleep builds it back up!"

    elif predicted_tag == 'injury':
        response  = "🩹 **Dealing with Workout Injuries**\n\n"
        response += "**First: Use the R.I.C.E Method**\n"
        response += "  • **Rest** — stop the activity immediately\n"
        response += "  • **Ice** — 15-20 mins every 2-3 hours\n"
        response += "  • **Compression** — wrap to reduce swelling\n"
        response += "  • **Elevation** — raise above heart level\n\n"
        response += "**Common Causes:**\n"
        response += "  • Poor form — always prioritize technique\n"
        response += "  • Too much weight too soon\n"
        response += "  • Skipping warm-up\n"
        response += "  • Overtraining without rest\n\n"
        response += "**When to see a doctor:**\n"
        response += "  • Sharp or severe pain\n"
        response += "  • Swelling that doesn't go down\n"
        response += "  • Pain that lasts more than a week\n\n"
        response += "💡 **Tip:** Never train through sharp pain. Train around injuries — work other muscle groups while you heal!"

    elif predicted_tag == 'gym_essentials':
        response  = "🎒 **Gym Bag Essentials**\n\n"
        response += "**Must Haves:**\n"
        response += "  • Water bottle (at least 1L)\n"
        response += "  • Towel (for equipment and sweat)\n"
        response += "  • Proper training shoes\n"
        response += "  • Comfortable workout clothes\n"
        response += "  • Headphones\n\n"
        response += "**Nice to Have:**\n"
        response += "  • Lifting belt (for heavy compound lifts)\n"
        response += "  • Wrist wraps (for bench/OHP)\n"
        response += "  • Knee sleeves (for squats)\n"
        response += "  • Protein shake for post-workout\n"
        response += "  • Resistance bands (for warm-up)\n\n"
        response += "💡 **Tip:** You don't need fancy gear to start. Water, towel, and shoes is all you really need!"

    elif predicted_tag == 'home_workout':
        response  = "🏠 **Home Workout — No Equipment Needed**\n\n"
        response += "**Full Body Home Routine (3x/week):**\n\n"
        response += "**Push:**\n"
        response += "  • Push-ups — 3x max reps\n"
        response += "  • Pike Push-ups (shoulders) — 3x10\n"
        response += "  • Tricep Dips on chair — 3x12\n\n"
        response += "**Pull:**\n"
        response += "  • Doorframe rows (if available) — 3x10\n"
        response += "  • Superman holds — 3x15\n\n"
        response += "**Legs:**\n"
        response += "  • Squats — 3x20\n"
        response += "  • Lunges — 3x15 each leg\n"
        response += "  • Glute bridges — 3x20\n\n"
        response += "**Core:**\n"
        response += "  • Plank — 3x45 seconds\n"
        response += "  • Crunches — 3x20\n\n"
        response += "💡 **Tip:** Add a backpack with books for extra resistance as you get stronger!"

    elif predicted_tag == 'plateau':
        response  = "📊 **Breaking Through a Plateau**\n\n"
        response += "**Why Plateaus Happen:**\n"
        response += "  • Your body has adapted to the same routine\n"
        response += "  • Calories need adjustment\n"
        response += "  • Not enough recovery\n\n"
        response += "**How to Break It:**\n"
        response += "  • Change your exercises — new stimulus = new growth\n"
        response += "  • Increase weight, reps, or sets (progressive overload)\n"
        response += "  • Take a deload week — lighter weights for recovery\n"
        response += "  • Recalculate your calories — they need to change as you lose weight\n"
        response += "  • Check your sleep and stress levels\n\n"
        response += "💡 **Tip:** A plateau means your body got efficient — that's actually a sign of progress. Time to level up!"

    elif predicted_tag == 'breathing':
        response  = "💨 **How to Breathe When Lifting**\n\n"
        response += "**The Golden Rule:**\n"
        response += "  • Exhale on EFFORT (the hard part)\n"
        response += "  • Inhale on the easier part\n\n"
        response += "**Examples:**\n"
        response += "  • Bench Press: Inhale going down, exhale pushing up\n"
        response += "  • Squat: Inhale going down, exhale coming up\n"
        response += "  • Deadlift: Big breath before lifting, exhale at top\n"
        response += "  • Curl: Exhale curling up, inhale lowering\n\n"
        response += "**Valsalva Maneuver (for heavy lifts):**\n"
        response += "  • Take a big breath into your belly\n"
        response += "  • Hold it and brace your core tight\n"
        response += "  • This creates intra-abdominal pressure for spine protection\n\n"
        response += "💡 **Tip:** Never hold your breath for multiple reps — only for max effort single lifts!"

    elif predicted_tag == 'gym_anxiety':
        response  = "😰 **Dealing with Gym Anxiety**\n\n"
        response += "**You're not alone — most beginners feel this way!**\n\n"
        response += "**Tips to Overcome It:**\n"
        response += "  • Go during off-peak hours (early morning, midday)\n"
        response += "  • Have a plan before you go — know your workout\n"
        response += "  • Use headphones — creates your own bubble\n"
        response += "  • Start with machines, not free weights\n"
        response += "  • Bring a friend for support\n\n"
        response += "**Remember:**\n"
        response += "  • Everyone is focused on themselves\n"
        response += "  • Even the biggest guy there was once a beginner\n"
        response += "  • Gym staff are there to help — ask them!\n\n"
        response += "💡 **Tip:** After 2-3 weeks of going regularly, the anxiety disappears completely. Push through those first sessions!"

    elif predicted_tag == 'post_workout_meal':
        response  = "🍗 **What to Eat After a Workout**\n\n"
        response += "**Eat within 30-60 minutes after training**\n\n"
        response += "**What you need:**\n"
        response += "  • Protein — to repair and build muscle\n"
        response += "  • Carbs — to replenish glycogen stores\n\n"
        response += "**Best Post-Workout Meals:**\n"
        response += "  • Chicken + rice + vegetables\n"
        response += "  • Whey protein shake + banana\n"
        response += "  • Eggs + toast\n"
        response += "  • Greek yogurt + granola + berries\n"
        response += "  • Tuna + rice cakes\n\n"
        response += "💡 **Tip:** The post-workout window is real — don't skip this meal. It's when your muscles absorb nutrients best!"

    elif predicted_tag == 'bulking':
        response  = "📈 **How to Bulk Properly**\n\n"
        response += "**Calorie Surplus:**\n"
        response += "  • Eat 300-500 kcal above maintenance\n"
        response += "  • Aim for 0.25-0.5kg gained per week\n"
        response += "  • More than that = mostly fat gain\n\n"
        response += "**Macros:**\n"
        response += "  • Protein: 1g per lb of bodyweight\n"
        response += "  • Carbs: 40-50% of total calories\n"
        response += "  • Fats: 25-35% of total calories\n\n"
        response += "**Training:**\n"
        response += "  • Heavy compound lifts\n"
        response += "  • Progressive overload every session\n"
        response += "  • Limit cardio to 1-2 light sessions/week\n\n"
        response += "💡 **Tip:** Dirty bulking (eating everything) just makes you fat. Clean bulk = slow and steady muscle gain!"

    elif predicted_tag == 'cutting':
        response  = "✂️ **How to Cut Properly**\n\n"
        response += "**Calorie Deficit:**\n"
        response += "  • Eat 300-500 kcal below maintenance\n"
        response += "  • Lose 0.5-1kg per week maximum\n"
        response += "  • Faster = more muscle loss\n\n"
        response += "**Macros:**\n"
        response += "  • Protein: 1-1.2g per lb (protect your muscle!)\n"
        response += "  • Carbs: 30-40% of total calories\n"
        response += "  • Fats: 20-30% of total calories\n\n"
        response += "**Training:**\n"
        response += "  • Keep lifting heavy — don't switch to light weights\n"
        response += "  • Add 2-3 cardio sessions/week\n"
        response += "  • HIIT is most time-efficient for fat burn\n\n"
        response += "💡 **Tip:** The biggest cutting mistake is dropping calories too fast. Slow and steady keeps your muscle!"                             

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

# ─── Run ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("\n🏋️  GymBuddy AI Backend Running")
    print("API: http://localhost:5000")
    print("Health check: http://localhost:5000/health\n")
    app.run(host='0.0.0.0', port=5000, debug=False)