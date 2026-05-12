import { useState } from "react";

const GOALS = [
  { value: "weight loss", label: "Weight Loss", icon: "🔥", desc: "Burn fat, get lean" },
  { value: "muscle gain", label: "Muscle Gain", icon: "💪", desc: "Build size and strength" },
  { value: "strength", label: "Strength", icon: "⚡", desc: "Get stronger, lift heavier" },
  { value: "general fitness", label: "General Fitness", icon: "🏃", desc: "Stay healthy and active" },
];

const EXPERIENCE_LEVELS = [
  { value: "beginner", label: "Beginner", icon: "🌱", desc: "New to working out" },
  { value: "intermediate", label: "Intermediate", icon: "🔶", desc: "Training for 1-2 years" },
  { value: "advanced", label: "Advanced", icon: "🏆", desc: "3+ years of training" },
];

const DAYS = [1, 2, 3, 4, 5, 6, 7];

export default function Onboarding({ onComplete }) {
  const [step, setStep] = useState(0);
  const [form, setForm] = useState({
    name: "", age: "", weight: "", height: "",
    goal: "", experience: "", daysPerWeek: 3,
  });
  const [animating, setAnimating] = useState(false);

  const set = (k, v) => setForm(p => ({ ...p, [k]: v }));

  const nextStep = () => {
    setAnimating(true);
    setTimeout(() => {
      setStep(s => s + 1);
      setAnimating(false);
    }, 300);
  };

  const prevStep = () => {
    setAnimating(true);
    setTimeout(() => {
      setStep(s => s - 1);
      setAnimating(false);
    }, 300);
  };

  const handleComplete = () => {
    const profile = {
      name: form.name,
      age: parseInt(form.age) || null,
      weight: parseFloat(form.weight) || null,
      height: parseFloat(form.height) || null,
      goal: form.goal,
      experience: form.experience,
      daysPerWeek: form.daysPerWeek,
    };
    onComplete(profile);
  };

  const steps = [
    // Step 0 — Welcome
    {
      content: (
        <div className="ob-welcome">
          <div className="ob-bot-avatar">
            <img src="/gymbuddy-avatar.jpg" alt="GymBuddy" />
          </div>
          <h1 className="ob-title">Welcome to <span>GymBuddy</span></h1>
          <p className="ob-subtitle">Your personal AI fitness coach. Let's set up your profile so I can give you the most personalized experience possible.</p>
          <p className="ob-time">⏱️ Takes less than 1 minute</p>
          <button className="ob-btn-primary" onClick={nextStep}>Let's Get Started →</button>
        </div>
      ),
      canSkip: false,
    },
    // Step 1 — Name
    {
      title: "What's your name?",
      subtitle: "I'll use this to personalize your experience.",
      content: (
        <div className="ob-step-content">
          <input
            className="ob-input"
            type="text"
            placeholder="Enter your name..."
            value={form.name}
            onChange={e => set("name", e.target.value)}
            autoFocus
          />
        </div>
      ),
      canContinue: form.name.trim().length > 0,
    },
    // Step 2 — Goal
    {
      title: `Nice to meet you, ${form.name || "there"}! 👋`,
      subtitle: "What's your main fitness goal?",
      content: (
        <div className="ob-grid-2">
          {GOALS.map(g => (
            <button
              key={g.value}
              className={`ob-card ${form.goal === g.value ? "selected" : ""}`}
              onClick={() => set("goal", g.value)}
            >
              <span className="ob-card-icon">{g.icon}</span>
              <span className="ob-card-label">{g.label}</span>
              <span className="ob-card-desc">{g.desc}</span>
            </button>
          ))}
        </div>
      ),
      canContinue: !!form.goal,
    },
    // Step 3 — Experience
    {
      title: "Your experience level?",
      subtitle: "This helps me tailor your workout plan.",
      content: (
        <div className="ob-grid-3">
          {EXPERIENCE_LEVELS.map(e => (
            <button
              key={e.value}
              className={`ob-card ${form.experience === e.value ? "selected" : ""}`}
              onClick={() => set("experience", e.value)}
            >
              <span className="ob-card-icon">{e.icon}</span>
              <span className="ob-card-label">{e.label}</span>
              <span className="ob-card-desc">{e.desc}</span>
            </button>
          ))}
        </div>
      ),
      canContinue: !!form.experience,
    },
    // Step 4 — Body Stats
    {
      title: "Your body stats",
      subtitle: "Used to calculate your BMI, calories, and protein targets.",
      content: (
        <div className="ob-stats-grid">
          <div className="ob-stat-field">
            <label>Age</label>
            <div className="ob-input-group">
              <input className="ob-input" type="number" placeholder="e.g. 21" value={form.age} onChange={e => set("age", e.target.value)} />
              <span className="ob-unit">yrs</span>
            </div>
          </div>
          <div className="ob-stat-field">
            <label>Weight</label>
            <div className="ob-input-group">
              <input className="ob-input" type="number" placeholder="e.g. 70" value={form.weight} onChange={e => set("weight", e.target.value)} />
              <span className="ob-unit">kg</span>
            </div>
          </div>
          <div className="ob-stat-field">
            <label>Height</label>
            <div className="ob-input-group">
              <input className="ob-input" type="number" placeholder="e.g. 175" value={form.height} onChange={e => set("height", e.target.value)} />
              <span className="ob-unit">cm</span>
            </div>
          </div>
        </div>
      ),
      canContinue: !!form.weight && !!form.height,
      canSkip: true,
    },
    // Step 5 — Days per week
    {
      title: "How many days per week?",
      subtitle: "How often can you commit to training?",
      content: (
        <div className="ob-days-grid">
          {DAYS.map(d => (
            <button
              key={d}
              className={`ob-day-btn ${form.daysPerWeek === d ? "selected" : ""}`}
              onClick={() => set("daysPerWeek", d)}
            >
              <span className="ob-day-num">{d}</span>
              <span className="ob-day-label">day{d > 1 ? "s" : ""}</span>
            </button>
          ))}
        </div>
      ),
      canContinue: true,
    },
    // Step 6 — Complete
    {
      content: (
        <div className="ob-welcome">
          <div className="ob-complete-icon">🎉</div>
          <h1 className="ob-title">You're all set, <span>{form.name || "Champ"}</span>!</h1>
          <div className="ob-summary">
            <div className="ob-summary-item"><span>🎯 Goal</span><strong>{form.goal}</strong></div>
            <div className="ob-summary-item"><span>⚡ Level</span><strong>{form.experience}</strong></div>
            {form.weight && <div className="ob-summary-item"><span>⚖️ Weight</span><strong>{form.weight}kg</strong></div>}
            {form.height && <div className="ob-summary-item"><span>📏 Height</span><strong>{form.height}cm</strong></div>}
            <div className="ob-summary-item"><span>📅 Days/week</span><strong>{form.daysPerWeek}</strong></div>
          </div>
          <button className="ob-btn-primary" onClick={handleComplete}>Start Training 💪</button>
        </div>
      ),
    },
  ];

  const currentStep = steps[step];
  const totalSteps = steps.length - 2; // exclude welcome and complete
  const progress = step === 0 ? 0 : step === steps.length - 1 ? 100 : ((step - 1) / totalSteps) * 100;

  return (
    <div className="ob-overlay">
      <div className={`ob-container ${animating ? "ob-exit" : "ob-enter"}`}>

        {/* Progress bar */}
        {step > 0 && step < steps.length - 1 && (
          <div className="ob-progress-wrap">
            <div className="ob-progress-bar" style={{ width: `${progress}%` }} />
          </div>
        )}

        {/* Step counter */}
        {step > 0 && step < steps.length - 1 && (
          <div className="ob-step-counter">Step {step} of {totalSteps}</div>
        )}

        {/* Title */}
        {currentStep.title && (
          <div className="ob-header">
            <h2 className="ob-step-title">{currentStep.title}</h2>
            {currentStep.subtitle && <p className="ob-step-subtitle">{currentStep.subtitle}</p>}
          </div>
        )}

        {/* Content */}
        <div className="ob-body">
          {currentStep.content}
        </div>

        {/* Navigation */}
        {step > 0 && step < steps.length - 1 && (
          <div className="ob-nav">
            <button className="ob-btn-back" onClick={prevStep}>← Back</button>
            <div className="ob-nav-right">
              {currentStep.canSkip && (
                <button className="ob-btn-skip" onClick={nextStep}>Skip</button>
              )}
              <button
                className="ob-btn-primary"
                onClick={nextStep}
                disabled={currentStep.canContinue === false}
              >
                Continue →
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}