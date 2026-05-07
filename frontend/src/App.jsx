import { useState, useRef, useEffect } from "react";
import ReactMarkdown from "react-markdown";

const API_URL = "https://gymbuddy-api-1wit.onrender.com";

const QUICK_PROMPTS = [
  { label: "💪 Workout Schedule", message: "Make me a workout schedule" },
  { label: "🥗 Diet Plan", message: "Give me a diet plan" },
  { label: "📊 Check My BMI", message: "Calculate my BMI" },
  { label: "🦵 Leg Day", message: "Leg workout" },
  { label: "🏃 Cardio Tips", message: "Best cardio exercises" },
  { label: "🍗 Protein Intake", message: "What is my protein intake" },
];

const EXPERIENCE_LEVELS = ["Beginner", "Intermediate", "Advanced"];
const GOALS = ["Weight Loss", "Muscle Gain", "Strength", "General Fitness"];

// ─── Follow-up suggestions per topic ─────────────────────────────────────────
const FOLLOW_UPS = {
  workout_schedule: ["What should I eat on training days?", "How long until I see results?", "What are sets and reps?"],
  diet_general: ["How much protein do I need?", "What are healthy snacks?", "Should I take supplements?"],
  bmi: ["Give me a diet plan", "What exercises should I do?", "How do I manage my weight?"],
  protein: ["What should I eat after workout?", "Should I take whey protein?", "Give me a diet plan"],
  leg_workout: ["Proper squat form", "How many sets should I do?", "How to avoid knee pain?"],
  chest_workout: ["Proper bench press form", "How many sets should I do?", "Shoulder workout"],
  back_workout: ["Proper deadlift form", "Pull up tips", "How many sets should I do?"],
  shoulder_workout: ["How many sets should I do?", "Arm workout", "Proper overhead press form"],
  arm_workout: ["How many sets should I do?", "Chest workout", "How much protein do I need?"],
  cardio: ["Should I mix cardio and lifting?", "How many calories do I burn running?", "Best time to do cardio"],
  rest_recovery: ["How much sleep do I need?", "What to eat on rest days?", "Active recovery tips"],
  beginner_advice: ["Make me a workout schedule", "What should I eat?", "How often should I work out?"],
  supplements: ["How much protein do I need?", "Is creatine safe?", "What to eat after workout?"],
  stretching: ["Pre workout warm up", "How to avoid injury?", "Recovery tips"],
  sets_reps: ["What weight should I use?", "How long should I rest?", "Progressive overload explained"],
  progressive_overload: ["Make me a workout schedule", "How long until I see results?", "What are sets and reps?"],
  abs_workout: ["Diet plan for flat stomach", "How to lose belly fat?", "Core workout tips"],
  sleep: ["Recovery tips", "How often should I work out?", "How to reduce soreness?"],
  weight_loss_tips: ["Give me a diet plan", "Best cardio for fat loss", "How long until I see results?"],
  bulking: ["How much protein do I need?", "Best exercises for muscle gain", "Workout schedule for bulking"],
  cutting: ["How much protein do I need?", "Best cardio for fat loss", "Diet plan for cutting"],
};

const DEFAULT_FOLLOW_UPS = [
  "Make me a workout schedule",
  "Give me a diet plan",
  "Calculate my BMI",
];

// ─── Exercise video links ─────────────────────────────────────────────────────
const EXERCISE_VIDEOS = {
  squat: { name: "Squat Tutorial", url: "https://www.youtube.com/watch?v=ultWZbUMPL8" },
  bench: { name: "Bench Press Tutorial", url: "https://www.youtube.com/watch?v=SCVCLChPQFY" },
  deadlift: { name: "Deadlift Tutorial", url: "https://www.youtube.com/watch?v=op9kVnSso6Q" },
  pullup: { name: "Pull-up Tutorial", url: "https://www.youtube.com/watch?v=eGo4IYlbE5g" },
  ohp: { name: "Overhead Press Tutorial", url: "https://www.youtube.com/watch?v=2yjwXTZQDDI" },
  row: { name: "Barbell Row Tutorial", url: "https://www.youtube.com/watch?v=T3N-TO4reLQ" },
  lunge: { name: "Lunge Tutorial", url: "https://www.youtube.com/watch?v=QOVaHwm-Q6U" },
  plank: { name: "Plank Tutorial", url: "https://www.youtube.com/watch?v=ASdvN_XEl_c" },
  curl: { name: "Bicep Curl Tutorial", url: "https://www.youtube.com/watch?v=ykJmrZ5v0Oo" },
  dip: { name: "Tricep Dip Tutorial", url: "https://www.youtube.com/watch?v=wjUmnZH528Y" },
};

function getExerciseVideo(tag) {
  if (tag === 'leg_workout' || tag === 'squat_form') return EXERCISE_VIDEOS.squat;
  if (tag === 'chest_workout' || tag === 'bench_form') return EXERCISE_VIDEOS.bench;
  if (tag === 'back_workout' || tag === 'deadlift_form') return EXERCISE_VIDEOS.deadlift;
  if (tag === 'shoulder_workout') return EXERCISE_VIDEOS.ohp;
  if (tag === 'arm_workout') return EXERCISE_VIDEOS.curl;
  return null;
}

const WELCOME_MESSAGE = {
  role: "assistant",
  content: "Hey! I'm **GymBuddy** 💪 Your personal AI fitness coach.\n\nI can help you with:\n- 🗓️ Custom workout schedules\n- 🏋️ Exercise guides & form tips\n- 🥗 Diet & nutrition advice\n- 📊 BMI calculator\n\nSet up your profile first for personalized plans, or just ask me anything!",
  tag: "greeting",
};

function loadSessions() {
  try { return JSON.parse(localStorage.getItem("gymbuddy_sessions") || "[]"); }
  catch { return []; }
}
function saveSessions(sessions) { localStorage.setItem("gymbuddy_sessions", JSON.stringify(sessions)); }
function loadProfile() {
  try { return JSON.parse(localStorage.getItem("gymbuddy_profile") || "{}"); }
  catch { return {}; }
}
function saveProfile(p) { localStorage.setItem("gymbuddy_profile", JSON.stringify(p)); }
function createNewSession() {
  return { id: Date.now().toString(), title: "New Session", createdAt: new Date().toLocaleDateString(), messages: [WELCOME_MESSAGE] };
}

// ─── Profile Modal ────────────────────────────────────────────────────────────
function ProfileModal({ profile, onSave, onClose }) {
  const [form, setForm] = useState(profile);
  const set = (k, v) => setForm(p => ({ ...p, [k]: v }));
  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal" onClick={e => e.stopPropagation()}>
        <div className="modal-header">
          <h2>Your Profile</h2>
          <button className="modal-close" onClick={onClose}>✕</button>
        </div>
        <div className="modal-body">
          <div className="form-row">
            <label>Name</label>
            <input type="text" placeholder="e.g. Adrian" value={form.name || ""} onChange={e => set("name", e.target.value)} />
          </div>
          <div className="form-row">
            <label>Age</label>
            <input type="number" placeholder="e.g. 21" value={form.age || ""} onChange={e => set("age", parseInt(e.target.value))} />
          </div>
          <div className="form-row">
            <label>Weight (kg)</label>
            <input type="number" placeholder="e.g. 70" value={form.weight || ""} onChange={e => set("weight", parseFloat(e.target.value))} />
          </div>
          <div className="form-row">
            <label>Height (cm)</label>
            <input type="number" placeholder="e.g. 175" value={form.height || ""} onChange={e => set("height", parseFloat(e.target.value))} />
          </div>
          <div className="form-row">
            <label>Experience Level</label>
            <div className="button-group">
              {EXPERIENCE_LEVELS.map(lvl => (
                <button key={lvl} className={`option-btn ${form.experience === lvl.toLowerCase() ? "active" : ""}`} onClick={() => set("experience", lvl.toLowerCase())}>{lvl}</button>
              ))}
            </div>
          </div>
          <div className="form-row">
            <label>Goal</label>
            <div className="button-group wrap">
              {GOALS.map(g => (
                <button key={g} className={`option-btn ${form.goal === g.toLowerCase() ? "active" : ""}`} onClick={() => set("goal", g.toLowerCase())}>{g}</button>
              ))}
            </div>
          </div>
          <div className="form-row">
            <label>Days/week available</label>
            <div className="button-group">
              {[1,2,3,4,5,6,7].map(d => (
                <button key={d} className={`option-btn ${form.daysPerWeek === d ? "active" : ""}`} onClick={() => set("daysPerWeek", d)}>{d}</button>
              ))}
            </div>
          </div>
        </div>
        <div className="modal-footer">
          <button className="btn-save" onClick={() => { onSave(form); onClose(); }}>Save Profile</button>
        </div>
      </div>
    </div>
  );
}

function TypingIndicator() {
  return (
    <div className="message assistant">
      <div className="avatar"><img src="/gymbuddy-avatar.jpg" alt="GymBuddy" /></div>
      <div className="bubble typing"><span /><span /><span /></div>
    </div>
  );
}

// ─── Main App ─────────────────────────────────────────────────────────────────
export default function App() {
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [sessions, setSessions] = useState(() => {
    const saved = loadSessions();
    if (saved.length === 0) { const s = createNewSession(); saveSessions([s]); return [s]; }
    return saved;
  });
  const [activeId, setActiveId] = useState(() => {
    const saved = loadSessions();
    return saved.length > 0 ? saved[0].id : null;
  });
  const [input, setInput] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const [showProfile, setShowProfile] = useState(false);
  const [profile, setProfile] = useState(loadProfile);
  const [deleteConfirm, setDeleteConfirm] = useState(null);
  const [lastTag, setLastTag] = useState(null);
  const bottomRef = useRef(null);

  const activeSession = sessions.find(s => s.id === activeId);
  const messages = activeSession?.messages || [];

  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: "smooth" }); }, [messages, isTyping]);

  const updateSession = (id, updater) => {
    setSessions(prev => { const u = prev.map(s => s.id === id ? updater(s) : s); saveSessions(u); return u; });
  };

  const addMessage = (id, message) => {
    updateSession(id, s => {
      const newMessages = [...s.messages, message];
      const title = s.title === "New Session" && message.role === "user"
        ? message.content.slice(0, 28) + (message.content.length > 28 ? "..." : "")
        : s.title;
      return { ...s, messages: newMessages, title };
    });
  };

  // Build context summary from recent messages
  const buildContext = (messages) => {
    const recent = messages.slice(-6); // last 6 messages
    return recent.map(m => `${m.role}: ${m.content.slice(0, 100)}`).join('\n');
  };

  const sendMessage = async (text) => {
    const msg = text || input.trim();
    if (!msg || !activeId) return;
    addMessage(activeId, { role: "user", content: msg });
    setInput("");
    setIsTyping(true);

    try {
      const context = buildContext(messages);
      const res = await fetch(`${API_URL}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: msg,
          user_profile: profile,
          context: context,
        }),
      });
      const data = await res.json();
      setLastTag(data.tag);

      // Add video link if relevant
      const video = getExerciseVideo(data.tag);
      let finalResponse = data.response;
      if (video) {
        finalResponse += `\n\n🎥 **Watch:** [${video.name}](${video.url})`;
      }

      addMessage(activeId, { role: "assistant", content: finalResponse, tag: data.tag });
    } catch {
      addMessage(activeId, { role: "assistant", content: "⚠️ Couldn't connect to the GymBuddy server. Make sure the backend is running!" });
    } finally {
      setIsTyping(false);
    }
  };

  const handleNewSession = () => {
    const s = createNewSession();
    setSessions(prev => { const u = [s, ...prev]; saveSessions(u); return u; });
    setActiveId(s.id);
    setLastTag(null);
  };

  const handleDelete = (id) => {
    setSessions(prev => {
      const u = prev.filter(s => s.id !== id);
      if (u.length === 0) { const fresh = createNewSession(); saveSessions([fresh]); setActiveId(fresh.id); return [fresh]; }
      saveSessions(u);
      if (activeId === id) setActiveId(u[0].id);
      return u;
    });
    setDeleteConfirm(null);
  };

  const handleSaveProfile = (p) => {
    setProfile(p);
    saveProfile(p);
    const name = p.name ? `${p.name}` : "Athlete";
    addMessage(activeId, {
      role: "assistant",
      content: `Profile saved! 🎯\n\nNice to meet you, **${name}**!\n\n**Goal:** ${p.goal || "Not set"}\n**Experience:** ${p.experience || "Not set"}\n**Days/week:** ${p.daysPerWeek || 3}\n\nNow I can give you fully personalized plans. Try asking for a workout schedule or diet plan!`,
      tag: "profile",
    });
  };

  const followUps = FOLLOW_UPS[lastTag] || DEFAULT_FOLLOW_UPS;
  const isProfileComplete = profile.weight && profile.height && profile.goal;
  const userName = profile.name ? `, ${profile.name}` : "";

  return (
    <div className="app">
      {/* Sidebar */}
      <div className={`sidebar ${sidebarOpen ? "open" : "closed"}`}>
        {sidebarOpen ? (
          <>
            <div className="sidebar-top">
<div className="sidebar-logo">
  <span className="logo-icon">🏋️</span>
  <span className="logo-text">GymBuddy</span>
</div>              <button className="toggle-btn" onClick={() => setSidebarOpen(false)}>‹</button>
            </div>

            <button className="new-chat-btn" onClick={handleNewSession}><span>+</span> New Session</button>

            <div className="sidebar-section">
              <p className="sidebar-label">Quick Actions</p>
              {QUICK_PROMPTS.map(p => (
                <button key={p.label} className="sidebar-item" onClick={() => sendMessage(p.message)}>{p.label}</button>
              ))}
            </div>

            <div className="sidebar-section history-section">
              <p className="sidebar-label">Chat History</p>
              {sessions.map(s => (
                <div key={s.id} className={`history-item ${s.id === activeId ? "active" : ""}`}>
                  <button className="history-btn" onClick={() => setActiveId(s.id)}>
                    <span className="history-title">{s.title}</span>
                    <span className="history-date">{s.createdAt}</span>
                  </button>
                  {deleteConfirm === s.id ? (
                    <div className="delete-confirm">
                      <button className="delete-yes" onClick={() => handleDelete(s.id)}>Del</button>
                      <button className="delete-no" onClick={() => setDeleteConfirm(null)}>✕</button>
                    </div>
                  ) : (
                    <button className="delete-btn" onClick={() => setDeleteConfirm(s.id)}>✕</button>
                  )}
                </div>
              ))}
            </div>

            <div className="sidebar-bottom">
              <button className={`profile-btn ${isProfileComplete ? "complete" : ""}`} onClick={() => setShowProfile(true)}>
                <span className="profile-avatar">{isProfileComplete ? "✓" : "?"}</span>
                <div className="profile-info">
                  <span className="profile-name">{profile.name || (isProfileComplete ? "Profile Set" : "Setup Profile")}</span>
                  <span className="profile-sub">{isProfileComplete ? `${profile.goal} · ${profile.experience}` : "For personalized plans"}</span>
                </div>
              </button>
            </div>
          </>
        ) : (
          <div className="sidebar-icons">
            <button className="icon-btn" onClick={() => setSidebarOpen(true)} title="Expand">›</button>
            <button className="icon-btn" onClick={handleNewSession} title="New Session">✏️</button>
            <div className="icon-divider" />
            {QUICK_PROMPTS.map(p => (
              <button key={p.label} className="icon-btn" onClick={() => sendMessage(p.message)} title={p.label}>
                {p.label.split(" ")[0]}
              </button>
            ))}
            <div className="icon-divider" />
            <button className={`icon-btn ${isProfileComplete ? "icon-complete" : ""}`} onClick={() => setShowProfile(true)} title="Profile">
              {isProfileComplete ? "✅" : "👤"}
            </button>
          </div>
        )}
      </div>

      {/* Main */}
      <div className="main">
        <div className="topbar">
          <div className="topbar-left">
            <span className="topbar-title">GymBuddy AI{userName}</span>
          </div>
          {!isProfileComplete && (
            <button className="topbar-profile-alert" onClick={() => setShowProfile(true)}>
              ⚡ Set up profile for personalized plans
            </button>
          )}
        </div>

        <div className="messages">
          {messages.map((m, i) => (
            <div key={i} className={`message ${m.role}`}>
              {m.role === "assistant" && (
                <div className="avatar"><img src="/gymbuddy-avatar.jpg" alt="GymBuddy" /></div>
              )}
              <div className="bubble">
                <ReactMarkdown components={{
                  a: ({ href, children }) => (
                    <a href={href} target="_blank" rel="noopener noreferrer" className="chat-link">{children}</a>
                  )
                }}>{m.content}</ReactMarkdown>
              </div>
              {m.role === "user" && <div className="avatar user-avatar">U</div>}
            </div>
          ))}
          {isTyping && <TypingIndicator />}
          <div ref={bottomRef} />
        </div>

        {/* Quick prompts on fresh chat */}
        {messages.length === 1 && (
          <div className="quick-prompts">
            {QUICK_PROMPTS.map(p => (
              <button key={p.label} className="quick-btn" onClick={() => sendMessage(p.message)}>{p.label}</button>
            ))}
          </div>
        )}

        {/* Follow-up suggestions */}
        {messages.length > 1 && !isTyping && lastTag && (
          <div className="follow-ups">
            {followUps.map(q => (
              <button key={q} className="follow-up-btn" onClick={() => sendMessage(q)}>{q}</button>
            ))}
          </div>
        )}

        <div className="input-area">
          <div className="input-box">
            <textarea
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendMessage(); } }}
              placeholder="Ask about workouts, diet, exercises..."
              rows={1}
            />
            <button className={`send-btn ${input.trim() ? "active" : ""}`} onClick={() => sendMessage()} disabled={!input.trim()}>↑</button>
          </div>
          <p className="input-hint">Press Enter to send · Shift+Enter for new line</p>
        </div>
      </div>

      {showProfile && <ProfileModal profile={profile} onSave={handleSaveProfile} onClose={() => setShowProfile(false)} />}
    </div>
  );
}