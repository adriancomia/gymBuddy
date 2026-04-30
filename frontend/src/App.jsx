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

const WELCOME_MESSAGE = {
  role: "assistant",
  content:
    "Hey! I'm **GymBuddy** 💪 Your personal AI fitness coach.\n\nI can help you with:\n- 🗓️ Custom workout schedules\n- 🏋️ Exercise guides & form tips\n- 🥗 Diet & nutrition advice\n- 📊 BMI calculator\n\nSet up your profile first for personalized plans, or just ask me anything!",
};

// ─── Storage Helpers ─────────────────────────────────────────────────────────
function loadSessions() {
  try {
    return JSON.parse(localStorage.getItem("gymbuddy_sessions") || "[]");
  } catch {
    return [];
  }
}

function saveSessions(sessions) {
  localStorage.setItem("gymbuddy_sessions", JSON.stringify(sessions));
}

function loadProfile() {
  try {
    return JSON.parse(localStorage.getItem("gymbuddy_profile") || "{}");
  } catch {
    return {};
  }
}

function saveProfile(profile) {
  localStorage.setItem("gymbuddy_profile", JSON.stringify(profile));
}

function createNewSession() {
  return {
    id: Date.now().toString(),
    title: "New Session",
    createdAt: new Date().toLocaleDateString(),
    messages: [WELCOME_MESSAGE],
  };
}

// ─── Profile Modal ───────────────────────────────────────────────────────────
function ProfileModal({ profile, onSave, onClose }) {
  const [form, setForm] = useState(profile);
  const handleChange = (key, value) => setForm((p) => ({ ...p, [key]: value }));
  const handleSave = () => { onSave(form); onClose(); };

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h2>Your Profile</h2>
          <button className="modal-close" onClick={onClose}>✕</button>
        </div>
        <div className="modal-body">
          <div className="form-row">
            <label>Age</label>
            <input type="number" placeholder="e.g. 21" value={form.age || ""} onChange={(e) => handleChange("age", parseInt(e.target.value))} />
          </div>
          <div className="form-row">
            <label>Weight (kg)</label>
            <input type="number" placeholder="e.g. 70" value={form.weight || ""} onChange={(e) => handleChange("weight", parseFloat(e.target.value))} />
          </div>
          <div className="form-row">
            <label>Height (cm)</label>
            <input type="number" placeholder="e.g. 175" value={form.height || ""} onChange={(e) => handleChange("height", parseFloat(e.target.value))} />
          </div>
          <div className="form-row">
            <label>Experience Level</label>
            <div className="button-group">
              {EXPERIENCE_LEVELS.map((lvl) => (
                <button key={lvl} className={`option-btn ${form.experience === lvl.toLowerCase() ? "active" : ""}`} onClick={() => handleChange("experience", lvl.toLowerCase())}>{lvl}</button>
              ))}
            </div>
          </div>
          <div className="form-row">
            <label>Goal</label>
            <div className="button-group wrap">
              {GOALS.map((g) => (
                <button key={g} className={`option-btn ${form.goal === g.toLowerCase() ? "active" : ""}`} onClick={() => handleChange("goal", g.toLowerCase())}>{g}</button>
              ))}
            </div>
          </div>
          <div className="form-row">
            <label>Days/week available</label>
            <div className="button-group">
              {[3, 4, 5, 6].map((d) => (
                <button key={d} className={`option-btn ${form.daysPerWeek === d ? "active" : ""}`} onClick={() => handleChange("daysPerWeek", d)}>{d}</button>
              ))}
            </div>
          </div>
        </div>
        <div className="modal-footer">
          <button className="btn-save" onClick={handleSave}>Save Profile</button>
        </div>
      </div>
    </div>
  );
}

function TypingIndicator() {
  return (
    <div className="message assistant">
      <div className="avatar">G</div>
      <div className="bubble typing"><span></span><span></span><span></span></div>
    </div>
  );
}

// ─── Main App ────────────────────────────────────────────────────────────────
export default function App() {
  const [sessions, setSessions] = useState(() => {
    const saved = loadSessions();
    if (saved.length === 0) {
      const first = createNewSession();
      saveSessions([first]);
      return [first];
    }
    return saved;
  });
  const [activeSessionId, setActiveSessionId] = useState(() => {
    const saved = loadSessions();
    return saved.length > 0 ? saved[0].id : null;
  });
  const [isTyping, setIsTyping] = useState(false);
  const [input, setInput] = useState("");
  const [showProfile, setShowProfile] = useState(false);
  const [profile, setProfile] = useState(loadProfile);
  const [deleteConfirm, setDeleteConfirm] = useState(null);
  const bottomRef = useRef(null);

  const activeSession = sessions.find((s) => s.id === activeSessionId);
  const messages = activeSession?.messages || [];

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isTyping]);

  const updateSession = (sessionId, updater) => {
    setSessions((prev) => {
      const updated = prev.map((s) => s.id === sessionId ? updater(s) : s);
      saveSessions(updated);
      return updated;
    });
  };

  const addMessage = (sessionId, message) => {
    updateSession(sessionId, (s) => {
      const newMessages = [...s.messages, message];
      // Auto-title from first user message
      const title = s.title === "New Session" && message.role === "user"
        ? message.content.slice(0, 30) + (message.content.length > 30 ? "..." : "")
        : s.title;
      return { ...s, messages: newMessages, title };
    });
  };

  const sendMessage = async (text) => {
    const msg = text || input.trim();
    if (!msg || !activeSessionId) return;

    const userMessage = { role: "user", content: msg };
    addMessage(activeSessionId, userMessage);
    setInput("");
    setIsTyping(true);

    try {
      const res = await fetch(`${API_URL}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: msg, user_profile: profile }),
      });
      const data = await res.json();
      addMessage(activeSessionId, { role: "assistant", content: data.response });
    } catch {
      addMessage(activeSessionId, {
        role: "assistant",
        content: "⚠️ Couldn't connect to the GymBuddy server. Make sure the backend is running!",
      });
    } finally {
      setIsTyping(false);
    }
  };

  const handleNewSession = () => {
    const session = createNewSession();
    setSessions((prev) => {
      const updated = [session, ...prev];
      saveSessions(updated);
      return updated;
    });
    setActiveSessionId(session.id);
  };

  const handleDeleteSession = (sessionId) => {
    setSessions((prev) => {
      const updated = prev.filter((s) => s.id !== sessionId);
      if (updated.length === 0) {
        const fresh = createNewSession();
        saveSessions([fresh]);
        setActiveSessionId(fresh.id);
        return [fresh];
      }
      saveSessions(updated);
      if (activeSessionId === sessionId) {
        setActiveSessionId(updated[0].id);
      }
      return updated;
    });
    setDeleteConfirm(null);
  };

  const handleSaveProfile = (p) => {
    setProfile(p);
    saveProfile(p);
    addMessage(activeSessionId, {
      role: "assistant",
      content: `Profile saved! 🎯\n\n**Goal:** ${p.goal || "Not set"}\n**Experience:** ${p.experience || "Not set"}\n**Days/week:** ${p.daysPerWeek || 3}\n\nNow I can give you fully personalized plans. Try asking for a workout schedule or diet plan!`,
    });
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendMessage(); }
  };

  const isProfileComplete = profile.weight && profile.height && profile.goal;

  return (
    <div className="app">
      {/* Sidebar */}
      <div className="sidebar">
        <div className="sidebar-logo">
          <span className="logo-icon">🏋️</span>
          <span className="logo-text">GymBuddy</span>
        </div>

        <button className="new-chat-btn" onClick={handleNewSession}>
          <span>+</span> New Session
        </button>

        <div className="sidebar-section">
          <p className="sidebar-label">Chat History</p>
          {sessions.map((s) => (
            <div key={s.id} className={`history-item ${s.id === activeSessionId ? "active" : ""}`}>
              <button className="history-btn" onClick={() => setActiveSessionId(s.id)}>
                <span className="history-title">{s.title}</span>
                <span className="history-date">{s.createdAt}</span>
              </button>
              {deleteConfirm === s.id ? (
                <div className="delete-confirm">
                  <button className="delete-yes" onClick={() => handleDeleteSession(s.id)}>Delete</button>
                  <button className="delete-no" onClick={() => setDeleteConfirm(null)}>Cancel</button>
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
              <span className="profile-name">{isProfileComplete ? "Profile Set" : "Setup Profile"}</span>
              <span className="profile-sub">{isProfileComplete ? `${profile.goal} · ${profile.experience}` : "For personalized plans"}</span>
            </div>
          </button>
        </div>
      </div>

      {/* Main */}
      <div className="main">
        <div className="topbar">
          <span className="topbar-title">{activeSession?.title || "GymBuddy AI"}</span>
          {!isProfileComplete && (
            <button className="topbar-profile-alert" onClick={() => setShowProfile(true)}>
              ⚡ Set up profile for personalized plans
            </button>
          )}
        </div>

        <div className="messages">
          {messages.map((m, i) => (
            <div key={i} className={`message ${m.role}`}>
              {m.role === "assistant" && <div className="avatar">G</div>}
              <div className="bubble"><ReactMarkdown>{m.content}</ReactMarkdown></div>
              {m.role === "user" && <div className="avatar user-avatar">U</div>}
            </div>
          ))}
          {isTyping && <TypingIndicator />}
          <div ref={bottomRef} />
        </div>

        {messages.length === 1 && (
          <div className="quick-prompts">
            {QUICK_PROMPTS.map((p) => (
              <button key={p.label} className="quick-btn" onClick={() => sendMessage(p.message)}>{p.label}</button>
            ))}
          </div>
        )}

        <div className="input-area">
          <div className="input-box">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
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