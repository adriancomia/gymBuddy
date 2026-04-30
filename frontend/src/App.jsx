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

function loadSessions() {
  try { return JSON.parse(localStorage.getItem("gymbuddy_sessions") || "[]"); }
  catch { return []; }
}
function saveSessions(sessions) {
  localStorage.setItem("gymbuddy_sessions", JSON.stringify(sessions));
}
function loadProfile() {
  try { return JSON.parse(localStorage.getItem("gymbuddy_profile") || "{}"); }
  catch { return {}; }
}
function saveProfile(p) {
  localStorage.setItem("gymbuddy_profile", JSON.stringify(p));
}
function createNewSession() {
  return { id: Date.now().toString(), title: "New Session", createdAt: new Date().toLocaleDateString(), messages: [WELCOME_MESSAGE] };
}

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
          <div className="form-row"><label>Age</label><input type="number" placeholder="e.g. 21" value={form.age || ""} onChange={e => set("age", parseInt(e.target.value))} /></div>
          <div className="form-row"><label>Weight (kg)</label><input type="number" placeholder="e.g. 70" value={form.weight || ""} onChange={e => set("weight", parseFloat(e.target.value))} /></div>
          <div className="form-row"><label>Height (cm)</label><input type="number" placeholder="e.g. 175" value={form.height || ""} onChange={e => set("height", parseFloat(e.target.value))} /></div>
          <div className="form-row">
            <label>Experience Level</label>
            <div className="button-group">
              {EXPERIENCE_LEVELS.map(lvl => <button key={lvl} className={`option-btn ${form.experience === lvl.toLowerCase() ? "active" : ""}`} onClick={() => set("experience", lvl.toLowerCase())}>{lvl}</button>)}
            </div>
          </div>
          <div className="form-row">
            <label>Goal</label>
            <div className="button-group wrap">
              {GOALS.map(g => <button key={g} className={`option-btn ${form.goal === g.toLowerCase() ? "active" : ""}`} onClick={() => set("goal", g.toLowerCase())}>{g}</button>)}
            </div>
          </div>
          <div className="form-row">
            <label>Days/week available</label>
            <div className="button-group">
{[1,2,3,4,5,6,7].map(d => <button key={d} className={`option-btn ${form.daysPerWeek === d ? "active" : ""}`} onClick={() => set("daysPerWeek", d)}>{d}</button>)}            </div>
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
      <div className="avatar">G</div>
      <div className="bubble typing"><span /><span /><span /></div>
    </div>
  );
}

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

  const sendMessage = async (text) => {
    const msg = text || input.trim();
    if (!msg || !activeId) return;
    addMessage(activeId, { role: "user", content: msg });
    setInput("");
    setIsTyping(true);
    try {
      const res = await fetch(`${API_URL}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: msg, user_profile: profile }),
      });
      const data = await res.json();
      addMessage(activeId, { role: "assistant", content: data.response });
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
    addMessage(activeId, {
      role: "assistant",
      content: `Profile saved! 🎯\n\n**Goal:** ${p.goal || "Not set"}\n**Experience:** ${p.experience || "Not set"}\n**Days/week:** ${p.daysPerWeek || 3}\n\nNow I can give you fully personalized plans!`,
    });
  };

  const isProfileComplete = profile.weight && profile.height && profile.goal;

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
              </div>
              <button className="toggle-btn" onClick={() => setSidebarOpen(false)} title="Collapse">‹</button>
            </div>

            <button className="new-chat-btn" onClick={handleNewSession}><span>+</span> New Session</button>

            {/* Quick Actions */}
            <div className="sidebar-section">
              <p className="sidebar-label">Quick Actions</p>
              {QUICK_PROMPTS.map(p => (
                <button key={p.label} className="sidebar-item" onClick={() => sendMessage(p.message)}>{p.label}</button>
              ))}
            </div>

            {/* Chat History */}
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
                  <span className="profile-name">{isProfileComplete ? "Profile Set" : "Setup Profile"}</span>
                  <span className="profile-sub">{isProfileComplete ? `${profile.goal} · ${profile.experience}` : "For personalized plans"}</span>
                </div>
              </button>
            </div>
          </>
        ) : (
          /* Collapsed sidebar - icons only */
          <div className="sidebar-icons">
            <button className="icon-btn" onClick={() => setSidebarOpen(true)} title="Expand">›</button>
            <button className="icon-btn" onClick={handleNewSession} title="New Session">✏️</button>
            <button className="icon-btn" onClick={() => { setSidebarOpen(true); }} title="History">🕘</button>
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
            <span className="topbar-title">{activeSession?.title || "GymBuddy AI"}</span>
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
            {QUICK_PROMPTS.map(p => (
              <button key={p.label} className="quick-btn" onClick={() => sendMessage(p.message)}>{p.label}</button>
            ))}
          </div>
        )}

        <div className="input-area">
          <div className="input-box">
            <textarea value={input} onChange={e => setInput(e.target.value)}
              onKeyDown={e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendMessage(); } }}
              placeholder="Ask about workouts, diet, exercises..." rows={1} />
            <button className={`send-btn ${input.trim() ? "active" : ""}`} onClick={() => sendMessage()} disabled={!input.trim()}>↑</button>
          </div>
          <p className="input-hint">Press Enter to send · Shift+Enter for new line</p>
        </div>
      </div>

      {showProfile && <ProfileModal profile={profile} onSave={handleSaveProfile} onClose={() => setShowProfile(false)} />}
    </div>
  );
}