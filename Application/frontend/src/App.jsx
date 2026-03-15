import { useState, useEffect } from 'react'
import axios from 'axios'
import './index.css'

function App() {
  const [screen, setScreen] = useState('login') 
  const [studentId, setStudentId] = useState('')
  const [name, setName] = useState('')
  const [preference, setPreference] = useState('')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [isEditing, setIsEditing] = useState(false)

  const genres = ["Lo-Fi", "Jazz", "Classical", "Pop", "Rock", "Electronic", "Reggae", "Blues", "Country", "Metal"]

  useEffect(() => {
    if (screen === 'login' || screen === 'onboarding') {
      document.body.className = '';
    }
  }, [screen])

  const handleLogin = async () => {
    try {
      const cleanId = studentId.toLowerCase().trim();
      const res = await axios.post('http://127.0.0.1:5000/login', { student_id: cleanId })
      setName(res.data.name)
      setPreference(res.data.genre)
      setScreen('dashboard')
    } catch {
      setScreen('onboarding')
    }
  }

  const handleRegister = async () => {
    if (!preference || !name) return alert("Please fill all fields")
    try {
      const cleanId = studentId.toLowerCase().trim();
      await axios.post('http://127.0.0.1:5000/register', { student_id: cleanId, name, genre: preference })
      setScreen('dashboard')
    } catch {
      alert("Registration failed. ID might already exist.")
    }
  }

  const handleUpdateGenre = async (newGenre) => {
    try {
        const cleanId = studentId.toLowerCase().trim();
        await axios.post('http://127.0.0.1:5000/update_preference', { student_id: cleanId, genre: newGenre });
        setPreference(newGenre);
        setIsEditing(false);
    } catch {
        alert("Failed to update preference");
    }
  };

  const handleSyncWatch = async () => {
    setLoading(true);
    setResult(null); 

    // Cinematic 3-second delay to simulate syncing and processing
    await new Promise(resolve => setTimeout(resolve, 3000));
    
    try {
      const res = await axios.post("https://researchg45-production.up.railway.app", { 
        student_id: studentId.toLowerCase().trim() 
      });
      setResult(res.data);
      const stressClass = res.data.stress_level.toLowerCase().replace(" ", "-") + "-bg";
      document.body.className = stressClass; 
    } catch (err) {
      alert("Sync Error: " + (err.response?.data?.error || "Could not connect to watch folder."));
    }
    setLoading(false);
  }

  if (screen === 'login') return (
    <div className="center-screen">
      <div className="modern-card">
        <img src="/vite.svg" alt="App Logo" className="app-logo" /> 
        <h1>StressWatch</h1>
        <input type="text" placeholder="Student ID" value={studentId} onChange={e => setStudentId(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && handleLogin()} className="modern-input" />
        <button onClick={handleLogin} className="modern-btn">Login</button>
      </div>
    </div>
  )

  if (screen === 'onboarding') return (
    <div className="center-screen">
      <div className="modern-card wide">
        <img src="/vite.svg" alt="App Logo" className="app-logo" style={{width: '80px'}}/>
        <h1 style={{marginTop: '0'}}>Setup Profile</h1>
        <input type="text" placeholder="Your Name" value={name} onChange={e => setName(e.target.value)} className="modern-input" />
        <div className="genre-grid">
          {genres.map(g => (
            <div key={g} className={`genre-box ${preference === g ? 'selected' : ''}`} onClick={() => setPreference(g)}>{g}</div>
          ))}
        </div>
        <button onClick={handleRegister} className="modern-btn">Save Preference</button>
      </div>
    </div>
  )

  return (
    <div className="container">
      <div className="dashboard-header">
        <div style={{display: 'flex', alignItems: 'center', gap: '15px'}}>
          <h2>Hi, {name}!</h2>
        </div>
        <div className="genre-badge" onClick={() => setIsEditing(!isEditing)} style={{cursor: 'pointer', userSelect: 'none'}} title="Click to change genre">
          {isEditing ? "✖ Cancel" : `${preference} Mode ⚙️`}
        </div>
      </div>

      {isEditing && (
        <div className="modern-card wide" style={{marginBottom: '20px', animation: 'fadeIn 0.3s'}}>
          <h4 style={{marginTop: '0'}}>Change Music Preference</h4>
          <div className="genre-grid">
            {genres.map(g => (
              <div key={g} className={`genre-box ${preference === g ? 'selected' : ''}`} onClick={() => handleUpdateGenre(g)}>{g}</div>
            ))}
          </div>
        </div>
      )}

      {/* AUTOMATED SYNC CARD */}
      <div className="modern-card full-width">
        <div style={{display: 'flex', justifyContent: 'center', marginBottom: '15px'}}>
           <img src="/vite.svg" alt="App Logo" style={{width: '60px'}} />
        </div>
        <h2 style={{marginTop: 0, marginBottom: '20px'}}>StressWatch</h2>
        
        <button onClick={handleSyncWatch} disabled={loading} className="modern-btn" style={{minHeight: '54px', display: 'flex', justifyContent: 'center', alignItems: 'center'}}>
          {loading ? (
             <div style={{display: 'flex', alignItems: 'center', gap: '10px'}}>
               <div className="heartbeat-loader"><div></div></div> 
               <span style={{marginLeft: '15px'}}>Syncing & Processing Data...</span>
             </div>
          ) : "Sync Current Watch Data"}
        </button>
      </div>

      {result && (
        <div className={`result-card ${result.stress_level.replace(" ", "-").toLowerCase()}`}>
          <div className="result-header">
            <h2 style={{margin: '0 0 10px 0', fontSize: '2.2rem'}}>{result.stress_level}</h2>
            <p style={{margin: 0, fontSize: '1.05rem', opacity: 0.9}}>{result.music_recommendation.message}</p>
          </div>

          {/* BIOLOGICAL STATS DASHBOARD */}
          <div className="stats-grid">
            <div className="stat-box">
              <span className="stat-label">Heart Rate</span>
              <span className="stat-value">{result.stats.bpm} <small>BPM</small></span>
            </div>
            <div className="stat-box">
              <span className="stat-label">Vagal Tone (RMSSD)</span>
              <span className="stat-value">{result.stats.rmssd}</span>
            </div>
            <div className="stat-box">
              <span className="stat-label">Stress Ratio (LF/HF)</span>
              <span className="stat-value">{result.stats.lfhf}</span>
            </div>
          </div>

          <div className="breathing-session">
            <div className={`breath-circle ${result.stress_level.toLowerCase().replace(" ", "-")}`}></div>
            <p style={{margin: 0, opacity: 0.8}}>Synchronize your breath with the circle</p>
          </div>
          <div className="result-body">
            <div className="video-container">
              <iframe src={result.music_recommendation.track.url} title="Music Player" frameBorder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowFullScreen></iframe>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
export default App