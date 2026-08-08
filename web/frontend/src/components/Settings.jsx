import React, { useState, useEffect } from 'react';

export default function Settings() {
  const [theme, setTheme] = useState(localStorage.getItem('rexa-theme') || 'dark');
  const [compact, setCompact] = useState(false);
  const [animations, setAnimations] = useState(true);

  // Sync theme changes with body tags and persist to localStorage
  useEffect(() => {
    document.body.className = theme === 'dark' ? 'dark-theme' : 'light-theme';
    localStorage.setItem('rexa-theme', theme);
  }, [theme]);

  return (
    <div className="settings-container">
      <div className="workspace-header">
        <h2>Settings</h2>
        <p className="subtitle">Configure your preferences and local model options.</p>
      </div>

      <div className="settings-content">
        {/* Appearance section */}
        <section className="settings-section">
          <h3>Appearance</h3>
          <div className="setting-card">
            <div className="setting-row">
              <div className="setting-label-col">
                <span className="setting-title">Theme</span>
                <span className="setting-desc">Switch between light mode and dark mode.</span>
              </div>
              <div className="setting-control-col">
                <select 
                  value={theme} 
                  onChange={(e) => setTheme(e.target.value)}
                  className="settings-select"
                >
                  <option value="light">Light</option>
                  <option value="dark">Dark</option>
                </select>
              </div>
            </div>
          </div>
        </section>

        {/* Interface section */}
        <section className="settings-section">
          <h3>Interface</h3>
          <div className="setting-card">
            <div className="setting-row">
              <div className="setting-label-col">
                <span className="setting-title">Compact Mode</span>
                <span className="setting-desc">Denser conversation padding and smaller text.</span>
              </div>
              <div className="setting-control-col">
                <input 
                  type="checkbox" 
                  checked={compact} 
                  onChange={(e) => setCompact(e.target.checked)} 
                  className="settings-checkbox"
                />
              </div>
            </div>
            
            <div className="setting-row">
              <div className="setting-label-col">
                <span className="setting-title">Animations</span>
                <span className="setting-desc">Enable smooth transitions and loading fade-ins.</span>
              </div>
              <div className="setting-control-col">
                <input 
                  type="checkbox" 
                  checked={animations} 
                  onChange={(e) => setAnimations(e.target.checked)} 
                  className="settings-checkbox"
                />
              </div>
            </div>
          </div>
        </section>

        {/* Model section */}
        <section className="settings-section">
          <h3>Model Information</h3>
          <div className="setting-card">
            <div className="setting-row">
              <div className="setting-label-col">
                <span className="setting-title">Model Identifier</span>
                <span className="setting-desc">Active SaaS production model.</span>
              </div>
              <div className="setting-control-col">
                <span className="settings-badge">REXA V4</span>
              </div>
            </div>

            <div className="setting-row">
              <div className="setting-label-col">
                <span className="setting-title">Infrastructure</span>
                <span className="setting-desc">Global cloud server network.</span>
              </div>
              <div className="setting-control-col">
                <span className="settings-badge">SaaS API Cloud</span>
              </div>
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}
