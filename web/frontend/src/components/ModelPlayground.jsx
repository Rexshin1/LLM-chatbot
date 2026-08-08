import React, { useState } from 'react';

export default function ModelPlayground() {
  const [temp, setTemp] = useState(0.7);
  const [topK, setTopK] = useState(10);
  const [maxTokens, setMaxTokens] = useState(50);
  
  return (
    <div className="playground-container">
      <div className="workspace-header">
        <h2>Model Playground</h2>
        <p className="subtitle">Experiment with generation parameters for REXA V4.</p>
      </div>

      <div className="playground-content">
        {/* Parameters Controls */}
        <div className="playground-sidebar">
          <h3>Parameters</h3>
          
          <div className="control-group">
            <div className="control-header">
              <label>Temperature</label>
              <span className="control-value">{temp}</span>
            </div>
            <input 
              type="range" 
              min="0.1" 
              max="1.5" 
              step="0.1" 
              value={temp} 
              onChange={(e) => setTemp(parseFloat(e.target.value))} 
            />
            <p className="control-desc">Controls randomness: lower is more deterministic, higher is more creative.</p>
          </div>

          <div className="control-group">
            <div className="control-header">
              <label>Top-K</label>
              <span className="control-value">{topK}</span>
            </div>
            <input 
              type="range" 
              min="0" 
              max="50" 
              step="1" 
              value={topK} 
              onChange={(e) => setTopK(parseInt(e.target.value))} 
            />
            <p className="control-desc">Limits token pool: only samples from the top K most probable tokens (0 to disable).</p>
          </div>

          <div className="control-group">
            <div className="control-header">
              <label>Max Tokens</label>
              <span className="control-value">{maxTokens}</span>
            </div>
            <input 
              type="range" 
              min="10" 
              max="200" 
              step="10" 
              value={maxTokens} 
              onChange={(e) => setMaxTokens(parseInt(e.target.value))} 
            />
            <p className="control-desc">Maximum number of tokens to generate in a single response.</p>
          </div>
        </div>

        {/* Info panel */}
        <div className="playground-main">
          <div className="playground-card">
            <h4>Active Configuration</h4>
            <div className="info-grid">
              <div className="info-cell">
                <span className="info-cell-label">Model</span>
                <span className="info-cell-value">REXA V4</span>
              </div>
              <div className="info-cell">
                <span className="info-cell-label">Architecture</span>
                <span className="info-cell-value">Transformer-based LLM</span>
              </div>
              <div className="info-cell">
                <span className="info-cell-label">Context Window</span>
                <span className="info-cell-value">4,096 tokens</span>
              </div>
              <div className="info-cell">
                <span className="info-cell-label">Service Type</span>
                <span className="info-cell-value">Text Generation</span>
              </div>
              <div className="info-cell">
                <span className="info-cell-label">Infrastructure</span>
                <span className="info-cell-value">Global Anycast Cloud</span>
              </div>
              <div className="info-cell">
                <span className="info-cell-label">Region</span>
                <span className="info-cell-value">Global</span>
              </div>
            </div>
          </div>
          
          <div className="playground-card info-status-card">
            <h4>Model Status</h4>
            <div className="status-flex">
              <span className="status-dot-green"></span>
              <span className="status-label">REXA V4 is online and ready for global execution.</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
