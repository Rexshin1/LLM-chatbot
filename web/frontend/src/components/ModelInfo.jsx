import React from 'react';

export default function ModelInfo({ isOpen, onClose }) {
  if (!isOpen) return null;

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        <header className="modal-header">
          <h3>REXA Model Information</h3>
          <button onClick={onClose} className="btn-close" aria-label="Close modal">×</button>
        </header>
        <div className="modal-body">
          <div className="info-row">
            <span className="info-label">Model Name</span>
            <span className="info-value">REXA V4</span>
          </div>
          <div className="info-row">
            <span className="info-label">Architecture</span>
            <span className="info-value">Transformer-based LLM</span>
          </div>
          <div className="info-row">
            <span className="info-label">Capabilities</span>
            <span className="info-value">Text generation, Coding assistant</span>
          </div>
          <div className="info-row">
            <span className="info-label">Context Window</span>
            <span className="info-value">4,096 tokens</span>
          </div>
          <div className="info-row">
            <span className="info-label">Status</span>
            <span className="info-value status-ready">Online</span>
          </div>
        </div>
      </div>
    </div>
  );
}
