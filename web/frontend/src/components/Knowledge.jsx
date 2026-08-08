import React from 'react';

export default function Knowledge() {
  return (
    <div className="knowledge-container">
      <div className="workspace-header">
        <h2>Knowledge</h2>
        <p className="subtitle">Manage knowledge sources for REXA.</p>
      </div>

      <div className="knowledge-content">
        <div className="knowledge-empty-state">
          <div className="empty-icon">
            <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" style={{color: 'var(--text-secondary)'}}>
              <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/>
              <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/>
            </svg>
          </div>
          <h4>No knowledge sources yet</h4>
          <p>Import text files, PDFs, or documentation pages to expand REXA's active context.</p>
          <button className="btn-primary" disabled>
            + Add Knowledge
          </button>
        </div>
      </div>
    </div>
  );
}
