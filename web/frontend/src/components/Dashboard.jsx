import React from 'react';
import QuickAction from './QuickAction';

export default function Dashboard({ onPageChange }) {
  const actions = [
    { 
      id: "chat", 
      title: "AI Chat", 
      description: "Chat with the model in a clean conversation space.", 
      icon: (
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
        </svg>
      ) 
    },
    { 
      id: "playground", 
      title: "Model Playground", 
      description: "Experiment with temperature, top-k, and sequence length.", 
      icon: (
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <line x1="4" y1="21" x2="4" y2="14"/>
          <line x1="4" y1="10" x2="4" y2="3"/>
          <line x1="12" y1="21" x2="12" y2="12"/>
          <line x1="12" y1="8" x2="12" y2="3"/>
          <line x1="20" y1="21" x2="20" y2="16"/>
          <line x1="20" y1="12" x2="20" y2="3"/>
          <line x1="1" y1="14" x2="7" y2="14"/>
          <line x1="9" y1="8" x2="15" y2="8"/>
          <line x1="17" y1="16" x2="23" y2="16"/>
        </svg>
      ) 
    },
    { 
      id: "knowledge", 
      title: "Knowledge Base", 
      description: "Manage local data sources and reference documents.", 
      icon: (
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/>
          <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/>
        </svg>
      ) 
    },
    { 
      id: "history", 
      title: "Recent Prompts", 
      description: "View past chat history and conversations.", 
      icon: (
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <circle cx="12" cy="12" r="10"/>
          <polyline points="12 6 12 12 16 14"/>
        </svg>
      ) 
    }
  ];

  return (
    <div className="dashboard-container">
      <div className="dashboard-hero">
        <h1 className="hero-title">Welcome to REXA</h1>
        <p className="hero-tagline">Next-generation AI workspace.</p>
        <p className="hero-subtitle">Ask questions, write code, analyze data, and accelerate your productivity.</p>
      </div>

      <div className="quick-actions-section">
        <h3 className="section-title">Quick Actions</h3>
        <div className="quick-actions-grid">
          {actions.map((act) => (
            <QuickAction
              key={act.id}
              title={act.title}
              description={act.description}
              icon={act.icon}
              onClick={() => onPageChange(act.id)}
            />
          ))}
        </div>
      </div>
    </div>
  );
}
