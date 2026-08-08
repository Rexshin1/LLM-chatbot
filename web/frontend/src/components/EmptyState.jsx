import React from 'react';

export default function EmptyState({ onSelectSuggestion }) {
  const suggestions = [
    "Create a simple landing page using HTML/CSS",
    "Write a Python function to reverse a string",
    "Explain the basic concepts of Object-Oriented Programming (OOP)",
    "How does the Transformer architecture work?"
  ];

  return (
    <div className="empty-state">
      <div className="empty-header" style={{display: 'flex', flexDirection: 'column', alignItems: 'center'}}>
        <svg width="60" height="60" viewBox="0 0 100 100" fill="none" xmlns="http://www.w3.org/2000/svg" style={{marginBottom: '16px'}}>
          <path d="M25 48V75L41 59L25 48Z" fill="#7698F9" />
          <path d="M52 60L77 77H52V60Z" fill="#3F6CD8" />
          <path d="M25 35H55C64.665 35 72.5 42.835 72.5 52.5C72.5 62.165 64.665 70 55 70H50" stroke="#5B86F7" strokeWidth="12" strokeLinecap="round" strokeLinejoin="round" />
          <circle cx="50" cy="70" r="8" fill="#5B86F7" />
        </svg>
        <h1 className="empty-brand">REXA</h1>
        <p className="empty-prompt">What can I help you with?</p>
      </div>
      
      <div className="empty-suggestions-container">
        <div className="suggestions">
          {suggestions.map((text, idx) => (
            <button 
              key={idx} 
              onClick={() => onSelectSuggestion(text)} 
              className="suggestion-chip"
            >
              {text}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
