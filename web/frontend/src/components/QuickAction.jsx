import React from 'react';

export default function QuickAction({ title, description, icon, onClick }) {
  return (
    <button onClick={onClick} className="quick-action-card">
      <div className="card-top">
        <span className="card-icon">{icon}</span>
        <span className="card-plus">+</span>
      </div>
      <div className="card-bottom">
        <h4>{title}</h4>
        <p>{description}</p>
      </div>
    </button>
  );
}
