import React from 'react';

export default function ChatHistory({ conversations, onSelectChat, onClearChats, onPageChange }) {
  const getGroupedChats = () => {
    const today = [];
    const yesterday = [];
    const previous7Days = [];
    const older = [];
    
    const now = new Date();
    const oneDayMs = 24 * 60 * 60 * 1000;
    
    conversations.forEach((chat) => {
      const chatDate = new Date(chat.updatedAt);
      const diffMs = now - chatDate;
      
      if (diffMs < oneDayMs) {
        today.push(chat);
      } else if (diffMs < 2 * oneDayMs) {
        yesterday.push(chat);
      } else if (diffMs < 7 * oneDayMs) {
        previous7Days.push(chat);
      } else {
        older.push(chat);
      }
    });
    
    return [
      { label: "Today", items: today },
      { label: "Yesterday", items: yesterday },
      { label: "Previous 7 days", items: previous7Days },
      { label: "Older", items: older }
    ].filter(group => group.items.length > 0);
  };
  
  const grouped = getGroupedChats();

  return (
    <div className="chat-history-page">
      <div className="workspace-header history-header">
        <div>
          <h2>Chat History</h2>
          <p className="subtitle">Manage and review your past chat conversations with REXA.</p>
        </div>
        {conversations.length > 0 && (
          <button 
            onClick={() => { if(confirm("Clear all conversations?")) onClearChats(); }} 
            className="btn-clear-history"
            style={{display: 'inline-flex', alignItems: 'center', gap: '6px'}}
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <polyline points="3 6 5 6 21 6"/>
              <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>
            </svg>
            Clear all conversations
          </button>
        )}
      </div>

      <div className="history-content">
        {conversations.length === 0 ? (
          <div className="history-empty-state">
            <div className="empty-icon">
              <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" style={{color: 'var(--text-secondary)'}}>
                <circle cx="12" cy="12" r="10"/>
                <polyline points="12 6 12 12 16 14"/>
              </svg>
            </div>
            <h4>No conversations found</h4>
            <p>Your local chat logs will appear here once you start messaging REXA.</p>
            <button onClick={() => onPageChange("chat")} className="btn-primary">
              Start new chat
            </button>
          </div>
        ) : (
          <div className="history-groups">
            {grouped.map((group, gIdx) => (
              <div key={gIdx} className="history-group">
                <h4>{group.label}</h4>
                <div className="history-items-grid">
                  {group.items.map((chat) => (
                    <div key={chat.id} className="history-item-row">
                      <button 
                        onClick={() => { onSelectChat(chat.id); onPageChange("chat"); }}
                        className="btn-history-item"
                      >
                        <span className="chat-icon">
                          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{color: 'var(--text-secondary)'}}>
                            <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
                          </svg>
                        </span>
                        <div className="item-details">
                          <span className="item-title">{chat.title}</span>
                          <span className="item-meta">{chat.messages.length} messages • Updated {new Date(chat.updatedAt).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}</span>
                        </div>
                      </button>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
