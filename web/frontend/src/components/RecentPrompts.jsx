import React from 'react';

export default function RecentPrompts({ conversations, currentChatId, onSelectChat }) {
  const recents = conversations.slice(0, 8);

  return (
    <div className="recent-prompts-panel">
      <h3>RECENT PROMPTS</h3>
      <div className="recent-list">
        {recents.length === 0 ? (
          <div className="no-recent">No recent prompts</div>
        ) : (
          recents.map((chat) => (
            <button
              key={chat.id}
              onClick={() => onSelectChat(chat.id)}
              className={`recent-card ${chat.id === currentChatId ? 'active' : ''}`}
              title={chat.title}
            >
              <div className="recent-card-title">{chat.title}</div>
              <div className="recent-card-date">
                {new Date(chat.updatedAt).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
              </div>
            </button>
          ))
        )}
      </div>
    </div>
  );
}
