import React from 'react';

export default function Sidebar({ 
  currentPage, 
  onPageChange, 
  onNewChat,
  onOpenModelInfo,
  isOpen,
  onClose,
  conversations = [],
  currentChatId,
  onSelectChat,
  onLogout,
  auth = { authenticated: false, mode: 'guest', user: null, limit: 10, usage: 0 },
  onDeleteConversation
}) {
  const sections = [
    {
      title: "MAIN",
      items: [
        { 
          id: "dashboard", 
          label: "Dashboard", 
          icon: (
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <rect x="3" y="3" width="7" height="7" rx="1"/>
              <rect x="14" y="3" width="7" height="7" rx="1"/>
              <rect x="14" y="14" width="7" height="7" rx="1"/>
              <rect x="3" y="14" width="7" height="7" rx="1"/>
            </svg>
          )
        },
        { 
          id: "chat", 
          label: "AI Chat", 
          icon: (
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
            </svg>
          )
        }
      ]
    },
    {
      title: "WORKSPACE",
      items: [
        { 
          id: "playground", 
          label: "Model Playground", 
          icon: (
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
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
          label: "Knowledge", 
          icon: (
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/>
              <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/>
            </svg>
          )
        },
        { 
          id: "history", 
          label: "Chat History", 
          icon: (
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="12" cy="12" r="10"/>
              <polyline points="12 6 12 12 16 14"/>
            </svg>
          )
        }
      ]
    },
    {
      title: "SYSTEM",
      items: [
        { 
          id: "settings", 
          label: "Settings", 
          icon: (
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="12" cy="12" r="3"/>
              <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/>
            </svg>
          )
        }
      ]
    }
  ];

  return (
    <>
      {/* Mobile drawer backdrop */}
      {isOpen && <div className="sidebar-backdrop" onClick={onClose}></div>}
      
      <aside className={`sidebar ${isOpen ? 'open' : ''}`}>
        <div className="sidebar-top">
          <div className="logo" onClick={() => { onPageChange("dashboard"); onClose(); }} style={{ cursor: "pointer" }}>
            <svg width="28" height="28" viewBox="0 0 100 100" fill="none" xmlns="http://www.w3.org/2000/svg" style={{flexShrink: 0}}>
              <path d="M25 48V75L41 59L25 48Z" fill="#7698F9" />
              <path d="M52 60L77 77H52V60Z" fill="#3F6CD8" />
              <path d="M25 35H55C64.665 35 72.5 42.835 72.5 52.5C72.5 62.165 64.665 70 55 70H50" stroke="#5B86F7" strokeWidth="12" strokeLinecap="round" strokeLinejoin="round" />
              <circle cx="50" cy="70" r="8" fill="#5B86F7" />
            </svg>
            <div className="logo-text">
              <h1>REXA</h1>
              <span className="subtitle">AI Assistant</span>
            </div>
          </div>

          <button onClick={() => { onNewChat(); onClose(); }} className="btn-new-chat">
            <span className="icon">+</span> New chat
          </button>

          <nav className="sidebar-nav">
            {sections.map((section, sIdx) => (
              <div key={sIdx} className="nav-group">
                <h3>{section.title}</h3>
                <div className="nav-items">
                  {section.items.map((item) => (
                    <button
                      key={item.id}
                      onClick={() => { onPageChange(item.id); onClose(); }}
                      className={`nav-item ${currentPage === item.id ? 'active' : ''}`}
                    >
                      <span className="nav-icon">{item.icon}</span>
                      <span className="nav-label">{item.label}</span>
                    </button>
                  ))}
                  
                  {section.title === "SYSTEM" && (
                    <button
                      onClick={() => { onOpenModelInfo(); onClose(); }}
                      className="nav-item"
                    >
                      <span className="nav-icon">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                          <circle cx="12" cy="12" r="10"/>
                          <line x1="12" y1="16" x2="12" y2="12"/>
                          <line x1="12" y1="8" x2="12.01" y2="8"/>
                        </svg>
                      </span>
                      <span className="nav-label">Model Info</span>
                    </button>
                  )}
                </div>
              </div>
            ))}

            <div className="nav-group recent-chats-section" style={{marginTop: '16px'}}>
              <h3>Recent Chats</h3>
              <div className="recent-chats-list">
                {conversations.slice(0, 15).map((chat) => (
                  <div
                    key={chat.id}
                    className={`recent-chat-item-container ${chat.id === currentChatId ? 'active' : ''}`}
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'space-between',
                      width: '100%',
                      borderRadius: '6px',
                      marginBottom: '2px',
                      backgroundColor: chat.id === currentChatId ? 'rgba(91, 92, 235, 0.08)' : 'transparent',
                      transition: 'background-color 0.15s'
                    }}
                  >
                    <button
                      onClick={() => { onSelectChat(chat.id); onPageChange("chat"); onClose(); }}
                      className="recent-chat-click-area"
                      style={{
                        background: 'transparent',
                        border: 'none',
                        display: 'flex',
                        alignItems: 'center',
                        gap: '8px',
                        padding: '6px 10px',
                        flexGrow: 1,
                        color: chat.id === currentChatId ? 'var(--primary)' : 'var(--text-secondary)',
                        textAlign: 'left',
                        cursor: 'pointer',
                        fontSize: '12px',
                        fontWeight: '500',
                        overflow: 'hidden',
                        whiteSpace: 'nowrap',
                        textOverflow: 'ellipsis'
                      }}
                      title={chat.title}
                    >
                      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{flexShrink: 0, opacity: 0.7}}>
                        <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
                      </svg>
                      <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{chat.title}</span>
                    </button>
                    {onDeleteConversation && (
                      <button
                        onClick={(e) => { e.stopPropagation(); onDeleteConversation(chat.id); }}
                        className="btn-delete-conv"
                        title="Delete chat"
                        style={{
                          background: 'transparent',
                          border: 'none',
                          color: 'var(--text-secondary)',
                          cursor: 'pointer',
                          padding: '4px',
                          marginRight: '4px',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          borderRadius: '4px',
                          opacity: 0.6,
                          transition: 'all 0.15s'
                        }}
                      >
                        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                          <polyline points="3 6 5 6 21 6"/>
                          <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>
                        </svg>
                      </button>
                    )}
                  </div>
                ))}
                {conversations.length === 0 && (
                  <span className="no-recent-text">No recent chats</span>
                )}
              </div>
            </div>
          </nav>
        </div>

        <div className="sidebar-bottom">
          {auth.authenticated ? (
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', width: '100%' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px', overflow: 'hidden' }}>
                {auth.user.avatar ? (
                  <img 
                    src={auth.user.avatar} 
                    alt={auth.user.name} 
                    style={{ width: '32px', height: '32px', borderRadius: '50%', flexShrink: 0 }}
                  />
                ) : (
                  <div style={{ width: '32px', height: '32px', borderRadius: '50%', backgroundColor: 'var(--primary)', color: '#FFFFFF', display: 'flex', alignItems: 'center', justifyContent: 'center', fontWeight: 'bold', fontSize: '14px', flexShrink: 0 }}>
                    {auth.user.name ? auth.user.name[0].toUpperCase() : 'U'}
                  </div>
                )}
                <div style={{ display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
                  <span style={{ fontSize: '13px', fontWeight: '600', color: 'var(--text-main)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                    {auth.user.name}
                  </span>
                  <span style={{ fontSize: '11px', color: 'var(--text-secondary)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                    {auth.user.email}
                  </span>
                </div>
              </div>
              <button 
                onClick={onLogout} 
                className="btn-logout" 
                title="Log out"
                style={{
                  background: 'transparent',
                  border: 'none',
                  color: 'var(--text-secondary)',
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  padding: '8px',
                  borderRadius: '6px',
                  transition: 'all 0.15s'
                }}
              >
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"/>
                  <polyline points="16 17 21 12 16 7"/>
                  <line x1="21" y1="12" x2="9" y2="12"/>
                </svg>
              </button>
            </div>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '2px' }}>
              <span style={{ fontSize: '11px', fontWeight: '600', color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Guest Mode</span>
              <span style={{ fontSize: '11px', color: 'var(--text-secondary)' }}>
                {auth.usage ?? 0} / {auth.limit ?? 10} messages used
              </span>
            </div>
          )}
        </div>
      </aside>
    </>
  );
}
