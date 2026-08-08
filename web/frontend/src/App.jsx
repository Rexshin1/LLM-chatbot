import React, { useState, useEffect } from 'react';
import Sidebar from './components/Sidebar';
import Header from './components/Header';
import Dashboard from './components/Dashboard';
import ChatPage from './components/ChatPage';
import ModelPlayground from './components/ModelPlayground';
import Knowledge from './components/Knowledge';
import ChatHistory from './components/ChatHistory';
import Settings from './components/Settings';
import ModelInfo from './components/ModelInfo';
import useChat from './hooks/useChat';

export default function App() {
  // Sync default dark theme on mount
  useEffect(() => {
    const savedTheme = localStorage.getItem('rexa-theme') || 'dark';
    document.body.className = savedTheme === 'dark' ? 'dark-theme' : 'light-theme';
  }, []);

  const {
    conversations,
    currentChatId,
    messages,
    loading,
    error,
    status,
    auth,
    selectedModel,
    setSelectedModel,
    startNewChat,
    selectConversation,
    clearAllConversations,
    deleteConversation,
    sendMessage,
    logout
  } = useChat();

  const [page, setPage] = useState('dashboard');
  const [input, setInput] = useState('');
  const [attachedFile, setAttachedFile] = useState(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [modelInfoOpen, setModelInfoOpen] = useState(false);

  const handleSend = () => {
    if ((!input.trim() && !attachedFile) || loading) return;
    sendMessage(input.trim(), attachedFile);
    setInput('');
    setAttachedFile(null);
  };

  const handleSelectSuggestion = (suggestionText) => {
    sendMessage(suggestionText);
  };

  const handleSelectChatFromHistory = (id) => {
    selectConversation(id);
    setPage('chat');
  };

  const handleStartNewChat = () => {
    startNewChat();
    setPage('chat');
  };

  const renderContent = () => {
    switch (page) {
      case 'dashboard':
        return <Dashboard onPageChange={setPage} />;
      case 'chat':
        return (
          <ChatPage
            messages={messages}
            onSelectSuggestion={handleSelectSuggestion}
            loading={loading}
            input={input}
            setInput={setInput}
            onSend={handleSend}
            selectedModel={selectedModel}
            onModelChange={setSelectedModel}
            attachedFile={attachedFile}
            onAttachFile={setAttachedFile}
            onRemoveFile={() => setAttachedFile(null)}
          />
        );
      case 'playground':
        return <ModelPlayground />;
      case 'knowledge':
        return <Knowledge />;
      case 'history':
        return (
          <ChatHistory
            conversations={conversations}
            onSelectChat={handleSelectChatFromHistory}
            onClearChats={clearAllConversations}
            onPageChange={setPage}
          />
        );
      case 'settings':
        return <Settings />;
      default:
        return <Dashboard onPageChange={setPage} />;
    }
  };

  return (
    <div className="app-container">
      <Sidebar
        currentPage={page}
        onPageChange={setPage}
        onNewChat={handleStartNewChat}
        onOpenModelInfo={() => setModelInfoOpen(true)}
        isOpen={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
        conversations={conversations}
        currentChatId={currentChatId}
        onSelectChat={selectConversation}
        onLogout={logout}
        auth={auth}
        onDeleteConversation={deleteConversation}
      />

      <main className="chat-area">
        <Header 
          status={status} 
          onMenuToggle={() => setSidebarOpen(prev => !prev)} 
          auth={auth}
          onLogout={logout}
        />

        {error && (
          <div className="error-banner" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexShrink: 0 }}>
            <span>⚠️ {error}</span>
            {error.includes("Guest limit") && (
              <button 
                onClick={() => window.location.href = "/api/auth/google"}
                className="btn-google-auth"
                style={{
                  margin: 0,
                  padding: '6px 12px',
                  fontSize: '12px',
                  width: 'auto',
                  display: 'inline-flex',
                  backgroundColor: '#FFFFFF',
                  color: '#000000',
                  borderRadius: '4px',
                  fontWeight: '600'
                }}
              >
                Continue with Google
              </button>
            )}
          </div>
        )}

        {/* ChatPage renders OUTSIDE workspace-content: it fills remaining height and owns its scroll */}
        {page === 'chat' ? (
          <ChatPage
            messages={messages}
            onSelectSuggestion={handleSelectSuggestion}
            loading={loading}
            input={input}
            setInput={setInput}
            onSend={handleSend}
            selectedModel={selectedModel}
            onModelChange={setSelectedModel}
            attachedFile={attachedFile}
            onAttachFile={setAttachedFile}
            onRemoveFile={() => setAttachedFile(null)}
          />
        ) : (
          <div className="workspace-content">
            {renderContent()}
          </div>
        )}
      </main>

      <ModelInfo
        isOpen={modelInfoOpen}
        onClose={() => setModelInfoOpen(false)}
      />
    </div>
  );
}
