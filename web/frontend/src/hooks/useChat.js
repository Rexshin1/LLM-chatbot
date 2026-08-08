import { useState, useEffect, useCallback } from 'react';
import { 
  sendChatMessage, 
  checkBackendHealth, 
  fetchAuthMe, 
  logoutAuth, 
  fetchConversations, 
  fetchMessages,
  deleteConversationApi
} from '../services/api';

export default function useChat() {
  const [conversations, setConversations] = useState([]);
  const [currentChatId, setCurrentChatId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [selectedModel, setSelectedModel] = useState('gemini');
  const [status, setStatus] = useState({ 
    type: 'loading', 
    text: 'Connecting to REXA backend...',
    localModel: false,
    geminiConfigured: false
  });
  const [auth, setAuth] = useState({
    authenticated: false,
    mode: 'guest',
    user: null,
    limit: 10,
    usage: 0
  });

  const loadConversations = useCallback(async () => {
    try {
      const list = await fetchConversations();
      setConversations(list);
    } catch (err) {
      console.error("Failed to load conversations", err);
    }
  }, []);

  const loadActiveMessages = useCallback(async (convId) => {
    if (!convId) {
      setMessages([]);
      return;
    }
    try {
      const msgList = await fetchMessages(convId);
      const mapped = msgList.map(m => ({
        sender: m.role,
        text: m.content,
        timestamp: m.created_at
      }));
      setMessages(mapped);
    } catch (err) {
      console.error("Failed to load messages", err);
    }
  }, []);

  const refreshAuth = useCallback(async () => {
    try {
      const authData = await fetchAuthMe();
      setAuth(authData);
      return authData;
    } catch (err) {
      console.error("Failed to fetch auth state", err);
      return null;
    }
  }, []);

  // Health and auth checks on load
  useEffect(() => {
    checkHealth();
    refreshAuth();
  }, [refreshAuth]);

  // Load conversations when auth changes
  useEffect(() => {
    loadConversations();
  }, [auth.mode, auth.user, loadConversations]);

  // Load messages when currentChatId changes
  useEffect(() => {
    loadActiveMessages(currentChatId);
  }, [currentChatId, loadActiveMessages]);

  const checkHealth = async () => {
    try {
      const data = await checkBackendHealth();
      setStatus({
        type: 'ok',
        text: 'Connected',
        localModel: data.local_model,
        geminiConfigured: data.gemini_configured
      });
      
      if (!data.gemini_configured && data.local_model) {
        setSelectedModel('local');
      }
    } catch (err) {
      setStatus({
        type: 'error',
        text: 'Offline (Backend down)',
        localModel: false,
        geminiConfigured: false
      });
    }
  };

  const startNewChat = () => {
    setCurrentChatId(null);
    setMessages([]);
    setError(null);
  };

  const selectConversation = (id) => {
    setCurrentChatId(id);
    setError(null);
  };

  const clearAllConversations = async () => {
    for (const c of conversations) {
      try {
        await deleteConversationApi(c.id);
      } catch (e) {}
    }
    setConversations([]);
    setCurrentChatId(null);
    setMessages([]);
  };

  const deleteConversation = async (id) => {
    try {
      await deleteConversationApi(id);
      if (currentChatId === id) {
        setCurrentChatId(null);
        setMessages([]);
      }
      loadConversations();
    } catch (err) {
      setError(err.message || 'Failed to delete conversation.');
    }
  };

  const handleSend = async (messageText, file = null) => {
    if ((!messageText.trim() && !file) || loading) return;
    
    // Check Guest limit in frontend if in guest mode
    if (!auth.authenticated && auth.usage >= auth.limit) {
      setError("Guest limit kamu sudah habis. Login dengan Google untuk melanjutkan.");
      return;
    }

    setError(null);
    setLoading(true);
    
    // Add temporary message to UI immediately for responsive feeling
    const tempUserMsg = {
      sender: 'user',
      text: messageText.trim(),
      timestamp: new Date().toISOString()
    };
    setMessages(prev => [...prev, tempUserMsg]);

    try {
      const history = messages.map(m => ({
        role: m.sender,
        content: m.text
      }));

      const data = await sendChatMessage(messageText.trim(), history, selectedModel, currentChatId);
      
      const botMsg = { 
        sender: 'assistant', 
        text: data.response || '(Tidak ada output)', 
        timestamp: new Date().toISOString() 
      };
      
      setMessages(prev => [...prev, botMsg]);
      
      // Update active Chat ID if it was a new chat
      if (!currentChatId && data.conversation_id) {
        setCurrentChatId(data.conversation_id);
      }
      
      // Refresh list & stats
      await loadConversations();
      await refreshAuth();
    } catch (err) {
      setError(err.message || 'Gagal mengirim pesan.');
      const errMsg = { 
        sender: 'assistant', 
        text: `Error: ${err.message || 'Gagal memproses pesan.'}`, 
        timestamp: new Date().toISOString() 
      };
      setMessages(prev => [...prev, errMsg]);
      await refreshAuth();
    } finally {
      setLoading(false);
    }
  };

  const logout = async () => {
    try {
      await logoutAuth();
      await refreshAuth();
      setCurrentChatId(null);
      setMessages([]);
    } catch (err) {
      console.error("Logout failed", err);
    }
  };

  return {
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
    sendMessage: handleSend,
    checkHealth,
    refreshAuth,
    logout
  };
}
