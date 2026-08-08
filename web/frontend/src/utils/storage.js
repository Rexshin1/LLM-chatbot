const CONVERSATIONS_KEY = 'rexshin_ai_conversations';

export const getConversations = () => {
  const data = localStorage.getItem(CONVERSATIONS_KEY);
  return data ? JSON.parse(data) : [];
};

export const saveConversations = (conversations) => {
  localStorage.setItem(CONVERSATIONS_KEY, JSON.stringify(conversations));
};
