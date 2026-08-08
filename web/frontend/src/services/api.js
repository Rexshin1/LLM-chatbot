export async function sendChatMessage(message, history = [], model = "local", conversationId = null) {
  const messagesPayload = [
    ...history.map(h => ({
      role: h.role || (h.sender === "assistant" ? "assistant" : "user"),
      content: h.content || h.text || ""
    })),
    { role: "user", content: message }
  ];

  const response = await fetch("/api/chat", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ 
      messages: messagesPayload,
      model: model,
      temperature: 0.7,
      top_k: 10,
      max_new_tokens: 150,
      conversation_id: conversationId
    })
  });
  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.error || errorData.detail || "Gagal memproses pesan.");
  }
  return await response.json();
}

export async function checkBackendHealth() {
  const response = await fetch("/api/health");
  if (!response.ok) {
    throw new Error("Backend offline");
  }
  return await response.json();
}

export async function fetchAuthMe() {
  const response = await fetch("/api/auth/me");
  if (!response.ok) {
    throw new Error("Failed to fetch auth state");
  }
  return await response.json();
}

export async function logoutAuth() {
  const response = await fetch("/api/auth/logout", {
    method: "POST"
  });
  if (!response.ok) {
    throw new Error("Failed to logout");
  }
  return await response.json();
}

export async function fetchConversations() {
  const response = await fetch("/api/conversations");
  if (!response.ok) {
    throw new Error("Failed to fetch conversations");
  }
  return await response.json();
}

export async function fetchMessages(convId) {
  const response = await fetch(`/api/conversations/${convId}`);
  if (!response.ok) {
    throw new Error("Failed to fetch messages");
  }
  return await response.json();
}

export async function deleteConversationApi(convId) {
  const response = await fetch(`/api/conversations/${convId}`, {
    method: "DELETE"
  });
  if (!response.ok) {
    throw new Error("Failed to delete conversation");
  }
  return await response.json();
}
