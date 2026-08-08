import React, { useRef, useEffect } from 'react';
import ChatMessage from './ChatMessage';
import EmptyState from './EmptyState';

export default function ChatWindow({ messages, onSelectSuggestion, loading }) {
  const containerRef = useRef(null);
  const lastMessagesLengthRef = useRef(messages.length);

  // Natural auto-scroll behavior
  useEffect(() => {
    if (containerRef.current) {
      const container = containerRef.current;
      const { scrollTop, scrollHeight, clientHeight } = container;
      
      // Check if user is scrolled near bottom (within 180px)
      const wasNearBottom = scrollHeight - scrollTop - clientHeight < 180;
      
      // Determine if a new user query was just submitted
      const isNewUserMessage = messages.length > lastMessagesLengthRef.current && 
                               messages[messages.length - 1]?.sender === 'user';
      
      // Scroll to bottom if user sent a message, or if they were already viewing the bottom
      if (isNewUserMessage || wasNearBottom || loading) {
        container.scrollTop = container.scrollHeight;
      }
    }
    lastMessagesLengthRef.current = messages.length;
  }, [messages, loading]);

  return (
    <div ref={containerRef} className="chat-messages-container">
      <div className="chat-messages-content">
        {messages.length === 0 ? (
          <EmptyState onSelectSuggestion={onSelectSuggestion} />
        ) : (
          <>
            {messages.map((msg, idx) => (
              <ChatMessage key={idx} message={msg} />
            ))}
            
            {loading && (
              <div className="message-row assistant typing">
                <div className="message-body">
                  <span className="sender-name">REXA</span>
                  <div className="message-content">
                    <div className="typing-dots">
                      <span></span>
                      <span></span>
                      <span></span>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Bottom spacer to prevent content from being covered by absolute positioned input box */}
            <div className="chat-bottom-spacer" style={{ height: '180px', flexShrink: 0 }} />
          </>
        )}
      </div>
    </div>
  );
}
