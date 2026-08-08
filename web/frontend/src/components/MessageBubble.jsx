import React from 'react';

export default function MessageBubble({ message }) {
  const { sender, text } = message;

  // Simple Markdown-like formatting helper
  const formatText = (rawText) => {
    if (!rawText) return '';
    
    // Escape HTML to prevent XSS
    let escaped = rawText
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
      
    // Format bold: **text** -> <strong>text</strong>
    escaped = escaped.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    
    // Format inline code: `code` -> <code>code</code>
    escaped = escaped.replace(/`(.*?)`/g, '<code>$1</code>');
    
    // Format newlines: \n -> <br />
    escaped = escaped.replace(/\n/g, '<br />');
    
    return <span dangerouslySetInnerHTML={{ __html: escaped }} />;
  };

  return (
    <div className={`message ${sender}`}>
      <div className="avatar">{sender === 'user' ? '👤' : '🤖'}</div>
      <div className="message-content">
        {formatText(text)}
      </div>
    </div>
  );
}
