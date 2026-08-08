import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

export default function ChatMessage({ message }) {
  const { sender, text } = message;

  const components = {
    code({ node, inline, className, children, ...props }) {
      const codeText = String(children).replace(/\n$/, '');
      const match = /language-(\w+)/.exec(className || '');
      const isInline = inline || (!match && !codeText.includes('\n'));

      if (isInline) {
        return (
          <code className="inline-code" {...props}>
            {codeText}
          </code>
        );
      }

      const language = match ? match[1] : 'code';

      const handleCopy = (e) => {
        navigator.clipboard.writeText(codeText);
        const btn = e.currentTarget;
        btn.textContent = 'Copied!';
        setTimeout(() => {
          btn.textContent = 'Copy';
        }, 2000);
      };

      return (
        <div className="code-block-container">
          <div className="code-block-header">
            <span className="code-lang">{language}</span>
            <button type="button" onClick={handleCopy} className="btn-copy-code">Copy</button>
          </div>
          <pre><code className={className} {...props}>{codeText}</code></pre>
        </div>
      );
    },
    table({ children }) {
      return (
        <div className="table-wrapper">
          <table>{children}</table>
        </div>
      );
    }
  };

  return (
    <div className={`message-row ${sender}`}>
      <div className="message-body">
        {sender === 'assistant' && <span className="sender-name">REXA</span>}
        
        {/* Render attached file if present in message metadata */}
        {message.file && (
          <div className="message-attached-file-preview">
            {message.file.type.startsWith('image/') ? (
              <img 
                src={message.file.url} 
                alt="Attached upload" 
                className="message-attached-image" 
              />
            ) : (
              <div className="message-attached-doc">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{marginRight: '8px', color: 'var(--text-secondary)'}}>
                  <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                  <polyline points="14 2 14 8 20 8"/>
                </svg>
                <div className="doc-details">
                  <span className="doc-name">{message.file.name}</span>
                  <span className="doc-size">Attached Document</span>
                </div>
              </div>
            )}
          </div>
        )}

        <div className="message-content">
          <ReactMarkdown remarkPlugins={[remarkGfm]} components={components}>
            {text}
          </ReactMarkdown>
        </div>
      </div>
    </div>
  );
}
