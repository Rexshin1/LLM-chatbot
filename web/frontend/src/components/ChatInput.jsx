import React, { useRef, useEffect } from 'react';

export default function ChatInput({ 
  value, 
  onChange, 
  onSend, 
  loading,
  selectedModel,
  onModelChange,
  attachedFile,
  onAttachFile,
  onRemoveFile
}) {
  const textareaRef = useRef(null);
  const fileInputRef = useRef(null);

  // Auto-resize textarea based on content
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 180)}px`;
    }
  }, [value]);

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      onSend();
    }
  };

  // Fixed: reset input value so the same file can be re-selected
  const handleAttachClick = () => {
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
      fileInputRef.current.click();
    }
  };

  const handleFileChange = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const fileUrl = URL.createObjectURL(file);
    onAttachFile({
      name: file.name,
      type: file.type,
      url: fileUrl,
      raw: file
    });
  };

  const isDisabled = loading || (!value.trim() && !attachedFile);

  return (
    <footer className="chat-input-container">
      <div className="input-box-wrapper">

        {/* Attachment preview */}
        {attachedFile && (
          <div className="input-file-preview">
            {attachedFile.type.startsWith('image/') ? (
              <div className="file-preview-thumb">
                <img src={attachedFile.url} alt="preview" />
                <button type="button" onClick={onRemoveFile} className="btn-remove-file">×</button>
              </div>
            ) : (
              <div className="file-preview-doc">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{marginRight:'6px',flexShrink:0,color:'var(--text-secondary)'}}>
                  <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                  <polyline points="14 2 14 8 20 8"/>
                </svg>
                <span style={{maxWidth:'140px',overflow:'hidden',textOverflow:'ellipsis',whiteSpace:'nowrap',fontSize:'12px'}}>
                  {attachedFile.name}
                </span>
                <button type="button" onClick={onRemoveFile} className="btn-remove-file">×</button>
              </div>
            )}
          </div>
        )}

        <textarea
          ref={textareaRef}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask REXA anything..."
          rows="1"
          disabled={loading}
        />

        <div className="input-toolbar">
          <div className="toolbar-left">

            {/* Attach button — fixed */}
            <button
              type="button"
              onClick={handleAttachClick}
              className="toolbar-btn"
              title="Attach image or file"
              disabled={loading}
            >
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{marginRight:'4px'}}>
                <path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48"/>
              </svg>
              Attach
            </button>

            {/* Hidden file input */}
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileChange}
              style={{display:'none'}}
              accept="image/*,application/pdf,text/plain,.py,.js,.ts,.jsx,.tsx,.json,.md,.csv"
            />

            {/* Model selector */}
            <div className="toolbar-model-dropdown-container">
              <select
                value={selectedModel}
                onChange={(e) => onModelChange(e.target.value)}
                className="toolbar-model-select"
                disabled={loading}
              >
                <option value="gemini">REXA 2.0</option>
                <option value="local">REXA 1.0</option>
              </select>
            </div>

          </div>

          {/* Send button */}
          <button
            type="button"
            onClick={onSend}
            className="btn-send"
            disabled={isDisabled}
            title="Send message"
            aria-label="Send message"
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <line x1="12" y1="19" x2="12" y2="5"/>
              <polyline points="5 12 12 5 19 12"/>
            </svg>
          </button>
        </div>
      </div>

      <div className="input-footer">
        REXA may make mistakes. Verify important information.
      </div>
    </footer>
  );
}
