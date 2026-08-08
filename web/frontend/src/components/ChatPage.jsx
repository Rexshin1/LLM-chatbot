import React from 'react';
import ChatWindow from './ChatWindow';
import ChatInput from './ChatInput';

export default function ChatPage({
  messages,
  onSelectSuggestion,
  loading,
  input,
  setInput,
  onSend,
  selectedModel,
  onModelChange,
  attachedFile,
  onAttachFile,
  onRemoveFile
}) {
  return (
    <div className="chat-page-layout">
      <div className="chat-workspace">
        <ChatWindow 
          messages={messages} 
          onSelectSuggestion={onSelectSuggestion} 
          loading={loading} 
        />
        <ChatInput 
          value={input} 
          onChange={setInput} 
          onSend={onSend} 
          loading={loading}
          selectedModel={selectedModel}
          onModelChange={onModelChange}
          attachedFile={attachedFile}
          onAttachFile={onAttachFile}
          onRemoveFile={onRemoveFile}
        />
      </div>
    </div>
  );
}
