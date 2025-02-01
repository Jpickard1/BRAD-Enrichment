import React, { useEffect, useState } from 'react';
import MessageList from './MessageList';
import MessageInput from './MessageInput';

import "highlight.js/styles/github.css";
import hljs from "highlight.js";

function ChatContainer({ messages, setMessages, setVideoSrc }) {
  useEffect(() => {
    hljs.highlightAll();
  }, [messages]);

  const handleSendMessage = async (message) => {
    setMessages([...messages, { id: Date.now(), text: message, sender: 'user' }]);
  
    let data = { "message": message };
    try {
      // Call the backend API
      const response = await fetch('/api/enrichment', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(data),
      });
  
      const result = await response.json();
      const file_url = result['file-url']; // Extract file URL
  
      // setMessages((messages) => [...messages, { id: Date.now(), text: bot_response, sender: 'bot' }]);
  
      // Automatically download the file if available
      if (file_url) {
        const fileResponse = await fetch(file_url);
        const blob = await fileResponse.blob();
        const url = window.URL.createObjectURL(blob);
        
        const a = document.createElement("a");
        a.href = url;
        a.download = "brad_enrichment.xlsx";
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
      }
    } catch (error) {
      console.error('Error:', error);
    }
  };  

  return (
    <div className="chat-container">
      <MessageInput onSendMessage={handleSendMessage} />
    </div>
  );
}

export default ChatContainer;
