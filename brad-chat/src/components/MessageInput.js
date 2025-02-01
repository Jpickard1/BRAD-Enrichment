import React, { useState, useEffect } from 'react';
import './../App.css'; // Import the CSS file

function MessageInput({ onSendMessage }) {
  const [message, setMessage] = useState('');
  const [animatedPlaceholder, setAnimatedPlaceholder] = useState('');
  const placeholderText = "Please enter a comma separated list of genes. Example: MYOD, P53, CDK2";
  const [typingIndex, setTypingIndex] = useState(0);

  useEffect(() => {
    if (message) {
      setAnimatedPlaceholder(''); // Hide animation while user types
      return;
    }

    const interval = setInterval(() => {
      if (typingIndex < placeholderText.length) {
        setAnimatedPlaceholder((prev) => prev + placeholderText[typingIndex]);
        setTypingIndex((prev) => prev + 1);
      } else {
        setTimeout(() => {
          setTypingIndex(0);
          setAnimatedPlaceholder('');
        }, 2000); // Pause before restarting animation
      }
    }, 50); // Typing speed

    return () => clearInterval(interval);
  }, [message, typingIndex]);

  const handleSubmit = (e) => {
    e.preventDefault();

    if (message.trim()) {
      onSendMessage(message);
      setMessage('');
    }
  };

  return (
    <form onSubmit={handleSubmit} className="message-input">
      <input
        type="text"
        value={message}
        onChange={(e) => setMessage(e.target.value)}
        placeholder={animatedPlaceholder}
        className="input-box"
      />
      <button type="submit">{String.fromCodePoint('0x25B2')}</button>
    </form>
  );
}

export default MessageInput;
