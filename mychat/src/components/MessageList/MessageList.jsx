// src/components/MessageList/MessageList.jsx
import React, { useEffect, useRef } from 'react';
import MessageBubble from '../MessageBubble/MessageBubble';
import styles from './MessageList.module.css';

function MessageList({ messages }) {
  const messagesEndRef = useRef(null); // Ref to scroll to

  // Function to scroll to the bottom
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  // Scroll to bottom whenever messages array changes
  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  return (
    <div className={styles.messageList}>
      {messages.map((msg, index) => (
        <MessageBubble key={index} message={msg} />
      ))}
      {/* Empty div at the end to scroll to */}
      <div ref={messagesEndRef} />
    </div>
  );
}

export default MessageList;