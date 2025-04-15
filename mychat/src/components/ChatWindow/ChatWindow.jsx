// src/components/ChatWindow/ChatWindow.jsx
import React from 'react';
import MessageList from '../MessageList/MessageList';
import styles from './ChatWindow.module.css';

function ChatWindow({ messages }) {
  return (
    <div className={styles.chatWindow}>
      <MessageList messages={messages} />
    </div>
  );
}

export default ChatWindow;