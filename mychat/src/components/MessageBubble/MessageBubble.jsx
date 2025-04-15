// src/components/MessageBubble/MessageBubble.jsx
import React from 'react';
import styles from './MessageBubble.module.css';

function MessageBubble({ message }) {
  // Destructure isError, default to false if not present
  const { role, content, isError = false } = message;
  const isUser = role === 'user';

  const bubbleClass = isUser
    ? styles.userBubble
    : (isError ? styles.errorBubble : styles.assistantBubble); // Use error style if isError is true

  const alignmentClass = isUser ? styles.userAlignment : styles.assistantAlignment;

  return (
    <div className={`${styles.messageContainer} ${alignmentClass}`}>
      <div className={`${styles.messageBubble} ${bubbleClass}`}>
        {content}
      </div>
    </div>
  );
}

export default MessageBubble;