// src/components/MessageInput/MessageInput.jsx
import React from 'react';
import styles from './MessageInput.module.css';

// Destructure the new 'disabled' prop
function MessageInput({ value, onChange, onSend, disabled }) {

  const handleKeyPress = (event) => {
    // Don't send if disabled or shift+enter
    if (disabled || (event.key === 'Enter' && event.shiftKey)) {
        return;
    }
    if (event.key === 'Enter') {
      event.preventDefault();
      onSend();
    }
  };

  return (
    <div className={styles.inputArea}>
      <textarea
        className={styles.inputField}
        value={value}
        onChange={onChange}
        onKeyPress={handleKeyPress}
        placeholder={disabled ? "Waiting for response..." : "Type your message..."}
        rows="1"
        disabled={disabled} // Disable the textarea
      />
      <button
        className={styles.sendButton}
        onClick={onSend}
        disabled={disabled} // Disable the button
      >
        Send
      </button>
    </div>
  );
}

export default MessageInput;