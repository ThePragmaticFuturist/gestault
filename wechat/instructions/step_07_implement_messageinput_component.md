**Step 7: Implement `MessageInput` Component**

*   **What:** Provides the text input field and send button.
*   **Why:** Allows the user to type and submit their messages.
*   **How:**

    ```jsx
    // src/components/MessageInput/MessageInput.jsx
    import React from 'react';
    import styles from './MessageInput.module.css';

    function MessageInput({ value, onChange, onSend }) {

      // Handle sending message on Enter key press
      const handleKeyPress = (event) => {
        if (event.key === 'Enter' && !event.shiftKey) { // Send on Enter, allow Shift+Enter for newline
          event.preventDefault(); // Prevent default form submission/newline
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
            placeholder="Type your message..."
            rows="1" // Start with one row, CSS will handle expansion
          />
          <button className={styles.sendButton} onClick={onSend}>
            Send
          </button>
        </div>
      );
    }

    export default MessageInput;
    ```

    *   Create `src/components/MessageInput/MessageInput.module.css`:
        ```css
        /* src/components/MessageInput/MessageInput.module.css */
        .inputArea {
          display: flex;
          padding: 10px;
          border-top: 1px solid #ccc;
          background-color: #f0f0f0;
          flex-shrink: 0; /* Prevent input area from shrinking */
        }

        .inputField {
          flex-grow: 1;
          border: 1px solid #ccc;
          border-radius: 18px;
          padding: 10px 15px;
          font-size: 1em;
          resize: none; /* Prevent manual resizing */
          overflow-y: auto; /* Allow scrolling if text gets tall */
          max-height: 100px; /* Limit max height */
          margin-right: 10px;
          line-height: 1.4;
          font-family: inherit; /* Use the app's font */
        }

        .inputField:focus {
          outline: none;
          border-color: #4a90e2;
          box-shadow: 0 0 0 2px rgba(74, 144, 226, 0.2);
        }

        .sendButton {
          padding: 10px 20px;
          border: none;
          background-color: #4a90e2;
          color: white;
          border-radius: 18px;
          cursor: pointer;
          font-size: 1em;
          font-weight: bold;
          transition: background-color 0.2s ease;
        }

        .sendButton:hover {
          background-color: #357abd;
        }

        .sendButton:disabled {
          background-color: #ccc;
          cursor: not-allowed;
        }
        ```
