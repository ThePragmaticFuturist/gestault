**Step 5: Implement `MessageList` Component**

*   **What:** Renders the individual message bubbles from the `messages` array. Handles scrolling to the bottom.
*   **Why:** To iterate through messages and manage the display flow within the `ChatWindow`.
*   **How:**

    ```jsx
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
    ```

    *   Create `src/components/MessageList/MessageList.module.css`:
        ```css
        /* src/components/MessageList/MessageList.module.css */
        .messageList {
          display: flex;
          flex-direction: column;
          gap: 10px; /* Space between messages */
        }
        ```
