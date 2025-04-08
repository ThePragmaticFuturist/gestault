**Step 6: Implement `MessageBubble` Component**

*   **What:** Displays a single message, styled differently based on the role (`user` or `assistant`).
*   **Why:** The visual representation of each turn in the conversation.
*   **How:**

    ```jsx
    // src/components/MessageBubble/MessageBubble.jsx
    import React from 'react';
    import styles from './MessageBubble.module.css';

    function MessageBubble({ message }) {
      const { role, content } = message;
      const isUser = role === 'user';

      // Apply different styles based on the role
      const bubbleClass = isUser ? styles.userBubble : styles.assistantBubble;
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
    ```

    *   Create `src/components/MessageBubble/MessageBubble.module.css`:
        ```css
        /* src/components/MessageBubble/MessageBubble.module.css */
        .messageContainer {
          display: flex;
          margin-bottom: 10px; /* Consistent spacing */
        }

        .messageBubble {
          max-width: 75%; /* Prevent bubbles from being too wide */
          padding: 10px 15px;
          border-radius: 18px; /* Rounded bubbles */
          word-wrap: break-word; /* Break long words */
          line-height: 1.4;
        }

        /* Alignment */
        .userAlignment {
          justify-content: flex-end; /* Align user messages to the right */
        }

        .assistantAlignment {
          justify-content: flex-start; /* Align assistant messages to the left */
        }

        /* Bubble specific styles */
        .userBubble {
          background-color: #0b93f6; /* Example blue for user */
          color: white;
          border-bottom-right-radius: 4px; /* Slightly different shape */
        }

        .assistantBubble {
          background-color: #e5e5ea; /* Example grey for assistant */
          color: black;
          border-bottom-left-radius: 4px; /* Slightly different shape */
        }
        ```
