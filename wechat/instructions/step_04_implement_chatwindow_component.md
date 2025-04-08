**Step 4: Implement `ChatWindow` Component**

*   **What:** A container component that receives the list of messages and renders the `MessageList`.
*   **Why:** To provide a dedicated area for the conversation display, handling potential scrolling.
*   **How:**

    ```jsx
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
    ```

    *   Create `src/components/ChatWindow/ChatWindow.module.css`:
        ```css
        /* src/components/ChatWindow/ChatWindow.module.css */
        .chatWindow {
          flex-grow: 1; /* Take up remaining vertical space */
          overflow-y: auto; /* Enable vertical scrolling */
          padding: 15px;
          background-color: #ffffff;
          display: flex; /* Needed for flex-grow */
          flex-direction: column; /* Stack messages vertically */
        }
        ```
