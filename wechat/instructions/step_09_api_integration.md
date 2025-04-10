# Step 9: API Integration

Okay, let's integrate the API call to your RAG LLM backend into the React app. We'll replace the `setTimeout` echo with a `fetch` call and add minimal session management (creating a session on the first message).

[Reasoning behind the code](reasoning/r09.md)  

**1. Configure API Base URL**

*   **Create `.env` file:** In the root of your `rag-chat-ui` project (alongside `package.json`), create a file named `.env`.
*   **Add URL:** Add the following line to `.env`, replacing the URL if your backend runs elsewhere:
    ```.env
    VITE_API_BASE_URL=http://localhost:8000
    ```
    *   **Note:** Vite requires environment variables exposed to the browser to be prefixed with `VITE_`.
*   **Add to `.gitignore`:** Make sure `.env` is listed in your project's `.gitignore` file.

**2. Update `App.jsx`**

Modify `src/App.jsx` to include session state, API call logic, loading state, and error handling.

```jsx
// src/App.jsx
import React, { useState, useEffect } from 'react';
import ChatWindow from './components/ChatWindow/ChatWindow';
import MessageInput from './components/MessageInput/MessageInput';
import styles from './App.module.css';

// Get API base URL from environment variables (defined in .env)
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'; // Fallback
console.log("Using API Base URL:", API_BASE_URL); // For debugging

function App() {
  const [messages, setMessages] = useState([
    { role: 'assistant', content: 'Hello! How can I help you today?' }
  ]);
  const [inputValue, setInputValue] = useState('');
  const [sessionId, setSessionId] = useState(null); // State for the session ID
  const [isLoading, setIsLoading] = useState(false); // State for loading indicator

  // --- Helper Function to Ensure Session Exists ---
  const ensureSession = async () => {
    if (sessionId) {
      console.log("Using existing session:", sessionId);
      return sessionId; // Return existing session ID
    }

    console.log("No active session, creating a new one...");
    setIsLoading(true); // Show loading while creating session
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/sessions/`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json',
        },
        // Optionally add default name or RAG docs here if desired
        body: JSON.stringify({ name: "React UI Session" }),
      });

      if (!response.ok) {
        // Attempt to parse error response from backend
        let errorDetail = `HTTP error! status: ${response.status}`;
        try {
          const errorData = await response.json();
          errorDetail = errorData.detail || errorDetail;
        } catch (e) { /* Ignore if error body isn't JSON */ }
        throw new Error(errorDetail);
      }

      const newSessionData = await response.json();
      console.log("New session created:", newSessionData.id);
      setSessionId(newSessionData.id); // Store the new session ID
      return newSessionData.id; // Return the new ID
    } catch (error) {
      console.error("Failed to create session:", error);
      // Display error in chat?
      setMessages(prevMessages => [
        ...prevMessages,
        { role: 'assistant', content: `Error: Could not start a session. ${error.message}`, isError: true } // Add isError flag
      ]);
      return null; // Indicate failure
    } finally {
      // Doesn't strictly need isLoading here, but good practice
      // setIsLoading(false); // We set loading false after message response
    }
  };

  // --- Handle Sending a Message ---
  const handleSendMessage = async () => { // Make the handler async
    const messageText = inputValue.trim();
    if (!messageText) return;

    const userMessage = { role: 'user', content: messageText };
    setMessages(prevMessages => [...prevMessages, userMessage]);
    setInputValue(''); // Clear input immediately
    setIsLoading(true); // Start loading indicator

    let currentSessionId = sessionId;

    try {
        // Ensure we have a session ID before sending the message
        if (!currentSessionId) {
            currentSessionId = await ensureSession();
        }

        // If session creation failed or still no ID, stop here
        if (!currentSessionId) {
             console.error("Cannot send message without a valid session ID.");
             // Error message was already added by ensureSession
             setIsLoading(false);
             return;
        }

        console.log(`Sending message to session ${currentSessionId}:`, messageText);

        // Send message to the backend API
        const response = await fetch(`${API_BASE_URL}/api/v1/sessions/${currentSessionId}/messages`, {
            method: 'POST',
            headers: {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            },
            body: JSON.stringify({ content: messageText, role: 'user' }), // role is 'user' here
        });

        if (!response.ok) {
            let errorDetail = `Error fetching response: ${response.status}`;
            try {
                const errorData = await response.json();
                errorDetail = errorData.detail || errorDetail;
            } catch (e) { /* Ignore if error body isn't JSON */ }
            throw new Error(errorDetail);
        }

        // Response should contain the assistant's message
        const assistantMessageData = await response.json();
        console.log("Received assistant response:", assistantMessageData);

        // Add assistant message to the state
        // Assuming backend returns { role: 'assistant', content: '...' } matching our structure
        setMessages(prevMessages => [...prevMessages, assistantMessageData]);

    } catch (error) {
      console.error("Error sending/receiving message:", error);
      // Add an error message bubble to the chat
      setMessages(prevMessages => [
        ...prevMessages,
        { role: 'assistant', content: `Sorry, something went wrong: ${error.message}`, isError: true } // Add isError flag
      ]);
    } finally {
      setIsLoading(false); // Stop loading indicator regardless of success/error
    }
  };


  return (
    <div className={styles.appContainer}>
      <h1 className={styles.appTitle}>RAG LLM Chat</h1>
      <ChatWindow messages={messages} />
      <MessageInput
        value={inputValue}
        onChange={(e) => setInputValue(e.target.value)}
        onSend={handleSendMessage}
        // Disable input while loading
        disabled={isLoading}
      />
       {/* Optional: Add a simple loading text indicator */}
       {isLoading && <div className={styles.loadingIndicator}>Assistant is thinking...</div>}
    </div>
  );
}

export default App;

```

**3. Update `MessageInput.jsx`**

Pass the `disabled` prop to the underlying elements.

```jsx
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
```

**4. Update `MessageBubble.jsx` (Optional Styling for Errors)**

Add styling based on the `isError` flag we added.

```jsx
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
```

*   Add styles for `.errorBubble` in `src/components/MessageBubble/MessageBubble.module.css`:
    ```css
    /* src/components/MessageBubble/MessageBubble.module.css */
    /* ... (existing styles) ... */

    .errorBubble {
      background-color: #ffebee; /* Light red background */
      color: #c62828; /* Darker red text */
      border: 1px solid #e57373;
      border-bottom-left-radius: 4px;
    }
    ```

**5. Add Loading Indicator Style (Optional)**

*   Add style to `src/App.module.css`:
    ```css
    /* src/App.module.css */
    /* ... (existing styles) ... */

    .loadingIndicator {
        text-align: center;
        padding: 5px;
        font-style: italic;
        color: #888;
        background-color: #f0f0f0; /* Match input area */
        border-top: 1px solid #ccc; /* Match input area */
        flex-shrink: 0;
    }
    ```

**Explanation of Changes:**

1.  **Environment Variable:** Reads the backend URL from `.env`.
2.  **State:** Added `sessionId` and `isLoading`.
3.  **`ensureSession` Function:** Handles creating a new session via `POST /api/v1/sessions/` if `sessionId` is null. Stores the returned ID. Includes basic error handling.
4.  **`handleSendMessage` (Async):**
    *   Marked as `async`.
    *   Clears input and sets `isLoading` to `true`.
    *   Calls `await ensureSession()` to get a valid ID. Exits if session creation fails.
    *   Uses `fetch` to make the `POST` request to `/api/v1/sessions/{currentSessionId}/messages` with the user's message content.
    *   **Handles Success:** If `response.ok`, it parses the JSON (expecting the assistant's message) and updates the `messages` state.
    *   **Handles Errors:** If `!response.ok` or if `fetch` throws an error (network issue), it catches the error, logs it, and adds a specific error message bubble to the chat.
    *   **`finally` block:** Sets `isLoading` back to `false` regardless of success or failure.
5.  **Component Updates:**
    *   `MessageInput` now accepts and uses the `disabled` prop.
    *   `MessageBubble` optionally checks for an `isError` prop to apply different styling.
    *   A simple text loading indicator is added to `App.jsx`.

**How to Test:**

1.  **Ensure Backend is Running:** Make sure your Python RAG LLM server is running on the URL specified in your `.env` file (e.g., `http://localhost:8000`).
2.  **Restart Frontend Dev Server:** Stop (`Ctrl+C`) and restart the Vite dev server (`npm run dev`) to ensure it picks up the `.env` file changes.
3.  **Open App:** Refresh the React app in your browser (or open the new URL if the port changed).
4.  **Send First Message:** Type a message and send it.
    *   **Check Backend Logs:** You should see a request to `POST /api/v1/sessions/` first (creating the session).
    *   **Check Frontend Console:** You should see logs about creating the session and using the new ID.
    *   **Check Backend Logs:** You should then see a request to `POST /api/v1/sessions/{session_id}/messages`.
    *   **UI:** The input should be disabled briefly ("Waiting for response..."). You should then see the actual assistant response appear (or an error message if something went wrong).
5.  **Send Subsequent Messages:** Send more messages. It should reuse the same `sessionId`.

You now have a React chat UI that communicates directly with your backend RAG LLM server!
