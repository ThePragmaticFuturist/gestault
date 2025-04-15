# Step 11: Error Handling #

Okay, let's enhance the error handling in the React frontend (`App.jsx`) to provide clearer feedback to the user within the chat interface when API calls fail or when the backend indicates the LLM isn't ready.

**Key Changes:**

1.  **Error Bubbles:** Utilize the `isError: true` flag and corresponding CSS to display distinct error messages in the chat flow when `fetch` fails or the backend returns an error status.
2.  **Backend Error Content Detection:** Check if the *content* of a *successful* response from the backend starts with `[ERROR:` (like our backend sends when the LLM isn't ready) and flag it as an error message in the UI.
3.  **Robust Loading State:** Ensure `isLoading` is consistently managed, especially in error paths.

**Modify `App.jsx`**

```jsx
// src/App.jsx
import React, { useState, useEffect, useCallback } from 'react';
import ChatWindow from './components/ChatWindow/ChatWindow';
import MessageInput from './components/MessageInput/MessageInput';
import styles from './App.module.css';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';
console.log("Using API Base URL:", API_BASE_URL);

const INITIAL_MESSAGE = { role: 'assistant', content: 'Hello! How can I help you today?' };

function App() {
  const [messages, setMessages] = useState([INITIAL_MESSAGE]);
  const [inputValue, setInputValue] = useState('');
  const [sessionId, setSessionId] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isSessionLoading, setIsSessionLoading] = useState(false);

  // --- Add Error Message Helper ---
  const addErrorMessage = useCallback((content) => {
    setMessages(prevMessages => [
      ...prevMessages,
      { role: 'assistant', content: `Error: ${content}`, isError: true }
    ]);
  }, []); // Empty dependency, setMessages is stable

  // --- Helper Function to Create/Ensure Session ---
  const ensureSession = useCallback(async (forceNew = false) => {
    if (!forceNew && sessionId) {
      return sessionId;
    }
    if (forceNew) {
      setSessionId(null);
    }

    setIsSessionLoading(true);
    let createdSessionId = null;

    try {
      console.log(forceNew ? "Forcing new session..." : "Creating initial session...");
      const response = await fetch(`${API_BASE_URL}/api/v1/sessions/`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Accept': 'application/json' },
        body: JSON.stringify({ name: "React UI Session" }),
      });

      if (!response.ok) {
        let errorDetail = `HTTP ${response.status}`;
        try {
          const errorData = await response.json();
          errorDetail = errorData.detail || errorDetail;
        } catch (e) { /* Ignore */ }
        throw new Error(`Failed to create session. ${errorDetail}`); // Throw specific error
      }

      const newSessionData = await response.json();
      createdSessionId = newSessionData.id;
      console.log("New session created/ensured:", createdSessionId);
      setSessionId(createdSessionId);

    } catch (error) {
      console.error("Session creation error:", error);
      addErrorMessage(`Could not start a session. ${error.message}`); // Use helper
      setSessionId(null);
    } finally {
      setIsSessionLoading(false);
    }
    return createdSessionId;
  }, [sessionId, addErrorMessage]); // Add addErrorMessage dependency


  // --- Effect to create session on initial load ---
  useEffect(() => {
    console.log("App component mounted. Ensuring initial session exists.");
    // Don't block input on initial load, let ensureSession handle its state
    // setIsLoading(true); // Removed
    (async () => {
      await ensureSession();
      // setIsLoading(false); // Removed
    })();
  }, [ensureSession]);


  // --- Handle Sending a Message ---
  const handleSendMessage = async () => {
    const messageText = inputValue.trim();
    if (!messageText || isLoading || isSessionLoading) return;

    const userMessage = { role: 'user', content: messageText };
    setMessages(prevMessages => [...prevMessages, userMessage]);
    setInputValue('');
    setIsLoading(true); // Start loading for message response

    let currentSessionId = sessionId;

    try {
        if (!currentSessionId) {
            currentSessionId = await ensureSession();
        }
        if (!currentSessionId) {
             // Error message already added by ensureSession if it failed
             console.error("Cannot send message without a valid session ID.");
             setIsLoading(false); // Stop loading as we can't proceed
             return;
        }

        console.log(`Sending message to session ${currentSessionId}:`, messageText);

        const response = await fetch(`${API_BASE_URL}/api/v1/sessions/${currentSessionId}/messages`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', 'Accept': 'application/json' },
            body: JSON.stringify({ content: messageText, role: 'user' }),
        });

        // --- Handle potential HTTP errors ---
        if (!response.ok) {
            let errorDetail = `Request failed with status ${response.status}`;
            try {
                const errorData = await response.json();
                // Try to get backend's specific error detail
                errorDetail = errorData.detail || errorDetail;
            } catch (e) { /* Ignore if error body isn't JSON */ }
            // Throw an error to be caught by the catch block
            throw new Error(errorDetail);
        }
        // --- End HTTP error check ---

        const assistantMessageData = await response.json();
        console.log("Received assistant response data:", assistantMessageData);

        // --- Check if backend sent an error *within* the content ---
        let isBackendError = false;
        if (assistantMessageData.content && typeof assistantMessageData.content === 'string' && assistantMessageData.content.startsWith('[ERROR:')) {
            console.warn("Backend indicated an error in response content:", assistantMessageData.content);
            isBackendError = true;
        }
        // --- End backend error check ---

        // Add assistant message, including the isError flag if detected
        setMessages(prevMessages => [
            ...prevMessages,
            { ...assistantMessageData, isError: isBackendError } // Add isError flag
        ]);

    } catch (error) {
      console.error("Error during message send/receive:", error);
      // Use the helper to add a formatted error bubble
      addErrorMessage(`Could not get response. ${error.message}`);
    } finally {
      setIsLoading(false); // Stop loading indicator
    }
  };

  // --- Handler for New Chat Button ---
  // ... (startNewChat function remains the same) ...


  return (
    <div className={styles.appContainer}>
      {/* --- Header Bar --- */}
      {/* ... (header bar structure remains the same) ... */}
      <div className={styles.headerBar}>
         <h1 className={styles.appTitle}>RAG LLM Chat</h1>
         <button onClick={startNewChat} className={styles.newChatButton} disabled={isLoading || isSessionLoading}>
             New Chat
         </button>
      </div>

      <ChatWindow messages={messages} />
      <MessageInput
        value={inputValue}
        onChange={(e) => setInputValue(e.target.value)}
        onSend={handleSendMessage}
        disabled={isLoading || isSessionLoading}
      />
       {/* Loading Indicators */}
       {isSessionLoading && <div className={styles.loadingIndicator}>Creating session...</div>}
       {isLoading && !isSessionLoading && <div className={styles.loadingIndicator}>Assistant is thinking...</div>}
    </div>
  );
}

export default App;
```

**Explanation of Changes:**

1.  **`addErrorMessage` Helper:** A `useCallback` memoized helper function is added to consistently format and add error messages to the `messages` state. This reduces code repetition in the `catch` blocks.
2.  **`ensureSession` Error Handling:**
    *   The `catch` block now calls `addErrorMessage` to display the session creation failure in the chat UI.
    *   It throws a more specific `Error` object including backend details if available.
3.  **`useEffect` Loading:** Removed the `setIsLoading(true/false)` calls around the initial `ensureSession`. We only want `isLoading` to reflect the *message sending* state to avoid blocking input unnecessarily right after the app loads. `isSessionLoading` handles the session creation indication.
4.  **`handleSendMessage` Error Handling:**
    *   The `catch` block now calls `addErrorMessage` for consistency.
    *   Added an explicit check `if (!response.ok)` after the `fetch` call. If the HTTP status indicates an error (e.g., 4xx, 5xx), it attempts to parse the error detail from the backend's JSON response and then *throws an Error*. This ensures that HTTP errors are caught by the main `catch` block and displayed using `addErrorMessage`.
5.  **Backend Error Content Check:** After successfully parsing the JSON response (`assistantMessageData`), it checks if `assistantMessageData.content` starts with `"[ERROR:"`. If it does, it sets `isBackendError = true`.
6.  **Adding Assistant Message:** When adding the received message to the state, it now includes `isError: isBackendError`. This ensures that even if the API call itself was successful (status 2xx), but the backend internally failed (e.g., LLM not ready) and put an error message in the content, that message will still be styled as an error in the UI.

**How to Test:**

1.  Restart the frontend dev server (`npm run dev`).
2.  Hard refresh the browser.
3.  **Test Session Failure:** Stop your backend server. Refresh the React app. You should see the "Error: Could not start a session..." message appear in the chat. The input should remain usable (as `isLoading` isn't stuck). Start the backend server again.
4.  **Test LLM Not Ready:**
    *   Start the backend server.
    *   Make sure **no LLM is loaded** (e.g., check `GET /api/v1/models/status`, status should be `Inactive` or `Failed`).
    *   Send a message in the React app.
    *   **Expected:** The backend should return a 201 response, but the `content` will be `"[ERROR: LLM not ready...]"`. The React app should detect this, and display that message in a red error bubble. The input should become enabled again.
5.  **Test General API Failure:** Load an LLM successfully. Send a message. While the backend is processing, quickly stop the backend server. The `fetch` call in the React app should fail (network error). You should see an "Error: Could not get response. TypeError: NetworkError..." message in the chat.
6.  **Test Normal Operation:** Load an LLM, send messages, verify normal responses appear without error styling.

This provides more robust visual feedback to the user about different kinds of problems that might occur during the interaction with the backend.
