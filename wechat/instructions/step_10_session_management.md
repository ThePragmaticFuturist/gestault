# Step 10: Implement session management in the React frontend (`App.jsx`)

Okay, let's implement session management in the React frontend (`App.jsx`). We'll automatically create a new session when the component first loads and store its ID in the state. We'll then use this ID for sending messages. We'll also add a "New Chat" button to allow users to reset the state and start fresh (which will trigger a new session creation on the next message).

**1. Modify `App.jsx`**

We'll add:
*   A `useEffect` hook to create the session on initial load.
*   A "New Chat" button and its handler function.
*   Refine loading/error handling in `ensureSession`.

```jsx
// // src/App.jsx   <<<--- Step 10
import React, { useState, useEffect, useCallback } from 'react'; // Import useCallback
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
  const [isSessionLoading, setIsSessionLoading] = useState(false); // Separate loading for session creation

  // --- Helper Function to Create/Ensure Session ---
  // useCallback helps prevent unnecessary recreation if passed as dependency elsewhere
  const ensureSession = useCallback(async (forceNew = false) => {
    // If we don't need a new one and already have one, return it
    if (!forceNew && sessionId) {
      console.log("Using existing session:", sessionId);
      return sessionId;
    }

    // If forcing new, clear existing ID first
    if(forceNew) {
        console.log("Forcing new session creation...");
        setSessionId(null); // Clear state immediately
    } else {
        console.log("No active session, creating a new one...");
    }

    setIsSessionLoading(true); // Indicate session specific loading
    setIsLoading(true); // Also set general loading for input disabling
    let createdSessionId = null;

    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/sessions/`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json',
        },
        body: JSON.stringify({ name: "React UI Session" }),
      });

      if (!response.ok) {
        let errorDetail = `HTTP error! status: ${response.status}`;
        try {
          const errorData = await response.json();
          errorDetail = errorData.detail || errorDetail;
        } catch (e) { /* Ignore */ }
        throw new Error(errorDetail);
      }

      const newSessionData = await response.json();
      createdSessionId = newSessionData.id;
      console.log("New session created:", createdSessionId);
      setSessionId(createdSessionId); // Store the new session ID

    } catch (error) {
      console.error("Failed to create session:", error);
      setMessages(prevMessages => [
        ...prevMessages,
        { role: 'assistant', content: `Error: Could not start a session. ${error.message}`, isError: true }
      ]);
      setSessionId(null); // Ensure session ID is null on error
    } finally {
      setIsSessionLoading(false); // Session loading finished
      // Keep isLoading true if called from handleSendMessage, let that handle it
      // If called from useEffect, set isLoading false here? Or just rely on message loading state?
      // Let's manage isLoading primarily around message sending for input disabling.
    }
    return createdSessionId; // Return ID (or null on failure)
  }, [sessionId]); // Dependency: re-create if sessionId changes (relevant for forceNew potentially)


  // --- Effect to create session on initial load ---
  useEffect(() => {
    console.log("App component mounted. Ensuring initial session exists.");
    // Use an IIAFE (Immediately Invoked Async Function Expression)
    // because useEffect callback itself cannot be async directly.

    setIsLoading(true); // Set loading true *before* calling ensureSession on initial load
    (async () => {
      await ensureSession(); // Initial session creation loading is handled within ensureSession
      setIsLoading(false); // Set loading false *after* initial ensureSession completes
    })();
  }, [ensureSession]); // Run only once on mount // Include ensureSession because it's defined outside but used inside


  // --- Handle Sending a Message ---
  const handleSendMessage = async () => {
    const messageText = inputValue.trim();
    if (!messageText || isLoading) return; // Prevent sending while loading

    const userMessage = { role: 'user', content: messageText };
    setMessages(prevMessages => [...prevMessages, userMessage]);
    setInputValue('');
    setIsLoading(true); // Start loading for message response

    let currentSessionId = sessionId;

    try {
        // Get session ID, creating if necessary (e.g., if initial load failed)
        if (!currentSessionId) {
            currentSessionId = await ensureSession(); // Try creating again
        }
        if (!currentSessionId) {
             console.error("Cannot send message without a valid session ID.");
             setIsLoading(false); // Stop loading if we can't proceed
             return;
        }

        console.log(`Sending message to session ${currentSessionId}:`, messageText);

        const response = await fetch(`${API_BASE_URL}/api/v1/sessions/${currentSessionId}/messages`, {
            method: 'POST',
            headers: {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            },
            body: JSON.stringify({ content: messageText, role: 'user' }),
        });

        if (!response.ok) {
            let errorDetail = `Error fetching response: ${response.status}`;
            try {
                const errorData = await response.json();
                errorDetail = errorData.detail || errorDetail;
            } catch (e) { /* Ignore */ }
            throw new Error(errorDetail);
        }

        const assistantMessageData = await response.json();
        console.log("Received assistant response:", assistantMessageData);
        setMessages(prevMessages => [...prevMessages, assistantMessageData]);

    } catch (error) {
      console.error("Error sending/receiving message:", error);
      setMessages(prevMessages => [
        ...prevMessages,
        { role: 'assistant', content: `Sorry, something went wrong: ${error.message}`, isError: true }
      ]);
    } finally {
      setIsLoading(false); // Stop loading indicator
    }
  };

  // --- Handler for New Chat Button ---
  const startNewChat = () => {
    console.log("Starting new chat...");
    setMessages([INITIAL_MESSAGE]); // Reset messages
    setInputValue(''); // Clear input
    setSessionId(null); // Clear current session ID
    setIsLoading(false); // Ensure loading is off
    setIsSessionLoading(false); // Ensure session loading is off
    // Let ensureSession handle creating the *next* session when the first message is sent
    // Or uncomment below to create immediately:
    // ensureSession(true); // Pass true to force creation
  };


  return (
    <div className={styles.appContainer}>
      <div className={styles.headerBar}>
         <h1 className={styles.appTitle}>RAG LLM Chat</h1>
         {/* Disable New Chat if session is creating OR message is sending */}
         <button onClick={startNewChat} className={styles.newChatButton} disabled={isLoading || isSessionLoading}>
             New Chat
         </button>
      </div>
      <ChatWindow messages={messages} />
      <MessageInput
        value={inputValue}
        onChange={(e) => setInputValue(e.target.value)}
        onSend={handleSendMessage}
        disabled={isLoading || isSessionLoading} // Disable if message OR session is loading
      />
       {/* Show appropriate loading text */}
       {isSessionLoading && <div className={styles.loadingIndicator}>Creating session...</div>}
       {isLoading && !isSessionLoading && <div className={styles.loadingIndicator}>Assistant is thinking...</div>}
    </div>
  );
}

export default App;

```

**2. Update `App.module.css`**

Add styles for the header bar and the new chat button.

```css
/* src/App.module.css */
.appContainer {
  display: flex;
  flex-direction: column;
  height: 100vh; /* Full viewport height */
  max-width: 800px; /* Max width for readability */
  margin: 0 auto; /* Center the container */
  border: 1px solid #ccc;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
  overflow: hidden; /* Prevent content spill */
  background-color: #f9f9f9;
}

/* --- NEW: Header Bar Styles --- */
.headerBar {
  display: flex;
  justify-content: space-between; /* Space title and button */
  align-items: center;
  padding: 0 15px; /* Adjust padding */
  background-color: #4a90e2; /* Example header color */
  color: white;
  flex-shrink: 0; /* Prevent header from shrinking */
}

.appTitle {
  /* Remove centering, padding, margin from original title style */
  padding: 15px 0;
  margin: 0;
  font-size: 1.5em;
  /* text-align: center; */ /* Removed */
}

.newChatButton {
  padding: 8px 12px;
  font-size: 0.9em;
  background-color: #f0f0f0;
  color: #333;
  border: 1px solid #ccc;
  border-radius: 5px;
  cursor: pointer;
  transition: background-color 0.2s ease;
}

.newChatButton:hover {
  background-color: #e0e0e0;
}

.newChatButton:disabled {
  background-color: #e0e0e0;
  color: #999;
  cursor: not-allowed;
}
/* --- END NEW --- */

.loadingIndicator {
    text-align: center;
    padding: 5px;
    font-style: italic;
    color: #888;
    background-color: #f0f0f0;
    border-top: 1px solid #ccc;
    flex-shrink: 0;
}

```

**Explanation of Changes:**

1.  **State:** Added `sessionId` (defaults to `null`) and `isSessionLoading`.
2.  **`ensureSession`:**
    *   Wrapped in `useCallback` for potential optimization (though not strictly necessary here).
    *   Accepts `forceNew` argument to allow explicitly creating a new session even if one exists.
    *   Sets `isSessionLoading` state during the API call.
    *   Returns the created session ID or `null` on error.
    *   Includes error handling that adds a message to the chat.
3.  **`useEffect` Hook:**
    *   Runs only once on component mount (`[]` dependency array).
    *   Calls `ensureSession()` to create the initial session automatically.
    *   Uses an IIAFE pattern to handle the `async` call within the synchronous `useEffect` callback.
    *   Includes `ensureSession` in the dependency array as per React hook linting rules (safe because `useCallback` stabilizes it).
4.  **`handleSendMessage`:**
    *   Checks if `sessionId` exists; if not, it calls `await ensureSession()` to try and create/get one *before* sending the message.
    *   Stops if no valid `sessionId` can be obtained.
    *   Uses the `currentSessionId` in the fetch URL.
5.  **`startNewChat` Function:**
    *   Resets the `messages` state to the initial greeting.
    *   Clears the `inputValue`.
    *   Sets `sessionId` back to `null`. This ensures that the *next* call to `handleSendMessage` will trigger `ensureSession` to create a brand new session ID.
    *   Resets loading flags.
6.  **UI:**
    *   Added a "New Chat" button in a new `headerBar` div.
    *   The button's `onClick` calls `startNewChat`.
    *   The button and input field are disabled using `isLoading || isSessionLoading`.
    *   The loading indicator text is slightly more specific.

**How to Test:**

1.  Restart the frontend dev server (`npm run dev`).
2.  Open the app.
    *   **Check Backend Logs:** You should immediately see a `POST /api/v1/sessions/` request as the `useEffect` runs `ensureSession`.
    *   **Check Frontend Console:** You should see logs "Ensuring session exists..." and "New session created: ...".
3.  Send a message. It should work using the automatically created session ID.
4.  Click the "New Chat" button.
    *   The message list should reset to the initial greeting.
    *   The input field should clear.
5.  Send another message.
    *   **Check Backend Logs:** You should see *another* `POST /api/v1/sessions/` request (a new session is created because `sessionId` was null).
    *   **Check Frontend Console:** Logs should indicate a new session was created.
    *   The message should be sent using the *new* session ID.

You now have basic session management integrated into the React app!
