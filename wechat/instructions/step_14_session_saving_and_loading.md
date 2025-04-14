# Step 14: Save, Load, and Manage Chat Sessions Locally #

Okay, let's add the functionality to save, load, and manage chat sessions using the browser's `localStorage`. This will involve:

1.  Adding state in `App.jsx` to track saved sessions and whether the current chat has unsaved changes.
2.  Creating functions to interact with `localStorage` (save, load, delete, list).
3.  Creating a new `SessionManager` component to display saved sessions and provide buttons for actions.
4.  Integrating the `SessionManager` into `App.jsx`.

**1. Define Storage Keys and Helper Functions (Optional but Recommended)**

It's good practice to centralize the keys used for `localStorage`. You could create a small utility file or define constants in `App.jsx`.

```javascript
// src/utils/localStorageKeys.js (Optional file)
export const SESSION_INDEX_KEY = 'rag_chat_session_index';
export const SESSION_PREFIX = 'rag_chat_session_';
```

**2. Update `App.jsx` State and Add Storage Logic**

We need state for the list of saved sessions and a 'dirty' flag for unsaved changes.

```jsx
// src/App.jsx
import React, { useState, useEffect, useCallback, useRef } from 'react'; // Added useRef
import ChatWindow from './components/ChatWindow/ChatWindow';
import MessageInput from './components/MessageInput/MessageInput';
import LLMManager from './components/LLMManager/LLMManager';
import SessionManager from './components/SessionManager/SessionManager'; // <-- Import new component
import styles from './App.module.css';

// Use constants for keys
const SESSION_INDEX_KEY = 'rag_chat_session_index';
const SESSION_PREFIX = 'rag_chat_session_';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';
console.log("Using API Base URL:", API_BASE_URL);

const INITIAL_MESSAGE = { role: 'assistant', content: 'Hello! How can I help you today?' };
const TYPING_INDICATOR_ID = 'typing-indicator-message';

function App() {
  const [messages, setMessages] = useState([INITIAL_MESSAGE]);
  const [inputValue, setInputValue] = useState('');
  const [sessionId, setSessionId] = useState(null); // Backend session ID for API calls
  const [isLoading, setIsLoading] = useState(false);
  const [isSessionLoading, setIsSessionLoading] = useState(false);

  // --- NEW State for Frontend Session Management ---
  const [savedSessions, setSavedSessions] = useState([]); // List of {id, name, last_updated_at} from index
  const [activeSessionId, setActiveSessionId] = useState(null); // Frontend ID of the *loaded* chat session
  const [isChatDirty, setIsChatDirty] = useState(false); // Track unsaved changes
  const sessionNameInputRef = useRef(null); // Ref for potential inline name editing
  // -------------------------------------------------

  // --- Load Saved Sessions Index on Mount ---
  const loadSessionIndex = useCallback(() => {
    try {
      const indexJson = localStorage.getItem(SESSION_INDEX_KEY);
      const index = indexJson ? JSON.parse(indexJson) : [];
      // Sort by date descending
      index.sort((a, b) => new Date(b.last_updated_at) - new Date(a.last_updated_at));
      setSavedSessions(index);
      console.log("Loaded session index:", index);
    } catch (error) {
      console.error("Failed to load or parse session index:", error);
      setSavedSessions([]); // Reset on error
      localStorage.removeItem(SESSION_INDEX_KEY); // Clear potentially corrupted index
    }
  }, []);

  useEffect(() => {
    loadSessionIndex();
  }, [loadSessionIndex]); // Load index once on mount

  // --- Function to update index in localStorage and state ---
  const updateSessionIndex = useCallback((newSessionMetaData) => {
    setSavedSessions(prevIndex => {
      const existingIndex = prevIndex.findIndex(s => s.id === newSessionMetaData.id);
      let updatedIndex;
      if (existingIndex > -1) {
        // Update existing entry
        updatedIndex = [...prevIndex];
        updatedIndex[existingIndex] = newSessionMetaData;
      } else {
        // Add new entry
        updatedIndex = [...prevIndex, newSessionMetaData];
      }
      // Sort and save
      updatedIndex.sort((a, b) => new Date(b.last_updated_at) - new Date(a.last_updated_at));
      try {
        localStorage.setItem(SESSION_INDEX_KEY, JSON.stringify(updatedIndex));
      } catch (error) {
        console.error("Failed to save session index to localStorage:", error);
        // Optionally notify user about storage error
      }
      return updatedIndex;
    });
  }, []);

  // --- Add Error Message Helper (unchanged) ---
  const addErrorMessage = useCallback((content) => {
    setMessages(prevMessages => [...prevMessages, { role: 'assistant', content: `Error: ${content}`, isError: true }]);
    setIsChatDirty(true); // Adding error message makes chat dirty
  }, []);

  // --- Backend Session Creation (unchanged, returns backend ID) ---
  const ensureBackendSession = useCallback(async (forceNew = false) => {
    // ... (This function remains the same as before, responsible ONLY for the backend session ID)
    // ... (It sets the 'sessionId' state variable)
        if (!forceNew && sessionId) {
      return sessionId;
    }
    if(forceNew) {
        setSessionId(null);
    }

    setIsSessionLoading(true);
    let createdSessionId = null;

    try {
      console.log(forceNew ? "Forcing new backend session..." : "Creating initial backend session...");
      const response = await fetch(`${API_BASE_URL}/api/v1/sessions/`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Accept': 'application/json' },
        body: JSON.stringify({ name: "React UI Backend Session" }), // Generic name for backend session
      });

      if (!response.ok) {
        let errorDetail = `HTTP ${response.status}`;
        try { const errorData = await response.json(); errorDetail = errorData.detail || errorDetail; } catch (e) { /* Ignore */ }
        throw new Error(`Failed to create backend session. ${errorDetail}`);
      }

      const newSessionData = await response.json();
      createdSessionId = newSessionData.id;
      console.log("New backend session created/ensured:", createdSessionId);
      setSessionId(createdSessionId); // Set backend session ID state

    } catch (error) {
      console.error("Backend session creation error:", error);
      addErrorMessage(`Could not start backend session. ${error.message}`);
      setSessionId(null);
    } finally {
      setIsSessionLoading(false);
    }
    return createdSessionId;
  }, [sessionId, addErrorMessage]);

  // --- Effect for initial backend session (unchanged) ---
  useEffect(() => {
    console.log("App component mounted. Ensuring initial backend session exists.");
     (async () => {
       await ensureBackendSession();
     })();
  }, [ensureBackendSession]);


  // --- Handle Sending a Message (Modified to set dirty flag) ---
  const handleSendMessage = async () => {
    const messageText = inputValue.trim();
    if (!messageText || isLoading || isSessionLoading) return;

    const userMessage = { role: 'user', content: messageText };
    // --- Add user message AND set chat dirty ---
    setMessages(prevMessages => [...prevMessages, userMessage]);
    setIsChatDirty(true); // Mark chat as dirty
    // --- End Add ---

    setInputValue('');
    setIsLoading(true);

    let currentBackendSessionId = sessionId;

    try {
        if (!currentBackendSessionId) {
            currentBackendSessionId = await ensureBackendSession();
        }
        if (!currentBackendSessionId) {
             setIsLoading(false);
             return;
        }

        // ... (fetch call to send message - unchanged) ...
        const response = await fetch(`${API_BASE_URL}/api/v1/sessions/${currentBackendSessionId}/messages`, { /* ... */ });
        if (!response.ok) { throw new Error(/* ... */); }
        const assistantMessageData = await response.json();

        let isBackendError = false;
        if (assistantMessageData.content && typeof assistantMessageData.content === 'string' && assistantMessageData.content.startsWith('[ERROR:')) {
            isBackendError = true;
        }

        // --- Add assistant message AND set chat dirty ---
        setMessages(prevMessages => [...prevMessages, { ...assistantMessageData, isError: isBackendError }]);
        setIsChatDirty(true); // Mark chat as dirty
        // --- End Add ---

    } catch (error) {
      console.error("Error during message send/receive:", error);
      // addErrorMessage already sets isChatDirty
      addErrorMessage(`Could not get response. ${error.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  // --- NEW Session Management Handlers ---

  const handleSaveSession = useCallback(async () => {
    if (!messages || messages.length <= 1) { // Don't save empty/initial chats
        alert("Nothing to save.");
        return;
    }

    // 1. Determine ID and Name
    const currentFrontendId = activeSessionId || `session_${Date.now()}`; // Generate ID if new
    const existingSession = savedSessions.find(s => s.id === currentFrontendId);
    const defaultName = `Session - ${new Date().toLocaleString()}`;
    const sessionName = prompt("Enter a name for this session:", existingSession?.name || defaultName);

    if (sessionName === null) return; // User cancelled prompt

    setIsLoading(true); // Indicate saving activity
    setErrorMessage('');

    try {
        // 2. Fetch current LLM status from backend
        let backendStatus = {};
        try {
            const statusResponse = await fetch(`${API_BASE_URL}/api/v1/models/status`);
            if(statusResponse.ok) {
                backendStatus = await statusResponse.json();
            } else {
                console.warn("Could not fetch LLM status for saving.");
            }
        } catch (statusError) {
            console.error("Error fetching LLM status:", statusError);
        }

        // 3. Prepare session data object
        const nowISO = new Date().toISOString();
        const sessionData = {
            id: currentFrontendId,
            name: sessionName || defaultName, // Use default if prompt returns empty
            created_at: existingSession?.created_at || nowISO, // Keep original creation date if exists
            last_updated_at: nowISO,
            backend_session_id: sessionId, // Store the backend API session ID too
            llm_backend_type: backendStatus?.backend_type,
            llm_active_model: backendStatus?.active_model,
            llm_generation_config: backendStatus?.generation_config,
            messages: messages, // Store the whole message array
        };

        // 4. Save to localStorage
        localStorage.setItem(`${SESSION_PREFIX}${currentFrontendId}`, JSON.stringify(sessionData));

        // 5. Update the index
        updateSessionIndex({
            id: sessionData.id,
            name: sessionData.name,
            last_updated_at: sessionData.last_updated_at,
        });

        // 6. Update state
        setActiveSessionId(currentFrontendId); // Ensure the active ID is set
        setIsChatDirty(false); // Mark as saved
        alert("Session saved successfully!");

    } catch (error) {
        console.error("Failed to save session:", error);
        addErrorMessage(`Could not save session. ${error.message}`);
    } finally {
        setIsLoading(false);
    }
  }, [activeSessionId, messages, savedSessions, updateSessionIndex, sessionId]); // Include sessionId

  const handleLoadSession = useCallback((sessionIdToLoad) => {
    if (isChatDirty && !window.confirm("You have unsaved changes. Discard changes and load session?")) {
      return; // User cancelled loading
    }

    console.log("Loading session:", sessionIdToLoad);
    setIsLoading(true); // Indicate loading
    try {
        const sessionJson = localStorage.getItem(`${SESSION_PREFIX}${sessionIdToLoad}`);
        if (!sessionJson) {
            throw new Error("Session data not found in storage.");
        }
        const sessionData = JSON.parse(sessionJson);

        // Restore state
        setMessages(sessionData.messages || [INITIAL_MESSAGE]);
        setActiveSessionId(sessionData.id);
        setSessionId(sessionData.backend_session_id || null); // Restore backend session ID
        setInputValue('');
        setIsChatDirty(false); // Loaded state is not dirty initially
        console.log("Session loaded successfully. Backend session ID:", sessionData.backend_session_id);
        // Optional: Display loaded model info? Add to UI later.
        // Optional: Try to set the backend model? Complex, skip for now.

    } catch (error) {
        console.error("Failed to load session:", error);
        addErrorMessage(`Could not load session. ${error.message}`);
        // Reset to a clean state?
        startNewChat(); // Go to a new chat state on load failure
    } finally {
      setIsLoading(false);
    }
  }, [isChatDirty, addErrorMessage]); // Add addErrorMessage dependency

  const handleDeleteSession = useCallback((sessionIdToDelete) => {
    const sessionToDelete = savedSessions.find(s => s.id === sessionIdToDelete);
    if (!sessionToDelete) return;

    if (!window.confirm(`Are you sure you want to delete session "${sessionToDelete.name}"?`)) {
      return; // User cancelled deletion
    }

    console.log("Deleting session:", sessionIdToDelete);
    try {
        // Remove session data
        localStorage.removeItem(`${SESSION_PREFIX}${sessionIdToDelete}`);

        // Remove from index and update state/storage
        setSavedSessions(prevIndex => {
            const updatedIndex = prevIndex.filter(s => s.id !== sessionIdToDelete);
            localStorage.setItem(SESSION_INDEX_KEY, JSON.stringify(updatedIndex));
            return updatedIndex;
        });

        // If the deleted session was the active one, start a new chat
        if (activeSessionId === sessionIdToDelete) {
            startNewChat(false); // Pass false to avoid confirm prompt again
        }
        console.log("Session deleted successfully.");

    } catch (error) {
        console.error("Failed to delete session:", error);
        addErrorMessage(`Could not delete session. ${error.message}`); // Show error in potentially new chat
    }
  }, [savedSessions, activeSessionId, addErrorMessage]); // Add addErrorMessage

  // --- Modified Handler for New Chat Button ---
  // Accepts optional flag to skip confirm prompt (used after delete)
  const startNewChat = useCallback((confirmIfDirty = true) => {
    if (confirmIfDirty && isChatDirty && !window.confirm("You have unsaved changes. Discard changes and start a new chat?")) {
        return; // User cancelled
    }
    console.log("Starting new chat...");
    setMessages([INITIAL_MESSAGE]);
    setInputValue('');
    setSessionId(null); // Clear backend session ID
    setActiveSessionId(null); // Clear frontend active session ID
    setIsLoading(false);
    setIsSessionLoading(false);
    setIsChatDirty(false); // New chat is not dirty
    // Ensure backend session is created for the next message
    ensureBackendSession(true); // Force creation of a new backend session
  }, [isChatDirty, ensureBackendSession]); // Add dependencies


  return (
    <div className={styles.appContainer}>
      {/* --- Pass Session Props --- */}
      <SessionManager
        savedSessions={savedSessions}
        activeSessionId={activeSessionId}
        onLoadSession={handleLoadSession}
        onSaveSession={handleSaveSession}
        onDeleteSession={handleDeleteSession}
        onNewChat={startNewChat} // Pass new chat handler
        disabled={isLoading || isSessionLoading} // Disable controls while busy
      />
      {/* ------------------------- */}

      <div className={styles.chatArea}>
        <ChatWindow messages={messages} />
        <MessageInput
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          onSend={handleSendMessage}
          disabled={isLoading || isSessionLoading}
        />
        {/* Loading Indicators */}
        {isSessionLoading && <div className={styles.loadingIndicator}>Working on session...</div>}
        {isLoading && !isSessionLoading && <div className={styles.loadingIndicator}>Assistant is thinking...</div>}
      </div>
    </div>
  );
}

export default App;

```

**3. Create `SessionManager.jsx`**

This component displays the list and buttons.

```jsx
// src/components/SessionManager/SessionManager.jsx
import React from 'react';
import styles from './SessionManager.module.css';

function SessionManager({
    savedSessions,
    activeSessionId,
    onLoadSession,
    onSaveSession,
    onDeleteSession,
    onNewChat, // Receive new chat handler
    disabled
}) {

    return (
        <div className={styles.sessionManager}>
            <div className={styles.controls}>
                <button onClick={onNewChat} disabled={disabled} title="Start a new empty chat">
                    New Chat
                </button>
                <button onClick={onSaveSession} disabled={disabled} title="Save the current chat session">
                    Save Current
                </button>
            </div>
            <div className={styles.sessionListContainer}>
                <h4>Saved Sessions</h4>
                {savedSessions.length === 0 ? (
                    <p className={styles.noSessions}>No saved sessions yet.</p>
                ) : (
                    <ul className={styles.sessionList}>
                        {savedSessions.map(session => (
                            <li
                                key={session.id}
                                className={`${styles.sessionItem} ${session.id === activeSessionId ? styles.active : ''}`}
                            >
                                <span
                                    className={styles.sessionName}
                                    onClick={() => !disabled && onLoadSession(session.id)} // Load on click
                                    title={`Load "${session.name}" (Last saved: ${new Date(session.last_updated_at).toLocaleString()})`}
                                >
                                    {session.name}
                                </span>
                                <button
                                    className={styles.deleteButton}
                                    onClick={() => !disabled && onDeleteSession(session.id)}
                                    disabled={disabled}
                                    title="Delete session"
                                >
                                    &times; {/* Simple 'x' delete symbol */}
                                </button>
                            </li>
                        ))}
                    </ul>
                )}
            </div>
        </div>
    );
}

export default SessionManager;
```

**4. Create `SessionManager.module.css`**

Add styles for the session list and controls.

```css
/* src/components/SessionManager/SessionManager.module.css */
.sessionManager {
  background-color: #e8f0fe; /* Lighter blue background */
  padding: 10px 15px;
  border-bottom: 1px solid #c9d7ea;
  display: flex;
  flex-direction: column;
  max-height: 35vh; /* Limit height */
  overflow: hidden; /* Hide overflow initially, list scrolls */
  flex-shrink: 0;
}

.controls {
  display: flex;
  gap: 10px;
  margin-bottom: 10px;
  flex-shrink: 0;
}

.controls button {
  padding: 6px 12px;
  border: 1px solid #a0b3cd;
  background-color: #ffffff;
  color: #1c5fb0;
  border-radius: 5px;
  cursor: pointer;
  font-size: 0.9em;
  transition: background-color 0.2s ease, border-color 0.2s ease;
}

.controls button:hover {
  background-color: #f0f4f8;
  border-color: #1c5fb0;
}

.controls button:disabled {
  background-color: #e8f0fe;
  color: #a0b3cd;
  border-color: #d8e2f1;
  cursor: not-allowed;
}

.sessionListContainer {
    flex-grow: 1; /* Allow list to take remaining space */
    overflow-y: auto; /* Enable scrolling for the list */
    min-height: 0; /* Necessary for scrolling in flex column */
}

.sessionListContainer h4 {
  margin-top: 0;
  margin-bottom: 8px;
  color: #1c5fb0;
  font-size: 1em;
  border-bottom: 1px solid #c9d7ea;
  padding-bottom: 4px;
}

.sessionList {
  list-style: none;
  padding: 0;
  margin: 0;
}

.sessionItem {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 6px 4px;
  border-bottom: 1px dashed #d8e2f1;
  cursor: pointer;
}
.sessionItem:last-child {
    border-bottom: none;
}

.sessionItem:hover {
    background-color: #d8e2f1;
    border-radius: 3px;
}

.sessionItem.active {
    background-color: #c9d7ea;
    font-weight: bold;
    border-radius: 3px;
}

.sessionName {
  flex-grow: 1;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  padding-right: 10px;
  font-size: 0.9em;
}

.deleteButton {
  background: none;
  border: none;
  color: #cf222e;
  font-size: 1.2em;
  line-height: 1;
  cursor: pointer;
  padding: 0 5px;
  flex-shrink: 0;
}

.deleteButton:hover {
  color: #a3131e;
}

.deleteButton:disabled {
    color: #ccc;
    cursor: not-allowed;
}

.noSessions {
    font-style: italic;
    color: #57606a;
    font-size: 0.9em;
}
```

**5. Update `App.module.css` Layout (Minor Adjustments)**

Ensure the layout works with the new manager component. The previous flexbox column layout should still work well. You might adjust `max-height` on `SessionManager` or overall layout as needed.

```css
/* src/App.module.css */
.appContainer {
  display: flex;
  flex-direction: column; /* Stack Manager and Chat vertically */
  height: 100vh;
  max-height: 98vh; /* Limit overall height slightly */
  width: 100%;
  max-width: 1000px;
  border: 1px solid #ccc;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
  overflow: hidden;
  background-color: #f9f9f9;
  border-radius: 8px;
}

.chatArea {
    flex-grow: 1; /* Make chat area take remaining vertical space */
    display: flex;
    flex-direction: column;
    min-height: 0; /* Important for flex-grow in column layout */
    background-color: #ffffff;
}

/* ... (loading indicator styles) ... */
```

**Explanation of Changes:**

1.  **State:** Added `savedSessions` (array from index), `activeSessionId` (ID of loaded session), `isChatDirty`.
2.  **Loading Index:** `useEffect` and `loadSessionIndex` load the list of saved sessions on mount.
3.  **`updateSessionIndex`:** Helper to update the `savedSessions` state and save the index to `localStorage`.
4.  **Backend Session:** `ensureBackendSession` now only manages the `sessionId` used for API calls. The concept of the *frontend's* currently active/displayed session is separate (`activeSessionId`).
5.  **Dirty Tracking:** `setIsChatDirty(true)` is called when messages are added (`handleSendMessage`, `addErrorMessage`). It's reset on save, load, and new chat.
6.  **`handleSaveSession`:** Prompts for name, fetches LLM status, constructs session data, saves to `localStorage` (using `SESSION_PREFIX + id`), updates the index, and resets dirty flag.
7.  **`handleLoadSession`:** Checks dirty flag, prompts user if needed, reads from `localStorage`, updates `messages`, `activeSessionId`, *and* `sessionId` (backend ID), resets dirty flag.
8.  **`handleDeleteSession`:** Confirms deletion, removes item and updates index in `localStorage`, updates state, and starts a new chat if the active one was deleted.
9.  **`startNewChat`:** Checks dirty flag, resets state variables (`messages`, `sessionId`, `activeSessionId`, `isChatDirty`, loading flags), forces creation of a *new backend session* for the next message.
10. **`SessionManager` Component:** Created to encapsulate the UI for listing, loading, saving, deleting, and starting new chats. It receives necessary state and handler props from `App`.
11. **Integration:** `SessionManager` is rendered within `App`, and the main chat area is grouped in `chatArea`.

**How to Test:**

1.  Restart frontend server, hard refresh browser.
2.  Chat for a bit.
3.  Click "Save Current". Enter a name when prompted. Check `localStorage` in browser dev tools (Application -> Local Storage) - you should see an index key and a session key. The "Saved Sessions" list should update.
4.  Click "New Chat". The chat area should reset.
5.  Send a message (this creates a new backend session).
6.  Click "Load Session" on the session you saved earlier in the list. Confirm if prompted about unsaved changes. The previous chat should load.
7.  Load the session again, make a change (send a message), then try to load it *again* or click "New Chat" - verify the "unsaved changes" prompt appears.
8.  Delete a session using the 'x' button. Verify it disappears from the list and storage. If it was the active one, verify a new chat starts.

This provides the core functionality for managing distinct chat sessions within the browser.
