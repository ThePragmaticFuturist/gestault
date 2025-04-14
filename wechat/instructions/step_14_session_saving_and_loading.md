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
We should keep the concept of needing a **Backend Session ID** for API calls separate from the **Frontend Active Session ID** which tracks the currently loaded chat in the UI.

```jsx
// src/App.jsx
import React, { useState, useEffect, useCallback, useRef } from 'react';
import ChatWindow from './components/ChatWindow/ChatWindow';
import MessageInput from './components/MessageInput/MessageInput';
import LLMManager from './components/LLMManager/LLMManager';
import SessionManager from './components/SessionManager/SessionManager';
import styles from './App.module.css';

const SESSION_INDEX_KEY = 'rag_chat_session_index';
const SESSION_PREFIX = 'rag_chat_session_';
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';
const INITIAL_MESSAGE = { role: 'assistant', content: 'Hello! How can I help you today?' };
const TYPING_INDICATOR_ID = 'typing-indicator-message';

console.log("Using API Base URL:", API_BASE_URL);

function App() {
  // --- Frontend State ---
  const [messages, setMessages] = useState([INITIAL_MESSAGE]);
  const [inputValue, setInputValue] = useState('');
  const [savedSessions, setSavedSessions] = useState([]); // Index from localStorage
  const [activeFrontendSessionId, setActiveFrontendSessionId] = useState(null); // ID of the session loaded from localStorage
  const [isChatDirty, setIsChatDirty] = useState(false);

  // --- Backend Interaction State ---
  const [backendSessionId, setBackendSessionId] = useState(null); // ID for API calls
  const [isLoading, setIsLoading] = useState(false); // For message send/response
  const [isSessionLoading, setIsSessionLoading] = useState(false); // For backend session *creation*

  // --- Error Message Helper ---
  const addErrorMessage = useCallback((content) => {
    setMessages(prevMessages => [...prevMessages, { role: 'assistant', content: `Error: ${content}`, isError: true }]);
    setIsChatDirty(true);
  }, []);

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


  // --- *** CORRECTED: Function to Ensure Backend Session Exists *** ---
  const ensureBackendSession = useCallback(async () => {
    // This function ONLY ensures we have a valid ID for API calls.
    // It doesn't necessarily mean starting a *new* chat flow in the UI.
    if (backendSessionId) {
      console.debug("Using existing backend session:", backendSessionId);
      return backendSessionId; // Return existing ID
    }

    console.log("No active backend session, creating a new one...");
    setIsSessionLoading(true); // Indicate loading *backend* session
    let createdBackendSessionId = null;

    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/sessions/`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Accept': 'application/json' },
        body: JSON.stringify({ name: "React UI Backend Session" }), // Generic name ok
      });

      if (!response.ok) {
        let errorDetail = `HTTP ${response.status}`;
        try { const errorData = await response.json(); errorDetail = errorData.detail || errorDetail; } catch (e) { /* Ignore */ }
        throw new Error(`Failed to create backend session. ${errorDetail}`);
      }

      const newSessionData = await response.json();
      createdBackendSessionId = newSessionData.id;
      console.log("New backend session created:", createdBackendSessionId);
      setBackendSessionId(createdBackendSessionId); // Set the state variable

    } catch (error) {
      console.error("Backend session creation error:", error);
      addErrorMessage(`Could not contact server to start session. ${error.message}`);
      setBackendSessionId(null); // Ensure null on error
    } finally {
      setIsSessionLoading(false); // Done trying to create backend session
    }
    return createdBackendSessionId; // Return new ID or null
  }, [backendSessionId, addErrorMessage]); // Depends on backendSessionId


  // --- Effect for initial backend session ---
  useEffect(() => {
    console.log("App component mounted. Ensuring initial backend session exists.");
     (async () => {
       await ensureBackendSession(); // Call the correctly named function
     })();
  }, [ensureBackendSession]); // Correct dependency


  // --- Handle Sending a Message ---
  const handleSendMessage = async () => {
    const messageText = inputValue.trim();
    if (!messageText || isLoading || isSessionLoading) return;

    const userMessage = { role: 'user', content: messageText };
    setMessages(prevMessages => [...prevMessages, userMessage]);
    setIsChatDirty(true);
    setInputValue('');
    setIsLoading(true); // Start message loading

    // --- Use the 'backendSessionId' state variable ---
    let currentBackendSessionId = backendSessionId;

    try {
        // Try to get/create a backend session ID *before* sending the message
        if (!currentBackendSessionId) {
            currentBackendSessionId = await ensureBackendSession();
        }
        // If still no backend ID after trying, we cannot proceed
        if (!currentBackendSessionId) {
             console.error("Cannot send message without a valid backend session ID.");
             // Error message was already added by ensureBackendSession if it failed
             setIsLoading(false); // Stop message loading
             return;
        }

        console.log(`Sending message to backend session ${currentBackendSessionId}:`, messageText);

        // Add Typing Indicator
        const typingIndicatorMessage = { id: TYPING_INDICATOR_ID, role: 'assistant', content: '...', isLoading: true };
        setMessages(prevMessages => prevMessages.concat([typingIndicatorMessage]));

        // --- Send message using currentBackendSessionId ---
        const response = await fetch(`${API_BASE_URL}/api/v1/sessions/${currentBackendSessionId}/messages`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', 'Accept': 'application/json' },
            body: JSON.stringify({ content: messageText, role: 'user' }),
        });

        // Remove typing indicator *before* processing response
        setMessages(prevMessages => prevMessages.filter(msg => msg.id !== TYPING_INDICATOR_ID));

        if (!response.ok) {
            let errorDetail = `Request failed with status ${response.status}`;
            try { const errorData = await response.json(); errorDetail = errorData.detail || errorDetail; } catch (e) { /* Ignore */ }
            throw new Error(errorDetail);
        }

        const assistantMessageData = await response.json();
        let isBackendError = assistantMessageData.content?.startsWith('[ERROR:') ?? false;

        // Add actual assistant message
        setMessages(prevMessages => [...prevMessages, { ...assistantMessageData, isError: isBackendError }]);
        setIsChatDirty(true); // Assistant response also makes it dirty

    } catch (error) {
      console.error("Error during message send/receive:", error);
      // Ensure typing indicator is removed on error too
      setMessages(prevMessages => prevMessages.filter(msg => msg.id !== TYPING_INDICATOR_ID));
      addErrorMessage(`Could not get response. ${error.message}`);
    } finally {
      setIsLoading(false); // Stop message loading indicator
    }
  };

  // --- Session Management Handlers (localStorage) ---

  const handleSaveSession = useCallback(async () => {
    if (!messages || messages.length <= 1) { alert("Nothing to save."); return; }

    const currentFrontendId = activeFrontendSessionId || `session_${Date.now()}`;
    const existingSessionMeta = savedSessions.find(s => s.id === currentFrontendId);
    const defaultName = `Session - ${new Date().toLocaleString()}`;
    const sessionName = prompt("Enter a name for this session:", existingSessionMeta?.name || defaultName);

    if (sessionName === null) return;

    setIsLoading(true); setErrorMessage(''); // Use general loading indicator

    try {
        let backendStatus = {};
        try { /* ... fetch LLM status ... */ } catch (statusError) { /* ... */ }

        const nowISO = new Date().toISOString();
        const sessionDataToSave = { // Data saved to localStorage
            id: currentFrontendId, // Frontend ID
            name: sessionName || defaultName,
            created_at: existingSessionMeta?.created_at || nowISO,
            last_updated_at: nowISO,
            backend_session_id: backendSessionId, // Store the *current* backend ID used by this chat
            llm_backend_type: backendStatus?.backend_type,
            llm_active_model: backendStatus?.active_model,
            llm_generation_config: backendStatus?.generation_config,
            messages: messages.filter(msg => msg.id !== TYPING_INDICATOR_ID), // Don't save typing indicator
        };

        localStorage.setItem(`${SESSION_PREFIX}${currentFrontendId}`, JSON.stringify(sessionDataToSave));

        updateSessionIndex({
            id: sessionDataToSave.id,
            name: sessionDataToSave.name,
            last_updated_at: sessionDataToSave.last_updated_at,
        });

        setActiveFrontendSessionId(currentFrontendId);
        setIsChatDirty(false);
        alert("Session saved successfully!");

    } catch (error) { /* ... error handling ... */ }
    finally { setIsLoading(false); }
  }, [activeFrontendSessionId, messages, savedSessions, updateSessionIndex, backendSessionId]); // Depend on backendSessionId


  const handleLoadSession = useCallback(async (frontendSessionIdToLoad) => { // Renamed param for clarity
    if (isChatDirty && !window.confirm("You have unsaved changes. Discard changes and load session?")) {
      return;
    }
    console.log("Loading frontend session:", frontendSessionIdToLoad);
    setIsLoading(true);

    try {
        const sessionJson = localStorage.getItem(`${SESSION_PREFIX}${frontendSessionIdToLoad}`);
        if (!sessionJson) throw new Error("Session data not found in storage.");
        const sessionData = JSON.parse(sessionJson);

        setMessages(sessionData.messages || [INITIAL_MESSAGE]);
        setActiveFrontendSessionId(sessionData.id);
        // --- CRITICAL: Restore or create NEW backend session ---
        // Option A: Try to reuse the saved backend ID (might be expired/invalid on server)
        // setBackendSessionId(sessionData.backend_session_id || null);
        // Option B (Safer): Always create a NEW backend session when loading an old chat
        console.log("Creating a new backend session for loaded chat history...");
        setBackendSessionId(null); // Clear old backend ID
        const newBackendId = await ensureBackendSession(); // Create a new one
        if (!newBackendId) {
            // Handle error if new backend session couldn't be made for the loaded chat
            throw new Error("Failed to create a new backend session for the loaded chat.");
        }
        console.log("Loaded frontend session will use new backend session:", newBackendId);
        // ----------------------------------------------------
        setInputValue('');
        setIsChatDirty(false); // Initial loaded state is clean

    } catch (error) { /* ... error handling ... */ }
    finally { setIsLoading(false); }
  }, [isChatDirty, addErrorMessage, ensureBackendSession]); // Added ensureBackendSession


  const handleDeleteSession = useCallback((frontendSessionIdToDelete) => { /* ... unchanged ... */ }, [savedSessions, activeFrontendSessionId, addErrorMessage]);


  // --- Corrected Handler for New Chat Button ---
  const startNewChat = useCallback((confirmIfDirty = true) => {
    if (confirmIfDirty && isChatDirty && !window.confirm(/* ... */)) return;

    console.log("Starting new chat...");
    setMessages([INITIAL_MESSAGE]);
    setInputValue('');
    setActiveFrontendSessionId(null); // Clear frontend active session ID
    setBackendSessionId(null);      // Clear backend session ID state
    setIsLoading(false);
    setIsSessionLoading(false);
    setIsChatDirty(false);
    // Force creation of a new backend session ID for the next message
    ensureBackendSession(); // No need to force, logic handles null state
  }, [isChatDirty, ensureBackendSession]); // Added ensureBackendSession

  return (
    <div className={styles.appContainer}>
      <SessionManager
        // Pass correct props
        savedSessions={savedSessions}
        activeSessionId={activeFrontendSessionId} // Pass frontend ID
        onLoadSession={handleLoadSession}
        onSaveSession={handleSaveSession}
        onDeleteSession={handleDeleteSession}
        onNewChat={startNewChat}
        disabled={isLoading || isSessionLoading}
      />
      <div className={styles.chatArea}>
        {/* ... ChatWindow, MessageInput */}
         <ChatWindow messages={messages} />
          <MessageInput
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onSend={handleSendMessage}
            disabled={isLoading || isSessionLoading} // Disable if message OR backend session is loading
          />
          {/* Loading Indicators */}
          {isSessionLoading && <div className={styles.loadingIndicator}>Connecting to backend session...</div>}
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
