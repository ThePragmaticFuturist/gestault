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

  const [currentGenConfig, setCurrentGenConfig] = useState({});

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
      const response = await fetch(`${API_BASE_URL}/api/v1/sessions`, {
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

    setIsLoading(true); // Use general loading indicator

    try {
        let backendStatus = {};
        try { 

          // Fetch current LLM status which includes the generation config
          let backendStatus = {};
          try {
              const statusResponse = await fetch(`${API_BASE_URL}/api/v1/models/status`);
              if(statusResponse.ok) {
                  backendStatus = await statusResponse.json();
                  // --- Store the fetched config ---
                  setCurrentGenConfig(backendStatus.generation_config || {}); // Update state too
                  // ------------------------------
              } else { /* ... warning ... */ }
          } catch (statusError) { 
            /* ... error handling ... */ 

          }

          const nowISO = new Date().toISOString();
          const sessionDataToSave = {
              id: currentFrontendId,
              name: sessionName || defaultName,
              created_at: existingSessionMeta?.created_at || nowISO,
              last_updated_at: nowISO,
              backend_session_id: backendSessionId,
              llm_backend_type: backendStatus?.backend_type,
              llm_active_model: backendStatus?.active_model,
              // --- SAVE the generation config used at save time ---
              llm_generation_config: backendStatus?.generation_config || currentGenConfig, // Use fetched or current state
              // ------------------------------------------------------
              messages: messages.filter(msg => msg.id !== TYPING_INDICATOR_ID),
          };

        } catch (statusError) {

         /* ... */ 

        }


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
        setInputValue('');
        setIsChatDirty(false); // Initial loaded state is clean

        console.log("Creating a new backend session for loaded chat history...");

        // --- Apply saved generation config ---
        const savedConfig = sessionData.llm_generation_config;
        if (savedConfig && Object.keys(savedConfig).length > 0) {
            console.log("Applying saved generation config:", savedConfig);
            try {
                // Send update request to backend
                const configResponse = await fetch(`${API_BASE_URL}/api/v1/models/config`, {
                    method: 'PUT',
                    headers: { 'Content-Type': 'application/json', 'Accept': 'application/json' },
                    body: JSON.stringify(savedConfig), // Send the saved config object
                });
                if (!configResponse.ok) {
                     let errorDetail = `HTTP ${configResponse.status}`;
                     try { const errorData = await configResponse.json(); errorDetail = errorData.detail || errorDetail; } catch(e){}
                     throw new Error(`Failed to apply saved config: ${errorDetail}`);
                }
                const appliedConfig = await configResponse.json();
                setCurrentGenConfig(appliedConfig); // Update state with applied config
                console.log("Successfully applied saved config to backend.");
            } catch (configError) {
                console.error("Error applying saved generation config:", configError);
                addErrorMessage(`Could not apply saved generation settings: ${configError.message}`);
                // Optionally fetch current config instead if applying failed
                fetchCurrentGenConfig(); // Fetch current settings from backend
            }
        } else {
            // If no config saved with session, fetch current settings from backend
            fetchCurrentGenConfig();
        }

        setBackendSessionId(null); // Clear old backend ID
        const newBackendId = await ensureBackendSession(); // Create a new one
        if (!newBackendId) {
            // Handle error if new backend session couldn't be made for the loaded chat
            throw new Error("Failed to create a new backend session for the loaded chat.");
        }
        console.log("Loaded frontend session will use new backend session:", newBackendId);

    } catch (error) { 
      /* ... error handling ... */ 
    }

    finally { setIsLoading(false); }
  }, [isChatDirty, addErrorMessage, ensureBackendSession]); // Added ensureBackendSession


  // --- Function to fetch current generation config ---
  const fetchCurrentGenConfig = useCallback(async () => {
       console.log("Fetching current generation config from backend...");
       try {
          const response = await fetch(`${API_BASE_URL}/api/v1/models/config`);
          // Handle case where no model is loaded (404)
          if (response.status === 404) {
               console.log("No model loaded, cannot fetch config.");
               setCurrentGenConfig({}); // Reset local config state
               return;
          }
          if (!response.ok) throw new Error(`HTTP error ${response.status}`);
          const data = await response.json();
          setCurrentGenConfig(data || {});
          console.log("Fetched current config:", data);
       } catch (error) {
           console.error("Failed to fetch current config:", error);
           addErrorMessage(`Could not fetch current LLM settings: ${error.message}`);
           setCurrentGenConfig({}); // Reset on error
       }
  }, [addErrorMessage]);

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

  const handleDeleteSession = useCallback((frontendSessionIdToDelete) => {

      const sessionToDelete = savedSessions.find(s => s.id === frontendSessionIdToDelete);
      if (!sessionToDelete) {
          console.error("Attempted to delete non-existent session ID:", frontendSessionIdToDelete);
          return; // Session not found in index
      }

      // Confirm with the user
      if (!window.confirm(`Are you sure you want to delete session "${sessionToDelete.name || 'Untitled Session'}"? This cannot be undone.`)) {
        return; // User cancelled deletion
      }

      console.log("Deleting frontend session:", frontendSessionIdToDelete);
      try {
          // 1. Remove session data from localStorage
          localStorage.removeItem(`${SESSION_PREFIX}${frontendSessionIdToDelete}`);
          console.log("Removed session data from localStorage for:", frontendSessionIdToDelete);

          // 2. Remove from index in state and update localStorage index
          let updatedIndex = []; // Define outside to check scope
          setSavedSessions(prevIndex => {
              updatedIndex = prevIndex.filter(s => s.id !== frontendSessionIdToDelete);
              try {
                   localStorage.setItem(SESSION_INDEX_KEY, JSON.stringify(updatedIndex));
                   console.log("Updated session index in localStorage.");
              } catch (error) {
                   console.error("Failed to update session index in localStorage:", error);
                   // Non-fatal, but log it
              }
              return updatedIndex; // Update state
          });

          // 3. If the deleted session was the active one, start a new chat UI state
          if (activeFrontendSessionId === frontendSessionIdToDelete) {
              console.log("Active session deleted, starting new chat UI state.");
              // Call startNewChat, passing false to skip the dirty check confirmation
              startNewChat(false);
          }
          console.log("Session deleted successfully.");

      } catch (error) {
          console.error("Failed to delete session:", error);
          // Show error in the potentially new chat UI if reset occurred
          addErrorMessage(`Could not delete session. ${error.message}`);
      }
    }, [savedSessions, activeFrontendSessionId, addErrorMessage, startNewChat]);

  // --- Effect for initial backend session ---
  useEffect(() => {
    console.log("App component mounted. Ensuring initial backend session exists.");
     (async () => {
       await ensureBackendSession(); // Call the correctly named function
       fetchCurrentGenConfig();
     })();
  }, [ensureBackendSession, fetchCurrentGenConfig]); // Correct dependency


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