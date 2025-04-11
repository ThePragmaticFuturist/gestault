# Step 13: Managing the LLM Connection #

Okay, let's add a dedicated component for managing the LLM connection. This will allow users to select between local/cached models, input a new model ID for local download, or configure an API backend connection.

**1. Create `LLMManager` Component Files**

1.  Inside `src/components/`, create a new folder: `LLMManager`.
2.  Inside `src/components/LLMManager/`, create two files:
    *   `LLMManager.jsx`
    *   `LLMManager.module.css`

**2. Implement `LLMManager.jsx`**

This component will fetch available models and status, manage UI state for selection, and handle the API call to load/configure the backend.

```jsx
// src/components/LLMManager/LLMManager.jsx
import React, { useState, useEffect, useCallback } from 'react';
import styles from './LLMManager.module.css';

// Get API base URL from environment variables
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

// Define Backend Types explicitly matching our Literal types
const BACKEND_TYPES = ["local", "ollama", "vllm", "gemini", "instructlab"];

function LLMManager() {
  // --- State for fetched data ---
  const [availableModels, setAvailableModels] = useState({ local_model_paths: [], cached_hub_models: [] });
  const [currentStatus, setCurrentStatus] = useState({ status: 'Unknown', active_model: null, backend_type: null, last_error: null });

  // --- State for UI selections ---
  const [selectedBackendType, setSelectedBackendType] = useState(''); // Default to empty, fetch status first
  const [selectedExistingModel, setSelectedExistingModel] = useState(''); // For dropdowns
  const [inputModelName, setInputModelName] = useState(''); // For text inputs

  // --- State for UI feedback ---
  const [isLoadingStatus, setIsLoadingStatus] = useState(false);
  const [isLoadingAvailable, setIsLoadingAvailable] = useState(false);
  const [isSubmittingLoad, setIsSubmittingLoad] = useState(false);
  const [loadMessage, setLoadMessage] = useState(''); // Feedback after load attempt
  const [errorMessage, setErrorMessage] = useState(''); // Errors during fetch/load

  // --- Fetch Available Models ---
  const fetchAvailableModels = useCallback(async () => {
    setIsLoadingAvailable(true);
    setErrorMessage('');
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/models/available`);
      if (!response.ok) throw new Error(`HTTP error ${response.status}`);
      const data = await response.json();
      setAvailableModels(data || { local_model_paths: [], cached_hub_models: [] });
    } catch (error) {
      console.error("Failed to fetch available models:", error);
      setErrorMessage(`Failed to fetch available models: ${error.message}`);
      setAvailableModels({ local_model_paths: [], cached_hub_models: [] }); // Reset on error
    } finally {
      setIsLoadingAvailable(false);
    }
  }, []);

  // --- Fetch Current LLM Status ---
  const fetchStatus = useCallback(async () => {
    setIsLoadingStatus(true);
    setErrorMessage(''); // Clear previous errors on status fetch
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/models/status`);
      if (!response.ok) throw new Error(`HTTP error ${response.status}`);
      const data = await response.json();
      setCurrentStatus(data || { status: 'Error Fetching', active_model: null });

      // --- Pre-fill UI based on current status ---
      if (data && data.status !== 'Inactive') {
        setSelectedBackendType(data.backend_type || ''); // Set selected type
        // If it's an API backend or an existing local model, prefill the input/selection
        if (data.backend_type !== 'local') {
            setInputModelName(data.active_model || '');
            setSelectedExistingModel(''); // Clear dropdown selection
        } else if (data.active_model) {
            // Check if active model is in local paths or hub cache
            const localMatch = availableModels.local_model_paths.find(p => p.endsWith(data.active_model)) || data.active_model; // Crude match for paths
            const hubMatch = availableModels.cached_hub_models.includes(data.active_model);

            if (availableModels.local_model_paths.includes(localMatch) || hubMatch) {
                 setSelectedExistingModel(data.active_model); // Select from dropdown
                 setInputModelName(''); // Clear text input
            } else {
                 // Model active but not in lists? Might be newly entered one
                 setInputModelName(data.active_model);
                 setSelectedExistingModel('');
            }
        }

        setLoadMessage('');
      } else {
         // Default to 'local' in UI if inactive or fetched type is invalid
         setSelectedBackendType(BACKEND_TYPES.includes(data?.backend_type) ? data.backend_type : 'local');
         setInputModelName('');
         setSelectedExistingModel('');
      }

    } catch (error) {
      console.error("Failed to fetch LLM status:", error);
      setErrorMessage(`Failed to fetch status: ${error.message}`);
      setCurrentStatus({ status: 'Error Fetching', active_model: null, backend_type: null, last_error: error.message });
      setSelectedBackendType('local'); // Default UI on error
    } finally {
      setIsLoadingStatus(false);
    }
  }, [availableModels]); // Re-run if availableModels changes (for pre-filling)


  // --- Initial data fetching on component mount ---
  useEffect(() => {
    fetchAvailableModels();
    fetchStatus();
    // Set interval to refresh status periodically (optional)
    const intervalId = setInterval(fetchStatus, 30000); // Refresh status every 30 seconds
    return () => clearInterval(intervalId); // Cleanup interval on unmount
  }, [fetchAvailableModels]); // fetchStatus is stable due to useCallback dependencies


   // --- Handle Model Load/Set Request ---
   const handleLoadModel = async (event) => {
    event.preventDefault(); // Prevent default form submission if wrapped in form
    setIsSubmittingLoad(true);
    setLoadMessage('');
    setErrorMessage('');

    let modelIdentifier = '';
    // Determine the model identifier based on backend type and UI selections
    if (selectedBackendType === 'local') {
        // Prioritize text input for new model, then dropdown for existing
        modelIdentifier = inputModelName.trim() || selectedExistingModel;
    } else {
        // For API backends, use the text input
        modelIdentifier = inputModelName.trim();
    }

    if (!modelIdentifier) {
        setErrorMessage("Please select or enter a model identifier.");
        setIsSubmittingLoad(false);
        return;
    }

    // Prepare request body - only include type if overriding server default? No, send determined type.
    const requestBody = {
        backend_type: selectedBackendType, // Send the type selected in the UI
        model_name_or_path: modelIdentifier,
        // Local-specific params could be added here if UI controls existed
        // device: null,
        // quantization: null,
    };

    console.log("Sending load request:", requestBody);

    try {
        const response = await fetch(`${API_BASE_URL}/api/v1/models/load`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', 'Accept': 'application/json' },
            body: JSON.stringify(requestBody),
        });

        const resultData = await response.json(); // Read body even on error if possible

        if (!response.ok) {
            throw new Error(resultData.detail || `HTTP error ${response.status}`);
        }

        setLoadMessage(resultData.message || "Request accepted.");
        // Refresh status after a short delay to allow backend processing (especially for local)
        setTimeout(fetchStatus, 2000); // Refresh status after 2 seconds

    } catch (error) {
        console.error("Failed to load/set model:", error);
        setErrorMessage(`Load request failed: ${error.message}`);
    } finally {
        setIsSubmittingLoad(false);
    }
   };

   // --- Handle selection changes ---
   const handleBackendTypeChange = (event) => {
       setSelectedBackendType(event.target.value);
       // Reset model selections when type changes
       setSelectedExistingModel('');
       setInputModelName('');
   };

   const handleExistingModelChange = (event) => {
       setSelectedExistingModel(event.target.value);
       setInputModelName(''); // Clear text input if dropdown used
   };

   const handleInputModelChange = (event) => {
       setInputModelName(event.target.value);
       // Clear dropdown if text input is used for local backend
       if (selectedBackendType === 'local') {
           setSelectedExistingModel('');
       }
   };


  // Combine local paths and cached hub models for the dropdown
  const allExistingLocalModels = [
      ...availableModels.local_model_paths.map(p => ({ value: p, label: `(Path) ${p.split(/[\\/]/).pop()}` })), // Extract filename for label
      ...availableModels.cached_hub_models.map(id => ({ value: id, label: `(Hub) ${id}` }))
  ].sort((a, b) => a.label.localeCompare(b.label)); // Sort alphabetically


  return (
    <div className={styles.llmManager}>
      <h2>LLM Configuration</h2>

      {/* --- Current Status Display --- */}
      <div className={styles.statusSection}>
        <h3>Current Status {isLoadingStatus ? '(Refreshing...)' : ''}</h3>
        <p>Backend: <strong>{currentStatus.backend_type || 'N/A'}</strong></p>
        <p>Active Model: <strong>{currentStatus.active_model || 'N/A'}</strong></p>
        <p>Status: <strong>{currentStatus.status || 'Unknown'}</strong></p>
        {currentStatus.last_error && <p className={styles.errorText}>Last Error: {currentStatus.last_error}</p>}
      </div>

      {/* --- Configuration Form --- */}
      <form onSubmit={handleLoadModel} className={styles.configForm}>
        <h3>Set Active LLM</h3>

        {/* Backend Type Selection */}
        <div className={styles.formGroup}>
          <label htmlFor="backendType">Backend Type:</label>
          <select
            id="backendType"
            value={selectedBackendType}
            onChange={handleBackendTypeChange}
            disabled={isSubmittingLoad}
          >
            {/* Default option based on server config? Or just list? */}
            {/* <option value="">-- Select Type (default: {settings.LLM_BACKEND_TYPE}) --</option> */}
             {BACKEND_TYPES.map(type => (
               <option key={type} value={type}>{type}</option>
             ))}
          </select>
        </div>

        {/* Model Selection/Input (Conditional) */}
        {selectedBackendType === 'local' && (
          <>
            <div className={styles.formGroup}>
              <label htmlFor="existingModel">Select Existing Local/Cached Model:</label>
              <select
                id="existingModel"
                value={selectedExistingModel}
                onChange={handleExistingModelChange}
                disabled={isSubmittingLoad || isLoadingAvailable}
              >
                <option value="">-- Select existing --</option>
                {allExistingLocalModels.map(model => (
                  <option key={model.value} value={model.value}>{model.label}</option>
                ))}
              </select>
               {isLoadingAvailable && <span> (Loading list...)</span>}
            </div>
            <div className={styles.formGroup}>
               <label htmlFor="newModelName">Or Enter New Hugging Face ID/Path:</label>
               <input
                type="text"
                id="newModelName"
                placeholder="e.g., gpt2 or /path/to/model"
                value={inputModelName}
                onChange={handleInputModelChange}
                disabled={isSubmittingLoad}
              />
            </div>
            {/* TODO: Add inputs for device/quantization if desired */}
          </>
        )}

        {/* Input for API Backend Model Name */}
        {selectedBackendType !== 'local' && selectedBackendType !== '' && (
           <div className={styles.formGroup}>
              <label htmlFor="apiModelName">Model Name (for {selectedBackendType}):</label>
              <input
                type="text"
                id="apiModelName"
                placeholder={`e.g., ${selectedBackendType === 'ollama' ? 'llama3' : selectedBackendType === 'gemini' ? 'gemini-1.5-flash-latest' : 'model-name'}`}
                value={inputModelName}
                onChange={handleInputModelChange} // Use same handler, it clears dropdown
                disabled={isSubmittingLoad}
                required // API backends need a model name
              />
           </div>
        )}

        {/* Submit Button */}
        <button type="submit" disabled={isSubmittingLoad || isLoadingStatus}>
          {isSubmittingLoad ? 'Processing...' : 'Load / Set Model'}
        </button>

         {/* Feedback Messages */}
         {loadMessage && <p className={styles.loadMessage}>{loadMessage}</p>}
         {errorMessage && <p className={styles.errorText}>{errorMessage}</p>}
      </form>

    </div>
  );
}

export default LLMManager;
```

**3. Create `LLMManager.module.css`**

Add some basic styling.

```css
/* src/components/LLMManager/LLMManager.module.css */
.llmManager {
  padding: 15px 20px;
  background-color: #f0f4f8; /* Light blue-grey background */
  border-bottom: 1px solid #d0d7de;
  flex-shrink: 0; /* Prevent shrinking */
  max-height: 40vh; /* Limit height */
  overflow-y: auto; /* Allow scrolling if content overflows */
}

.llmManager h2 {
  margin-top: 0;
  margin-bottom: 15px;
  color: #1f2328;
  font-size: 1.3em;
  border-bottom: 1px solid #d0d7de;
  padding-bottom: 5px;
}

.llmManager h3 {
  margin-top: 0;
  margin-bottom: 10px;
  color: #333;
  font-size: 1.1em;
}

.statusSection {
  background-color: #fff;
  border: 1px solid #d0d7de;
  border-radius: 6px;
  padding: 10px 15px;
  margin-bottom: 20px;
  font-size: 0.95em;
}

.statusSection p {
  margin: 5px 0;
  line-height: 1.4;
}

.statusSection strong {
  color: #0969da; /* Blueish color for values */
}

.configForm {
  display: flex;
  flex-direction: column;
  gap: 15px; /* Space between form groups */
}

.formGroup {
  display: flex;
  flex-direction: column; /* Stack label and input */
  gap: 5px; /* Space between label and input */
}

.formGroup label {
  font-weight: bold;
  color: #57606a;
  font-size: 0.9em;
}

.formGroup select,
.formGroup input[type="text"] {
  padding: 8px 10px;
  border: 1px solid #d0d7de;
  border-radius: 6px;
  font-size: 1em;
  font-family: inherit;
}

.formGroup select:disabled,
.formGroup input[type="text"]:disabled {
  background-color: #f6f8fa;
  cursor: not-allowed;
}


.configForm button[type="submit"] {
  align-self: flex-start; /* Align button to the left */
  padding: 8px 16px;
  border: 1px solid #1f883d;
  background-color: #2c974b;
  color: white;
  border-radius: 6px;
  cursor: pointer;
  font-size: 1em;
  font-weight: bold;
  transition: background-color 0.2s ease;
}

.configForm button[type="submit"]:hover {
  background-color: #21813b;
}

.configForm button[type="submit"]:disabled {
  background-color: #94d3a2;
  border-color: #82c08f;
  cursor: not-allowed;
}

.loadMessage {
  margin-top: 10px;
  color: #0969da;
  font-size: 0.9em;
}

.errorText {
  margin-top: 10px;
  color: #cf222e; /* Red for errors */
  font-size: 0.9em;
  font-weight: bold;
}

```

**4. Add `LLMManager` to `App.jsx`**

Import and render the new component.

```jsx
// src/App.jsx
import React, { useState, useEffect, useCallback } from 'react';
import ChatWindow from './components/ChatWindow/ChatWindow';
import MessageInput from './components/MessageInput/MessageInput';
import LLMManager from './components/LLMManager/LLMManager'; // <-- Import
import styles from './App.module.css';

// ... (API_BASE_URL, INITIAL_MESSAGE, TYPING_INDICATOR_ID) ...

function App() {
  // ... (existing state: messages, inputValue, sessionId, isLoading, isSessionLoading) ...

  // ... (addErrorMessage, ensureSession, useEffect for initial session) ...
  // ... (handleSendMessage, startNewChat) ...


  return (
    // --- Adjust App Container slightly ---
    // Use grid or allow manager to have flexible height?
    // Grid might be better for distinct sections.
    <div className={styles.appContainer}>
        {/* --- LLM Manager Section --- */}
        <LLMManager /> {/* <-- Render the LLM Manager */}
        {/* --- End LLM Manager Section --- */}

        {/* --- Chat Area --- */}
        <div className={styles.chatArea}> {/* Wrap chat window and input */}
            <ChatWindow messages={messages} />
            <MessageInput
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onSend={handleSendMessage}
                disabled={isLoading || isSessionLoading}
            />
            {/* Loading Indicators (kept simple for now) */}
            {isSessionLoading && <div className={styles.loadingIndicator}>Creating session...</div>}
            {isLoading && !isSessionLoading && <div className={styles.loadingIndicator}>Assistant is thinking...</div>}
        </div>
        {/* --- End Chat Area --- */}
    </div>
  );
}

export default App;
```

**5. Update `App.module.css` for Layout**

Modify the main container to better accommodate the two sections. A simple flexbox layout is fine.

```css
/* src/App.module.css */
.appContainer {
  display: flex;
  flex-direction: column; /* Stack Manager and Chat vertically */
  height: 100vh;
  max-height: 98vh;
  width: 100%;
  max-width: 1000px; /* Increased max-width slightly */
  /* margin: 0 auto; */ /* Centering handled by parent (#root) now */
  border: 1px solid #ccc;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
  overflow: hidden; /* Prevent content spill */
  background-color: #f9f9f9;
  border-radius: 8px;
}

/* Remove headerBar styles if not needed, manager has its own header */
/* .headerBar { ... } */
/* .appTitle { ... } */
/* .newChatButton { ... } */

/* --- NEW: Style for Chat Area --- */
.chatArea {
    flex-grow: 1; /* Make chat area take remaining vertical space */
    display: flex;
    flex-direction: column;
    min-height: 0; /* Important for flex-grow in column layout */
    background-color: #ffffff; /* Give chat area distinct background */
}
/* --- END NEW --- */


.loadingIndicator {
    /* ... (styles remain same) ... */
    flex-shrink: 0;
}

```
*   Made `AppContainer` a column flexbox.
*   Wrapped `ChatWindow`, `MessageInput`, and loading indicators in a `chatArea` div.
*   Set `flex-grow: 1` on `chatArea` so it expands to fill space not taken by `LLMManager`.
*   Set `min-height: 0` on `chatArea`, often needed for flex children to shrink correctly.
*   Removed old `headerBar` styles (optional, depends if you want the New Chat button elsewhere).

**How to Test:**

1.  Restart the frontend (`npm run dev`).
2.  Hard refresh the browser.
3.  You should see the new "LLM Configuration" section at the top.
4.  It should fetch and display the current status (likely inactive or showing the default).
5.  It should fetch and populate the dropdown for local/cached models.
6.  **Test Local:**
    *   Select "local" backend type.
    *   Select an existing model from the dropdown OR type a *new* valid HF ID (like `distilgpt2`).
    *   Click "Load / Set Model".
    *   Observe the status updates and feedback messages. Check server logs.
7.  **Test API Backend (e.g., Ollama):**
    *   Ensure Ollama server is running with a model (e.g., `llama3`).
    *   Select "ollama" backend type.
    *   Type the *Ollama model name* (e.g., `llama3`) into the "Model Name (for ollama)" input field.
    *   Click "Load / Set Model".
    *   Observe status updates (should quickly go to `Ready`).
8.  **Test Chatting:** After successfully setting a model (local or API), try chatting in the section below. It should use the model you just configured.

This provides the requested UI for selecting and configuring the LLM backend and model.
