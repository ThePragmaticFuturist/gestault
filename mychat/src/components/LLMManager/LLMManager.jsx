// src/components/LLMManager/LLMManager.jsx
import React, { useState, useEffect, useCallback } from 'react';
import GenerationConfig from '../GenerationConfig/GenerationConfig';
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

  const [fetchTimer, setFetchTimer] = useState(null)

  const [currentGenConfig, setCurrentGenConfig] = useState({});

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

      setCurrentGenConfig(data?.generation_config || {});

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
    // const intervalId = setInterval(fetchStatus, 30000); // Refresh status every 30 seconds
    // return () => clearInterval(intervalId); // Cleanup interval on unmount

    return () => {
    	clearTimeout(fetchTimer);
    	setFetchTimer(null);
    }
    
  }, [fetchAvailableModels]); // fe}tchStatus is stable due to useCallback dependencies


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
        fetchStatus();
        setFetchTimer(setTimeout(fetchStatus, 2000)); // Refresh status after 2 seconds

    } catch (error) {
        console.error("Failed to load/set model:", error);
        setErrorMessage(`Load request failed: ${error.message}`);
    } finally {
        setIsSubmittingLoad(false);
    }
   };

  const handleConfigUpdate = useCallback((newConfig) => {
      console.log("Config updated from child:", newConfig);
      setCurrentGenConfig(newConfig); // Update state managed by LLMManager
      // Optionally, notify App if App needs the config? Not currently needed.
  }, []);

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