// src/components/GenerationConfig/GenerationConfig.jsx
import React, { useState, useEffect, useCallback } from 'react';
import styles from './GenerationConfig.module.css';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

// Define parameter details (min, max, step)
const PARAM_CONFIG = {
    max_new_tokens: { label: "Max New Tokens", min: 1, max: 4096, step: 1, type: 'number' },
    temperature: { label: "Temperature", min: 0.0, max: 2.0, step: 0.05, type: 'range' },
    top_p: { label: "Top P", min: 0.0, max: 1.0, step: 0.05, type: 'range' },
    top_k: { label: "Top K", min: 1, max: 100, step: 1, type: 'number' }, // Min 1 for Gemini
    repetition_penalty: { label: "Repetition Penalty", min: 1.0, max: 2.0, step: 0.05, type: 'range' }
};

function GenerationConfig({ currentConfig, onConfigChange, disabled }) {
    const [localConfig, setLocalConfig] = useState(currentConfig);
    const [isUpdating, setIsUpdating] = useState(false);
    const [updateMessage, setUpdateMessage] = useState('');
    const [errorMessage, setErrorMessage] = useState('');

    // Update local state when props change (e.g., after loading a session or fetching status)
    useEffect(() => {
        setLocalConfig(currentConfig || {}); // Use empty object if currentConfig is null/undefined
    }, [currentConfig]);

    const handleInputChange = (event) => {
        const { name, value } = event.target;
        setLocalConfig(prev => ({
            ...prev,
            // Convert number/range inputs back to numbers
            [name]: (PARAM_CONFIG[name]?.type === 'number' || PARAM_CONFIG[name]?.type === 'range') ? parseFloat(value) : value
        }));
        setUpdateMessage(''); // Clear message on change
        setErrorMessage('');
    };

    // Debounce update function to avoid spamming API on range slider changes
    const debounce = (func, delay) => {
        let timeoutId;
        return function(...args) {
            clearTimeout(timeoutId);
            timeoutId = setTimeout(() => {
                func.apply(this, args);
            }, delay);
        };
    };

    // Update backend (debounced for range sliders)
    const updateBackendConfig = useCallback(async (paramName, paramValue) => {
        setIsUpdating(true);
        setUpdateMessage(`Updating ${paramName}...`);
        setErrorMessage('');
        console.log(`Updating backend config: ${paramName}=${paramValue}`);

        try {
            const body = { [paramName]: paramValue };
            const response = await fetch(`${API_BASE_URL}/api/v1/models/config`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json', 'Accept': 'application/json' },
                body: JSON.stringify(body),
            });

            if (!response.ok) {
                 let errorDetail = `HTTP ${response.status}`;
                 try { const errorData = await response.json(); errorDetail = errorData.detail || errorDetail; } catch(e){}
                 throw new Error(`Update failed: ${errorDetail}`);
            }

            const updatedConfig = await response.json();
            setUpdateMessage(`${paramName} updated successfully!`);
            onConfigChange(updatedConfig); // Notify parent of the full updated config
            setTimeout(() => setUpdateMessage(''), 2000); // Clear message after 2s

        } catch (error) {
            console.error(`Failed to update ${paramName}:`, error);
            setErrorMessage(`Error updating ${paramName}: ${error.message}`);
            // Revert local state? Optional, might be complex.
        } finally {
            setIsUpdating(false);
            // Clear specific param update message slightly later if needed
            // setTimeout(() => { if (updateMessage.startsWith(`Updating ${paramName}`)) setUpdateMessage(''); }, 2500);
        }
    }, [onConfigChange]);

    // Create debounced version for range sliders
    const debouncedUpdateBackend = useCallback(debounce(updateBackendConfig, 500), [updateBackendConfig]); // 500ms debounce

    // Handle change commit (e.g., for range sliders on mouse up, or number input on blur)
    const handleValueCommit = (event) => {
        const { name, value } = event.target;
        const numericValue = parseFloat(value); // Ensure it's a number
         // Don't trigger update if value hasn't actually changed from backend state
        if (currentConfig && currentConfig[name] !== numericValue) {
            if (PARAM_CONFIG[name]?.type === 'range') {
                debouncedUpdateBackend(name, numericValue);
            } else { // Update immediately for number inputs on blur/change commit
                 updateBackendConfig(name, numericValue);
            }
        }
    };


    return (
        <div className={styles.genConfig}>
            <h4>Generation Parameters</h4>
            <div className={styles.paramsGrid}>
                {Object.entries(PARAM_CONFIG).map(([key, details]) => {
                    const currentValue = localConfig[key] ?? ''; // Use '' if undefined/null
                    return (
                        <div key={key} className={styles.paramControl}>
                            <label htmlFor={key}>
                                {details.label}: {details.type === 'range' ? parseFloat(currentValue).toFixed(2) : currentValue}
                            </label>
                            <input
                                type={details.type}
                                id={key}
                                name={key}
                                min={details.min}
                                max={details.max}
                                step={details.step}
                                value={currentValue}
                                onChange={handleInputChange}
                                // Use onMouseUp for range sliders to commit value after dragging
                                onMouseUp={details.type === 'range' ? handleValueCommit : undefined}
                                // Use onBlur for number inputs to commit value when leaving field
                                onBlur={details.type === 'number' ? handleValueCommit : undefined}
                                disabled={disabled || isUpdating}
                            />
                        </div>
                    );
                })}
            </div>
            <div className={styles.feedbackArea}>
                 {isUpdating && <span className={styles.updateMessage}>{updateMessage || 'Updating...'}</span>}
                 {errorMessage && <span className={styles.errorText}>{errorMessage}</span>}
                 {!isUpdating && updateMessage && <span className={styles.updateMessage}>{updateMessage}</span>}
            </div>
        </div>
    );
}

export default GenerationConfig;