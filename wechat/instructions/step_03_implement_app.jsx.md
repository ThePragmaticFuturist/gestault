**Step 3: Implement `App.jsx` (Main Container)**

*   **What:** Set up the main application layout and basic state management for messages and the input field.
*   **Why:** The `App` component will orchestrate the overall application, hold the conversation state, and pass data/functions down to child components.
*   **How:**

    ```jsx
    // src/App.jsx
    import React, { useState, useEffect } from 'react'; // Import hooks
    import ChatWindow from './components/ChatWindow/ChatWindow';
    import MessageInput from './components/MessageInput/MessageInput';
    import styles from './App.module.css'; // Import CSS Module for App

    function App() {
      // State for the list of messages { role: 'user'/'assistant', content: 'text' }
      const [messages, setMessages] = useState([
        // Initial placeholder message (optional)
        { role: 'assistant', content: 'Hello! How can I help you today?' }
      ]);
      // State for the text currently in the input field
      const [inputValue, setInputValue] = useState('');

      // --- TODO LATER ---
      // State for the current chat session ID
      // const [sessionId, setSessionId] = useState(null);
      // Function to start a new session
      // Function to fetch message history for a session
      // ------------------

      // Function to handle sending a message
      const handleSendMessage = () => {
        if (!inputValue.trim()) return; // Don't send empty messages

        const userMessage = { role: 'user', content: inputValue };
        // Add user message immediately to the UI
        setMessages(prevMessages => [...prevMessages, userMessage]);

        // --- TODO LATER ---
        // 1. Get or Create Session ID
        // 2. Send userMessage.content + sessionID to backend API
        // 3. Receive assistant response from backend
        // 4. Add assistant response to messages state
        //    const assistantResponse = { role: 'assistant', content: '...' };
        //    setMessages(prevMessages => [...prevMessages, assistantResponse]);
        // ------------------

        // For now, simulate an assistant response after a short delay
        const simulatedResponse = { role: 'assistant', content: `Echo: ${inputValue}` };
        setTimeout(() => {
          setMessages(prevMessages => [...prevMessages, simulatedResponse]);
        }, 500);


        setInputValue(''); // Clear the input field
      };

      return (
        <div className={styles.appContainer}>
          <h1 className={styles.appTitle}>RAG LLM Chat</h1>
          <ChatWindow messages={messages} />
          <MessageInput
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onSend={handleSendMessage}
          />
        </div>
      );
    }

    export default App;
    ```

    *   Create `src/App.module.css`:
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

        .appTitle {
          text-align: center;
          padding: 15px;
          margin: 0;
          background-color: #4a90e2; /* Example header color */
          color: white;
          font-size: 1.5em;
          flex-shrink: 0; /* Prevent header from shrinking */
        }
        ```
