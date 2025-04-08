**Step 1: Clean Up Default Project**

*   **What:** Remove the default Vite/React demo content.
*   **Why:** To start with a clean slate for our chat application.
*   **How:**
    1.  Open the `rag-chat-ui` project folder in your code editor (VS Code).
    2.  Edit `src/App.jsx`: Remove the boilerplate content inside the `return (...)` statement and the `useState` hook. Replace it with a basic structure:
        ```jsx
        // src/App.jsx
        import React from 'react'; // Import useState later
        import './App.css'; // Keep global styles for now

        function App() {
          // We will add state later

          return (
            <div className="app-container">
              <h1>RAG LLM Chat</h1>
              {/* Chat components will go here */}
            </div>
          );
        }

        export default App;
        ```
    3.  Edit `src/App.css`: Delete all the existing CSS rules. We'll add minimal global styles later if needed, but component styles will be separate.
    4.  Delete `src/assets/react.svg`.
    5.  (Optional) Edit `index.html` in the project root to change the `<title>`.

**Step 2: Create Component Structure**

*   **What:** Create folders and files for our main chat UI components.
*   **Why:** To organize the code logically and make it easier to manage and style individual parts. We will use **CSS Modules** for styling to keep styles scoped to their components, fulfilling the "custom CSS per component" requirement without needing frameworks. CSS Modules automatically generate unique class names, preventing conflicts.
*   **How:**
    1.  Inside the `src` folder, create a new folder named `components`.
    2.  Inside `src/components`, create folders for each component:
        *   `ChatWindow`
        *   `MessageList`
        *   `MessageBubble`
        *   `MessageInput`
    3.  Inside *each* of these component folders, create two files:
        *   `ComponentName.jsx` (e.g., `ChatWindow.jsx`) - The React component code.
        *   `ComponentName.module.css` (e.g., `ChatWindow.module.css`) - The CSS styles for that specific component.

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

**Step 4: Implement `ChatWindow` Component**

*   **What:** A container component that receives the list of messages and renders the `MessageList`.
*   **Why:** To provide a dedicated area for the conversation display, handling potential scrolling.
*   **How:**

    ```jsx
    // src/components/ChatWindow/ChatWindow.jsx
    import React from 'react';
    import MessageList from '../MessageList/MessageList';
    import styles from './ChatWindow.module.css';

    function ChatWindow({ messages }) {
      return (
        <div className={styles.chatWindow}>
          <MessageList messages={messages} />
        </div>
      );
    }

    export default ChatWindow;
    ```

    *   Create `src/components/ChatWindow/ChatWindow.module.css`:
        ```css
        /* src/components/ChatWindow/ChatWindow.module.css */
        .chatWindow {
          flex-grow: 1; /* Take up remaining vertical space */
          overflow-y: auto; /* Enable vertical scrolling */
          padding: 15px;
          background-color: #ffffff;
          display: flex; /* Needed for flex-grow */
          flex-direction: column; /* Stack messages vertically */
        }
        ```

**Step 5: Implement `MessageList` Component**

*   **What:** Renders the individual message bubbles from the `messages` array. Handles scrolling to the bottom.
*   **Why:** To iterate through messages and manage the display flow within the `ChatWindow`.
*   **How:**

    ```jsx
    // src/components/MessageList/MessageList.jsx
    import React, { useEffect, useRef } from 'react';
    import MessageBubble from '../MessageBubble/MessageBubble';
    import styles from './MessageList.module.css';

    function MessageList({ messages }) {
      const messagesEndRef = useRef(null); // Ref to scroll to

      // Function to scroll to the bottom
      const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
      };

      // Scroll to bottom whenever messages array changes
      useEffect(() => {
        scrollToBottom();
      }, [messages]);

      return (
        <div className={styles.messageList}>
          {messages.map((msg, index) => (
            <MessageBubble key={index} message={msg} />
          ))}
          {/* Empty div at the end to scroll to */}
          <div ref={messagesEndRef} />
        </div>
      );
    }

    export default MessageList;
    ```

    *   Create `src/components/MessageList/MessageList.module.css`:
        ```css
        /* src/components/MessageList/MessageList.module.css */
        .messageList {
          display: flex;
          flex-direction: column;
          gap: 10px; /* Space between messages */
        }
        ```

**Step 6: Implement `MessageBubble` Component**

*   **What:** Displays a single message, styled differently based on the role (`user` or `assistant`).
*   **Why:** The visual representation of each turn in the conversation.
*   **How:**

    ```jsx
    // src/components/MessageBubble/MessageBubble.jsx
    import React from 'react';
    import styles from './MessageBubble.module.css';

    function MessageBubble({ message }) {
      const { role, content } = message;
      const isUser = role === 'user';

      // Apply different styles based on the role
      const bubbleClass = isUser ? styles.userBubble : styles.assistantBubble;
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

    *   Create `src/components/MessageBubble/MessageBubble.module.css`:
        ```css
        /* src/components/MessageBubble/MessageBubble.module.css */
        .messageContainer {
          display: flex;
          margin-bottom: 10px; /* Consistent spacing */
        }

        .messageBubble {
          max-width: 75%; /* Prevent bubbles from being too wide */
          padding: 10px 15px;
          border-radius: 18px; /* Rounded bubbles */
          word-wrap: break-word; /* Break long words */
          line-height: 1.4;
        }

        /* Alignment */
        .userAlignment {
          justify-content: flex-end; /* Align user messages to the right */
        }

        .assistantAlignment {
          justify-content: flex-start; /* Align assistant messages to the left */
        }

        /* Bubble specific styles */
        .userBubble {
          background-color: #0b93f6; /* Example blue for user */
          color: white;
          border-bottom-right-radius: 4px; /* Slightly different shape */
        }

        .assistantBubble {
          background-color: #e5e5ea; /* Example grey for assistant */
          color: black;
          border-bottom-left-radius: 4px; /* Slightly different shape */
        }
        ```

**Step 7: Implement `MessageInput` Component**

*   **What:** Provides the text input field and send button.
*   **Why:** Allows the user to type and submit their messages.
*   **How:**

    ```jsx
    // src/components/MessageInput/MessageInput.jsx
    import React from 'react';
    import styles from './MessageInput.module.css';

    function MessageInput({ value, onChange, onSend }) {

      // Handle sending message on Enter key press
      const handleKeyPress = (event) => {
        if (event.key === 'Enter' && !event.shiftKey) { // Send on Enter, allow Shift+Enter for newline
          event.preventDefault(); // Prevent default form submission/newline
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
            placeholder="Type your message..."
            rows="1" // Start with one row, CSS will handle expansion
          />
          <button className={styles.sendButton} onClick={onSend}>
            Send
          </button>
        </div>
      );
    }

    export default MessageInput;
    ```

    *   Create `src/components/MessageInput/MessageInput.module.css`:
        ```css
        /* src/components/MessageInput/MessageInput.module.css */
        .inputArea {
          display: flex;
          padding: 10px;
          border-top: 1px solid #ccc;
          background-color: #f0f0f0;
          flex-shrink: 0; /* Prevent input area from shrinking */
        }

        .inputField {
          flex-grow: 1;
          border: 1px solid #ccc;
          border-radius: 18px;
          padding: 10px 15px;
          font-size: 1em;
          resize: none; /* Prevent manual resizing */
          overflow-y: auto; /* Allow scrolling if text gets tall */
          max-height: 100px; /* Limit max height */
          margin-right: 10px;
          line-height: 1.4;
          font-family: inherit; /* Use the app's font */
        }

        .inputField:focus {
          outline: none;
          border-color: #4a90e2;
          box-shadow: 0 0 0 2px rgba(74, 144, 226, 0.2);
        }

        .sendButton {
          padding: 10px 20px;
          border: none;
          background-color: #4a90e2;
          color: white;
          border-radius: 18px;
          cursor: pointer;
          font-size: 1em;
          font-weight: bold;
          transition: background-color 0.2s ease;
        }

        .sendButton:hover {
          background-color: #357abd;
        }

        .sendButton:disabled {
          background-color: #ccc;
          cursor: not-allowed;
        }
        ```

**Step 8: Run and Verify**

*   If the development server (`npm run dev`) is still running, your browser should have automatically updated. If not, restart it.
*   You should now see a basic chat interface:
    *   A title bar.
    *   A chat area with the initial "Hello!" message from the assistant.
    *   An input area at the bottom.
*   Try typing a message and pressing Enter or clicking Send.
    *   Your message should appear aligned to the right (blue bubble).
    *   After a short delay, an "Echo:" response should appear from the assistant (grey bubble).
    *   The chat window should scroll automatically.

**Next Steps:**

1.  **API Integration:** Replace the `setTimeout` echo logic in `App.jsx`'s `handleSendMessage` with actual `fetch` calls to your RAG LLM server's API (`POST /api/v1/sessions/{session_id}/messages`).
2.  **Session Management:** Implement logic to create a new session (`POST /api/v1/sessions`) when the app starts or the user wants a new chat, store the `session_id` in the React state, and use it in API calls.
3.  **Error Handling:** Add handling for API errors (e.g., display an error message in the chat).
4.  **Loading Indicators:** Show some visual feedback while waiting for the assistant's response from the backend.

---

**[c] Glossary of Terms (React Focused)**

*   **React:** A JavaScript library for building user interfaces, particularly single-page applications, using a component-based architecture.
*   **Component:** A reusable, self-contained piece of UI (User Interface) in React (e.g., `MessageBubble`, `MessageInput`). Written as JavaScript functions or classes.
*   **JSX (JavaScript XML):** A syntax extension for JavaScript that looks similar to HTML and is used within React components to describe the UI structure. It gets compiled into regular JavaScript.
*   **Vite:** A fast frontend build tool used to set up the development environment, bundle code for production, and provide features like hot module replacement.
*   **npm (Node Package Manager):** A command-line tool used to install and manage project dependencies (JavaScript libraries).
*   **`package.json`:** A file in the project root that lists project dependencies and defines scripts (like `npm run dev`).
*   **State:** Data that determines a component's behavior and rendering. Managed within components using Hooks like `useState`. When state changes, React re-renders the component.
*   **Props (Properties):** Data passed down from a parent component to a child component (read-only for the child). Used for communication (e.g., passing `messages` from `App` to `ChatWindow`).
*   **Hooks:** Special functions in React (like `useState`, `useEffect`, `useRef`) that let you "hook into" React features like state and lifecycle methods from functional components.
    *   **`useState`:** A Hook to add state variables to functional components. Returns the current state value and a function to update it.
    *   **`useEffect`:** A Hook to perform side effects in components (e.g., fetching data, setting up subscriptions, manually changing the DOM). Used here to scroll the message list.
    *   **`useRef`:** A Hook to create a mutable ref object whose `.current` property can hold a value that persists across renders without causing a re-render itself. Used here to get a reference to the DOM element for scrolling.
*   **CSS Modules:** A CSS file type (`.module.css`) where all class names and animations are scoped locally to the component that imports them by default. Prevents global style conflicts. Imported as an object (`styles`) in JS.
*   **Hot Module Replacement (HMR):** A feature (provided by Vite) that automatically updates modules in a running application without a full page reload, speeding up development.
*   **Controlled Component:** An input element (like `<textarea>`) whose value is controlled by React state (`inputValue` state and `onChange` handler).

---

**[d] List of Future Features for React App**

*(Building on the "Future Features" for the backend)*

**Core Chat Functionality:**

1.  **API Integration:** Connect `handleSendMessage` to the backend API, handle responses.
2.  **Session Management:** Create/select sessions, store `session_id`.
3.  **Error Handling:** Display API/backend errors gracefully in the UI.
4.  **Loading Indicators:** Show typing indicators or spinners while waiting for responses.
5.  **Streaming Response Display:** Update the assistant message bubble incrementally as tokens arrive from the backend (requires backend streaming support).

**UI/UX Enhancements:**

6.  **Markdown Rendering:** Render Markdown formatting (bold, italics, lists, code blocks) in message bubbles.
7.  **Code Syntax Highlighting:** Apply syntax highlighting to code blocks within messages.
8.  **User Feedback:** Allow users to rate responses or provide feedback.
9.  **Message Copying:** Add a button to easily copy message content.
10. **Theme Toggling:** Allow switching between light and dark themes.
11. **Responsive Design Improvements:** Fine-tune CSS for optimal layout on various screen sizes (tablets, different mobile widths).
12. **Input Autosize:** Make the textarea height grow dynamically based on content.

**Advanced Features:**

13. **Session List & Switching:** Display a list of past sessions, allow users to switch between them, rename, and delete them via API calls.
14. **Model/Backend Selection UI:** Allow users to select the LLM backend and model via the UI (calling the `/api/v1/models/load` endpoint).
15. **Generation Parameter UI:** Allow users to adjust temperature, max tokens, etc., via the UI (calling the `/api/v1/models/config` endpoint).
16. **RAG Document Management UI:** Interface to upload documents, view processing status, list available documents, and select documents to associate with a session.
17. **Display Retrieved Context:** Optionally show which RAG chunks were used for a specific assistant response.
18. **System Status Display:** Show key metrics from the `/api/v1/system/status` endpoint in the UI.
19. **Authentication UI:** Login/logout forms if backend authentication is added.
20. **Agentic Features:** (More complex) UI elements to define agent goals, tools, trigger agent runs, and display intermediate steps or thoughts.
