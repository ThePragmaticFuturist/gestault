# Step 12: Loading Indicators #

Okay, we already have the basic "Assistant is thinking..." text indicator controlled by the `isLoading` state. Let's enhance this slightly to be more visually distinct and potentially add a classic "typing dots" animation.

**1. Enhance Loading Indicator Style (`src/App.module.css`)**

We can make the existing indicator a bit more prominent and add a simple pulsing animation.

```css
/* src/App.module.css */
/* ... (other styles) ... */

/* Enhanced Loading Indicator */
.loadingIndicator {
    display: flex; /* Use flex for alignment */
    align-items: center;
    justify-content: center; /* Center content */
    padding: 8px 15px; /* Adjust padding */
    font-style: italic;
    color: #555; /* Slightly darker grey */
    background-color: #e9e9e9; /* Lighter background */
    border-top: 1px solid #dcdcdc;
    flex-shrink: 0;
    min-height: 30px; /* Ensure it has some height */
    box-sizing: border-box;
}

/* Simple pulsing dots animation (optional) */
.loadingIndicator::after {
  content: '.';
  animation: dots 1.4s infinite;
  animation-timing-function: ease-in-out;
  margin-left: 2px; /* Space after text */
}

@keyframes dots {
  0%, 20% {
    content: '.';
  }
  40% {
    content: '..';
  }
  60% {
    content: '...';
  }
  80%, 100% {
    content: '.';
  }
}

```
*   Changed background/border/text colors slightly.
*   Used flexbox to center the text better.
*   Added a CSS `::after` pseudo-element with a keyframe animation (`@keyframes dots`) to create simple animated ellipsis (...) to give a sense of activity.

**2. Implement Typing Indicator within `MessageBubble` (More Advanced)**

A common pattern is to show the typing indicator *as if it were an assistant message* while waiting. This requires adding a temporary placeholder message to the `messages` state.

**Modify `App.jsx`:**

```jsx
// src/App.jsx
import React, { useState, useEffect, useCallback } from 'react';
import ChatWindow from './components/ChatWindow/ChatWindow';
import MessageInput from './components/MessageInput/MessageInput';
import styles from './App.module.css';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';
console.log("Using API Base URL:", API_BASE_URL);

const INITIAL_MESSAGE = { role: 'assistant', content: 'Hello! How can I help you today?' };
// --- Add a unique ID for the typing indicator ---
const TYPING_INDICATOR_ID = 'typing-indicator-message';

function App() {
  const [messages, setMessages] = useState([INITIAL_MESSAGE]);
  const [inputValue, setInputValue] = useState('');
  const [sessionId, setSessionId] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isSessionLoading, setIsSessionLoading] = useState(false);

  // ... (addErrorMessage helper - unchanged) ...
  // ... (ensureSession function - unchanged) ...
  // ... (useEffect for initial session - unchanged) ...


  // --- Handle Sending a Message (Modified for Typing Indicator) ---
  const handleSendMessage = async () => {
    const messageText = inputValue.trim();
    if (!messageText || isLoading || isSessionLoading) return;

    const userMessage = { role: 'user', content: messageText };
    // --- Add Typing Indicator ---
    const typingIndicatorMessage = {
        id: TYPING_INDICATOR_ID, // Assign unique ID
        role: 'assistant',
        content: '...', // Placeholder content (will be styled)
        isLoading: true, // Add a flag to identify this bubble
    };
    setMessages(prevMessages => [...prevMessages, userMessage, typingIndicatorMessage]); // Add user msg AND indicator
    // --- End Add Typing Indicator ---

    setInputValue('');
    setIsLoading(true); // Start loading

    let currentSessionId = sessionId;

    try {
        if (!currentSessionId) {
            currentSessionId = await ensureSession();
        }
        if (!currentSessionId) {
             console.error("Cannot send message without a valid session ID.");
             // --- Remove typing indicator on session error ---
             setMessages(prevMessages => prevMessages.filter(msg => msg.id !== TYPING_INDICATOR_ID));
             setIsLoading(false);
             return;
        }

        const response = await fetch(`${API_BASE_URL}/api/v1/sessions/${currentSessionId}/messages`, { /* ... */ });

        if (!response.ok) { /* ... error handling ... */ throw new Error(/* ... */); }

        const assistantMessageData = await response.json();
        let isBackendError = false;
        if (assistantMessageData.content && typeof assistantMessageData.content === 'string' && assistantMessageData.content.startsWith('[ERROR:')) {
            isBackendError = true;
        }

        // --- Replace Typing Indicator with Actual Response ---
        setMessages(prevMessages =>
            // Filter out the typing indicator...
            prevMessages.filter(msg => msg.id !== TYPING_INDICATOR_ID)
            // ...and add the actual response (or error from backend)
            .concat([{ ...assistantMessageData, isError: isBackendError }])
        );
        // --- End Replace Indicator ---

    } catch (error) {
      console.error("Error during message send/receive:", error);
      // --- Remove typing indicator and Add Error Bubble ---
      setMessages(prevMessages =>
          prevMessages.filter(msg => msg.id !== TYPING_INDICATOR_ID) // Remove indicator
      );
      addErrorMessage(`Could not get response. ${error.message}`); // Add separate error message
      // --- End ---
    } finally {
      setIsLoading(false); // Stop loading indicator
    }
  };

  // --- Handler for New Chat Button ---
  // ... (startNewChat function - unchanged) ...


  return (
    <div className={styles.appContainer}>
      {/* Header Bar */}
      {/* ... (unchanged) ... */}
      <ChatWindow messages={messages} />
      <MessageInput
        value={inputValue}
        onChange={(e) => setInputValue(e.target.value)}
        onSend={handleSendMessage}
        disabled={isLoading || isSessionLoading}
      />
      {/* --- Remove the separate text indicator --- */}
      {/* {isSessionLoading && <div className={styles.loadingIndicator}>Creating session...</div>} */}
      {/* {isLoading && !isSessionLoading && <div className={styles.loadingIndicator}>Assistant is thinking...</div>} */}
      {/* --- End Remove --- */}
    </div>
  );
}

export default App;
```

**Modify `MessageBubble.jsx`:**

Check for the `isLoading` prop on the message and render the typing animation instead of the content.

```jsx
// src/components/MessageBubble/MessageBubble.jsx
import React from 'react';
import styles from './MessageBubble.module.css'; // Assuming you have the animation here

function MessageBubble({ message }) {
  // Destructure isLoading flag
  const { role, content, isError = false, isLoading = false } = message;
  const isUser = role === 'user';

  const bubbleClass = isUser
    ? styles.userBubble
    : (isError ? styles.errorBubble : styles.assistantBubble);

  const alignmentClass = isUser ? styles.userAlignment : styles.assistantAlignment;

  // --- Add conditional rendering for loading state ---
  const bubbleContent = isLoading
    ? ( <div className={styles.typingIndicator}>
          <span></span><span></span><span></span> {/* Dots for animation */}
        </div> )
    : content; // Render normal content if not loading
  // --- End conditional rendering ---

  // Add loading class to container if needed for specific alignment/styling
  const containerClass = `${styles.messageContainer} ${alignmentClass} ${isLoading ? styles.loadingContainer : ''}`;

  return (
    // Use containerClass
    <div className={containerClass}>
      {/* Don't apply errorBubble style if it's just the loading indicator */}
      <div className={`${styles.messageBubble} ${isLoading ? '' : bubbleClass}`}>
        {bubbleContent}
      </div>
    </div>
  );
}

export default MessageBubble;

```

**Modify `MessageBubble.module.css`:**

Add styles for the typing indicator animation.

```css
/* src/components/MessageBubble/MessageBubble.module.css */
/* ... (existing styles) ... */

/* Style for the loading container if needed */
.loadingContainer {
 /* Add specific styles if needed, e.g., alignment */
}

/* Typing Indicator dots */
.typingIndicator {
  display: flex;
  align-items: center;
  padding: 10px 0; /* Adjust padding for indicator */
}

.typingIndicator span {
  height: 8px;
  width: 8px;
  margin: 0 2px;
  background-color: #999; /* Dot color */
  border-radius: 50%;
  display: inline-block;
  animation: typing 1.4s infinite;
  animation-timing-function: ease-in-out;
}

/* Animation delays for the dots */
.typingIndicator span:nth-of-type(1) {
  animation-delay: 0s;
}
.typingIndicator span:nth-of-type(2) {
  animation-delay: 0.2s;
}
.typingIndicator span:nth-of-type(3) {
  animation-delay: 0.4s;
}

/* Keyframes for the typing animation */
@keyframes typing {
  0%, 100% {
    opacity: 0.3;
    transform: scale(0.7);
  }
  50% {
    opacity: 1;
    transform: scale(1);
  }
}

```

**Explanation of Changes (Typing Indicator Method):**

1.  **`App.jsx`:**
    *   A unique ID `TYPING_INDICATOR_ID` is defined.
    *   When `handleSendMessage` starts, it *immediately* adds both the user message and a special `typingIndicatorMessage` (with `isLoading: true` and the ID) to the `messages` state.
    *   When the API call succeeds, `setMessages` is used to *filter out* the message with `TYPING_INDICATOR_ID` and concatenate the actual `assistantMessageData`.
    *   When the API call fails (either network error or session creation error), `setMessages` is used to filter out the typing indicator, and then `addErrorMessage` adds the separate error bubble.
    *   The separate text-based loading indicator at the bottom of `App.jsx` is removed.
2.  **`MessageBubble.jsx`:**
    *   It now destructures the `isLoading` prop from the message object.
    *   It uses conditional rendering: If `isLoading` is true, it renders the `div` with the animated dots (`typingIndicator`); otherwise, it renders the regular message `content`.
    *   It prevents applying error styling to the loading indicator bubble itself.
3.  **`MessageBubble.module.css`:**
    *   Adds styles for `.typingIndicator` and its `span` elements.
    *   Defines the `@keyframes typing` animation for the pulsing dot effect.

**Choose Your Preferred Method:**

*   **Method 1 (Text Indicator):** Simpler to implement, shows loading at the bottom of the app. Use the CSS enhancement from Step 1 and the *first* version of `App.jsx` in this answer (which kept the text indicator).
*   **Method 2 (Typing Indicator Bubble):** More conventional chat UX, shows the indicator within the message flow. Use the `App.jsx` and `MessageBubble.jsx`/`.css` changes from Steps 2 & 3.

Pick the method you prefer, implement the corresponding code changes, restart your dev server, and test!
