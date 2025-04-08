# A Step-by-Step Guide to Build Your Chat Web App #  

> ### One Prompt Shall Rule Them All ###
> Now we are going to start a new project built to connect a user to the functionality of the RAG LLM server we just finished. I want to start out with a basic Generative AI Chat web app built in React.js for cross browser desktop and mobile device deployment. We want to make it flexible so we can add features later with the ultimate goal a robust Chat and AI Agent application platform. We are going to presume the user knows HTML5, CSS, and JavaScript. They are beginner level React.js developers. Provide instruction and full installation of dependencies, node.js modules, libraries, etc. Use custom CSS for each component creating a modern generic user experience. Do not use any CSS frameworks because we want the developer to be able to modify the user interface elements for their own preferences.
>
> [Click here for the full prompt transcript with bug fixing](wechat/instructions/z4_the_complete_prompt_response_thread.md)

***

*Here's a breakdown of the thought process behind generating the initial project structure and the React Chat App:*

1.  **Understand the Goal:** The user wants to build a React frontend for the RAG LLM server we previously created. Key requirements:
    *   React.js.
    *   Basic GenAI Chat functionality (input, send, display messages).
    *   Cross-browser/device compatible (implies standard web tech).
    *   Flexible for future expansion (component-based).
    *   Target audience: Beginner React developers familiar with HTML/CSS/JS.
    *   Use custom CSS per component, *no* CSS frameworks (Tailwind, Bootstrap, etc.).
    *   Provide full setup instructions including Node.js and dependencies.

2.  **Identify Core Components:** A basic chat application needs:
    *   **App Container:** The main wrapper.
    *   **Chat Window:** Displays the conversation history.
    *   **Message List:** Holds individual messages within the chat window.
    *   **Message Bubble:** Represents a single message (user or assistant).
    *   **Input Area:** Text input field and send button.

3.  **Choose a Setup Method:** For beginners, `create-react-app` (CRA) used to be standard, but it's largely been superseded. **Vite** is the modern, much faster, and generally recommended way to bootstrap React projects. It's simpler and quicker to get started with. So, Vite is the choice.

4.  **Prerequisites:** What does someone need *before* starting?
    *   **Node.js and npm/yarn:** Essential for any JavaScript development, including React/Vite. Explain how to check and install.
    *   **Text Editor:** Standard requirement (VS Code recommended).
    *   **Basic Web Knowledge:** HTML, CSS, JS (as specified by the user).
    *   **RAG LLM Server Running:** The backend needs to be accessible. Mention this and its default URL.

5.  **Step-by-Step Setup using Vite:**
    *   **Command:** Provide the `npm create vite@latest` command. Explain the prompts (project name, framework=React, variant=JavaScript).
    *   **Navigate:** `cd project-name`.
    *   **Install Dependencies:** `npm install`.
    *   **Start Development Server:** `npm run dev`. Show the expected output and URL.

6.  **Project Structure (Initial):** Explain the basic files/folders created by Vite (`public`, `src`, `index.html`, `package.json`). Focus on `src` where most work will happen.

7.  **Component Creation Strategy:**
    *   Start with the main `App.jsx`. Clean up the default Vite template.
    *   Create a `components` folder within `src`.
    *   Create separate folders for each core component identified in step 2 (e.g., `ChatWindow`, `MessageInput`, `MessageList`, `MessageBubble`).
    *   Inside each component folder, create:
        *   `ComponentName.jsx`: The React component code.
        *   `ComponentName.module.css`: The CSS Module file for component-specific styling. Explain *why* CSS Modules are a good choice here (local scope, avoids global conflicts, fits the "custom CSS per component" requirement without frameworks).

8.  **Implement Basic Components (with Placeholders):**
    *   **`App.jsx`:** Set up basic state for messages and input value. Include placeholders for child components. Add basic layout CSS.
    *   **`ChatWindow.jsx/.module.css`:** Basic container, pass messages down. Style the window.
    *   **`MessageList.jsx/.module.css`:** Map over messages, render `MessageBubble`. Style the list area (scrolling).
    *   **`MessageBubble.jsx/.module.css`:** Display message content and role. Style user vs. assistant messages differently.
    *   **`MessageInput.jsx/.module.css`:** Controlled input field and button. Handle input changes and send action (initially just console log). Style the input area.

9.  **State Management:** Use simple `useState` hooks in `App.jsx` for initial message list and input field value. Explain lifting state up.

10. **Handling User Input & Sending:**
    *   Implement `handleInputChange` and `handleSendMessage` functions in `App.jsx`.
    *   Pass these functions down as props to `MessageInput`.
    *   In `handleSendMessage`, initially just:
        *   Add the user's message to the `messages` state array.
        *   Clear the input field.
        *   (Placeholder for API call).

11. **CSS Styling:**
    *   Emphasize creating corresponding `.module.css` files.
    *   Show how to import and use CSS Modules (`import styles from './...'`, `className={styles.someClass}`).
    *   Provide basic, functional CSS for each component (layout, colors, message alignment, scrolling). Keep it generic and minimal as requested.

12. **Connecting to the Backend (Next Step):** Explicitly state that the next logical step is to replace the placeholder send logic with an actual `fetch` (or `axios`) call to the RAG LLM server's chat endpoint (`POST /api/v1/sessions/{session_id}/messages`). Mention the need for session management on the frontend.

13. **Review and Refine:** Read through the generated guide.
    *   Is it clear for a beginner?
    *   Are all setup steps included?
    *   Does it meet all user requirements (React, Vite, no CSS frameworks, custom CSS)?
    *   Is the code functional (even with placeholders)?
    *   Is the explanation logical?
    *   (Self-correction): Initially, I might forget to explicitly mention CSS Modules, but realizing the "no frameworks" and "custom CSS per component" requirements makes them the ideal solution to introduce. Ensure the import/usage syntax is shown. Ensure basic styling makes the app *look* like a chat app immediately. Ensure the flow of props (state and handlers) is explained.

***   

**[Setup 1: Environment Setup Quick Start Guide](wechat/instructions/setup_1_environment_quick_setup.md)**
