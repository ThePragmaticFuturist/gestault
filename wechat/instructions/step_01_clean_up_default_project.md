# Step 1: Clean Up Default Project

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
