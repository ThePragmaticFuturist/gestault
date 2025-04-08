# Step 2: Create Component Structure

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
