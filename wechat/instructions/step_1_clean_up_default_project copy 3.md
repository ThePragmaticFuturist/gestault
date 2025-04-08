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
