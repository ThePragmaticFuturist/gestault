**Future Features for WeChat React App**

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
