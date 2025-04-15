// src/components/SessionManager/SessionManager.jsx
import React from 'react';
import styles from './SessionManager.module.css';

function SessionManager({
    savedSessions,
    activeSessionId,
    onLoadSession,
    onSaveSession,
    onDeleteSession,
    onNewChat, // Receive new chat handler
    disabled
}) {

    return (
        <div className={styles.sessionManager}>
            <div className={styles.controls}>
                <button onClick={onNewChat} disabled={disabled} title="Start a new empty chat">
                    New Chat
                </button>
                <button onClick={onSaveSession} disabled={disabled} title="Save the current chat session">
                    Save Current
                </button>
            </div>
            <div className={styles.sessionListContainer}>
                <h4>Saved Sessions</h4>
                {savedSessions.length === 0 ? (
                    <p className={styles.noSessions}>No saved sessions yet.</p>
                ) : (
                    <ul className={styles.sessionList}>
                        {savedSessions.map(session => (
                            <li
                                key={session.id}
                                className={`${styles.sessionItem} ${session.id === activeSessionId ? styles.active : ''}`}
                            >
                                <span
                                    className={styles.sessionName}
                                    onClick={() => !disabled && onLoadSession(session.id)} // Load on click
                                    title={`Load "${session.name}" (Last saved: ${new Date(session.last_updated_at).toLocaleString()})`}
                                >
                                    {session.name}
                                </span>
                                <button
                                    className={styles.deleteButton}
                                    onClick={() => !disabled && onDeleteSession(session.id)}
                                    disabled={disabled}
                                    title="Delete session"
                                >
                                    × {/* Simple 'x' delete symbol */}
                                </button>
                            </li>
                        ))}
                    </ul>
                )}
            </div>
        </div>
    );
}

export default SessionManager;