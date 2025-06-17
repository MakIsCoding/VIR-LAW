// src/components/QueryPage.jsx
import React, { useState, useEffect, useRef } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { auth, db } from "../firebase";
import {
  collection,
  addDoc,
  query,
  orderBy,
  onSnapshot,
  serverTimestamp,
  doc,
  updateDoc,
  getDoc,
} from "firebase/firestore";
import axios from "axios"; // Import axios for making HTTP requests to your Flask backend

const QueryPage = () => {
  const { queryId } = useParams();
  const navigate = useNavigate();

  const [input, setInput] = useState("");
  const [messages, setMessages] = useState([]);
  const [isLoadingResponse, setIsLoadingResponse] = useState(false); // Controls AI thinking state
  const [sessionTitle, setSessionTitle] = useState("New Chat"); // Default title, should match Sidebar's initial title
  const [isSessionLoading, setIsSessionLoading] = useState(true); // For initial session data load
  const [sessionLoadError, setSessionLoadError] = useState(null); // For session loading errors
  const [sendMessageError, setSendMessageError] = useState(null); // New state for send message errors
  const messagesEndRef = useRef(null);

  // New state to temporarily hold a message that needs to be sent AFTER a new session is created
  const [pendingMessage, setPendingMessage] = useState(null);

  const currentUserId = auth.currentUser?.uid; // Get UID safely

  // Effect to manage session data based on URL's queryId
  useEffect(() => {
    // Reset state for a clean slate whenever queryId changes
    setMessages([]);
    setInput("");
    setIsLoadingResponse(false); // Reset AI loading state
    setIsSessionLoading(true);
    setSessionLoadError(null);
    setSendMessageError(null); // Also reset send message error on queryId change

    // If no user is logged in, or if queryId is missing/invalid, do nothing or redirect
    if (!currentUserId) {
      setSessionLoadError("User not authenticated.");
      setIsSessionLoading(false);
      return;
    }

    // Case 1: Starting a new query (URL is /dashboard/new) - temporary state
    if (queryId === "new") {
      setSessionTitle("New Chat"); // Initial title for this temporary state
      setMessages([]); // Ensure messages are empty for a fresh start
      setIsSessionLoading(false); // No data to load for a new session
      return; // Exit as no Firestore listener is needed yet for a /new route
    }

    // Case 2: Loading an existing query (URL is /dashboard/:queryId)
    if (queryId) {
      const sessionDocRef = doc(
        db,
        "users",
        currentUserId,
        "querySessions",
        queryId
      );

      // Listener for the session document itself (for title, existence)
      const unsubscribeSession = onSnapshot(
        sessionDocRef,
        (docSnap) => {
          if (docSnap.exists()) {
            setSessionTitle(
              docSnap.data().title || `Session ${queryId.substring(0, 8)}`
            );
            setSessionLoadError(null);
          } else {
            // If the document doesn't exist, handle it (e.g., redirect to new query)
            console.warn(
              `Firestore document for session ID "${queryId}" not found for user "${currentUserId}".`
            );
            setSessionLoadError("Session not found.");
            setSessionTitle("Session Not Found");
            setMessages([]); // Clear any old messages
            // Optionally, navigate to a new query if an invalid ID is in the URL
            navigate("/dashboard/new"); // Redirect to a new temporary session
          }
          setIsSessionLoading(false); // Session info loaded (or determined not to exist)
        },
        (error) => {
          console.error("Error fetching session document:", error);
          setSessionLoadError("Failed to load session details.");
          setIsSessionLoading(false);
        }
      );

      // Listener for messages within this session
      const messagesCollectionRef = collection(sessionDocRef, "messages");
      const q = query(messagesCollectionRef, orderBy("createdAt"));

      const unsubscribeMessages = onSnapshot(
        q,
        (snapshot) => {
          const loadedMessages = snapshot.docs.map((doc) => ({
            id: doc.id, // Firestore message ID
            ...doc.data(), // message text, sender, createdAt
          }));
          setMessages(loadedMessages);
          console.log(
            "Firestore onSnapshot updated messages state:",
            loadedMessages
          ); // Console: Verify messages loaded from Firestore
        },
        (error) => {
          console.error("Error fetching messages for session:", error);
        }
      );

      // Cleanup listeners when component unmounts or queryId/user changes
      return () => {
        unsubscribeSession();
        unsubscribeMessages();
      };
    }

    // Default case if no queryId is provided (e.g., direct /dashboard access without /new or an ID)
    // We treat this as a new query
    setSessionTitle("New Chat"); // Ensure initial title matches Sidebar's.
    setMessages([]);
    setIsSessionLoading(false);
  }, [queryId, currentUserId, db, navigate]); // Depend on queryId, userId, db, and navigate

  // Effect for auto-scrolling to the latest message
  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
    console.log("Current messages state (after render/scroll):", messages); // Console: Verify current messages state
  }, [messages]); // Scroll when messages change

  // NEW useEffect to handle sending the pending message once queryId is stable
  useEffect(() => {
    // Only proceed if there's a pending message, queryId is valid (not 'new' or undefined), and session isn't loading
    if (
      pendingMessage &&
      queryId !== "new" &&
      queryId !== undefined &&
      !isSessionLoading &&
      currentUserId
    ) {
      console.log(
        "DEBUG useEffect: Processing pending message for new session:",
        queryId
      );
      const sendMessageToFirestore = async () => {
        try {
          const sessionDocRef = doc(
            db,
            "users",
            currentUserId,
            "querySessions",
            queryId // Now queryId is guaranteed to be the actual session ID
          );

          const docSnap = await getDoc(sessionDocRef);
          console.log("DEBUG useEffect: docSnap.exists():", docSnap.exists());
          if (docSnap.exists()) {
            console.log(
              "DEBUG useEffect: docSnap.data().title:",
              docSnap.data().title
            );
            if (docSnap.data().title === "New Chat") {
              const updatedTitle =
                pendingMessage.substring(0, 50) +
                (pendingMessage.length > 50 ? "..." : "");
              await updateDoc(sessionDocRef, {
                title: updatedTitle,
                lastUpdated: serverTimestamp(),
              });
              setSessionTitle(updatedTitle);
              console.log(
                "DEBUG useEffect: Session title updated in useEffect to:",
                updatedTitle
              );
            } else {
              await updateDoc(sessionDocRef, {
                lastUpdated: serverTimestamp(),
              });
              console.log(
                "DEBUG useEffect: Session lastUpdated field updated in useEffect."
              );
            }
          }

          // Add the user's message to the messages subcollection in Firestore
          await addDoc(
            collection(
              db,
              "users",
              currentUserId,
              "querySessions",
              queryId,
              "messages"
            ),
            {
              text: pendingMessage,
              sender: "user",
              createdAt: serverTimestamp(),
            }
          );
          console.log(
            "DEBUG useEffect: User message added to Firestore via useEffect."
          );

          // --- START RAG INTEGRATION (Moved here) ---
          setIsLoadingResponse(true);
          console.log(
            "DEBUG useEffect: AI typing indicator set to true in useEffect."
          );

          let aiResponseText =
            "An error occurred while getting a response from VirLaw AI.";
          try {
            console.log(
              "DEBUG useEffect: Sending prompt to Flask backend from useEffect:",
              pendingMessage
            );
            const ragResponse = await axios.post(
              "http://localhost:8000/gemini-rag",
              {
                prompt: pendingMessage,
              }
            );
            aiResponseText = ragResponse.data.response;
            console.log(
              "DEBUG useEffect: AI response received from Flask in useEffect:",
              aiResponseText
            );
          } catch (ragError) {
            console.error(
              "ERROR useEffect: Error calling Python RAG API in useEffect:",
              ragError
            );
            if (ragError.response) {
              aiResponseText = `VirLaw AI: Failed to get a response (Code: ${ragError.response.status}). Please check the Python backend.`;
            } else if (ragError.request) {
              aiResponseText =
                "VirLaw AI: No response from the AI server. Is the Python backend running?";
            } else {
              aiResponseText = `VirLaw AI: Error sending request: ${ragError.message}`;
            }
            setSendMessageError(aiResponseText);
          } finally {
            setIsLoadingResponse(false);
            console.log(
              "DEBUG useEffect: AI typing indicator set to false in useEffect."
            );
          }

          await addDoc(
            collection(
              db,
              "users",
              currentUserId,
              "querySessions",
              queryId,
              "messages"
            ),
            {
              text: aiResponseText,
              sender: "bot",
              createdAt: serverTimestamp(),
            }
          );
          console.log(
            "DEBUG useEffect: AI message added to Firestore via useEffect."
          );
          // --- END RAG INTEGRATION ---
        } catch (error) {
          console.error(
            "ERROR useEffect: Error processing pending message in useEffect:",
            error
          );
          setSendMessageError(
            "Failed to send message or save session after creation. Please try again."
          );
          setIsLoadingResponse(false);
        } finally {
          setPendingMessage(null); // Clear the pending message after processing
        }
      };

      sendMessageToFirestore();
    }
  }, [pendingMessage, queryId, isSessionLoading, currentUserId, db, navigate]); // Add db and navigate to dependencies

  // Handler for sending a message (user or AI)
  const handleSendMessage = async (e) => {
    e.preventDefault();
    if (!input.trim() || !currentUserId) {
      if (!currentUserId) {
        setSendMessageError("You must be logged in to send messages.");
      }
      return;
    }

    const userMessageText = input;
    console.log(
      "User input captured:",
      userMessageText,
      typeof userMessageText
    );
    setInput("");
    setSendMessageError(null);

    // Optimistic UI update for the user's message
    const tempUserMessage = {
      id: "temp-" + Date.now(),
      text: userMessageText,
      sender: "user",
      createdAt: new Date(),
    };
    setMessages((prevMessages) => [...prevMessages, tempUserMessage]);
    console.log("Optimistic UI: User message added to local state.");

    // If it's a new session, create it and navigate. The rest will be handled by useEffect.
    if (!queryId || queryId === "new") {
      console.log(
        "DEBUG handleSendMessage: Entered 'create new session' block."
      ); // NEW LOG
      try {
        console.log(
          "DEBUG handleSendMessage: Attempting to add new session document to Firestore..."
        ); // NEW LOG
        const newSessionRef = await addDoc(
          collection(db, "users", currentUserId, "querySessions"),
          {
            title: "New Chat",
            createdAt: serverTimestamp(),
            lastUpdated: serverTimestamp(),
          }
        );
        const newSessionId = newSessionRef.id;
        console.log(
          "DEBUG handleSendMessage: New session document added. ID:",
          newSessionId
        ); // NEW LOG

        setPendingMessage(userMessageText); // Store the message to be processed after navigation
        console.log(
          "DEBUG handleSendMessage: Pending message set. About to navigate."
        ); // NEW LOG
        navigate(`/dashboard/${newSessionId}`, { replace: true }); // Navigate to the new session
        console.log(
          "DEBUG handleSendMessage: Navigation initiated to new session ID:",
          newSessionId
        ); // This might not appear if navigation causes an immediate unmount
      } catch (error) {
        console.error(
          "ERROR handleSendMessage: Failed to create new session:",
          error
        ); // NEW LOG
        setSendMessageError("Failed to create new session. Please try again.");
        setIsLoadingResponse(false);
        // Remove the optimistically added message if session creation failed
        setMessages((prevMessages) =>
          prevMessages.filter((msg) => msg.id !== tempUserMessage.id)
        );
      }
      return; // Exit here, useEffect will handle message sending after navigation
    }

    // If it's an existing session, proceed directly to send the message
    // This part is for subsequent messages in an already existing session
    console.log(
      "DEBUG handleSendMessage: Handling message for existing session."
    ); // NEW LOG
    let currentSessionFirestoreId = queryId; // This should already be valid

    try {
      const sessionDocRef = doc(
        db,
        "users",
        currentUserId,
        "querySessions",
        currentSessionFirestoreId
      );

      const docSnap = await getDoc(sessionDocRef);
      console.log("docSnap.exists():", docSnap.exists());
      if (docSnap.exists()) {
        console.log("docSnap.data():", docSnap.data());
        console.log(
          "docSnap.data().title:",
          docSnap.data().title,
          typeof docSnap.data().title
        );

        if (docSnap.data().title === "New Chat") {
          // Should ideally not happen for existing sessions, but as a fallback
          const updatedTitle =
            userMessageText.substring(0, 50) +
            (userMessageText.length > 50 ? "..." : "");
          await updateDoc(sessionDocRef, {
            title: updatedTitle,
            lastUpdated: serverTimestamp(),
          });
          setSessionTitle(updatedTitle);
          console.log("Session title updated to:", updatedTitle);
        } else {
          await updateDoc(sessionDocRef, { lastUpdated: serverTimestamp() });
          console.log("Session lastUpdated field updated.");
        }
      }

      // Add the user's message to Firestore for existing session
      await addDoc(
        collection(
          db,
          "users",
          currentUserId,
          "querySessions",
          currentSessionFirestoreId,
          "messages"
        ),
        {
          text: userMessageText,
          sender: "user",
          createdAt: serverTimestamp(),
        }
      );
      console.log("User message added to Firestore for existing session.");

      // --- START RAG INTEGRATION (for existing sessions) ---
      setIsLoadingResponse(true);
      console.log("AI typing indicator set to true.");

      let aiResponseText =
        "An error occurred while getting a response from VirLaw AI.";
      try {
        console.log("Sending prompt to Flask backend:", userMessageText);
        const ragResponse = await axios.post(
          "http://localhost:8000/gemini-rag",
          {
            prompt: userMessageText,
          }
        );
        aiResponseText = ragResponse.data.response;
        console.log("AI response received from Flask:", aiResponseText);
      } catch (ragError) {
        console.error("Error calling Python RAG API:", ragError);
        if (ragError.response) {
          aiResponseText = `VirLaw AI: Failed to get a response (Code: ${ragError.response.status}). Please check the Python backend.`;
        } else if (ragError.request) {
          aiResponseText =
            "VirLaw AI: No response from the AI server. Is the Python backend running?";
        } else {
          aiResponseText = `VirLaw AI: Error sending request: ${ragError.message}`;
        }
        setSendMessageError(aiResponseText);
        console.error(
          "Error with RAG API call, AI response set to:",
          aiResponseText
        );
      } finally {
        setIsLoadingResponse(false);
        console.log("AI typing indicator set to false.");
      }

      await addDoc(
        collection(
          db,
          "users",
          currentUserId,
          "querySessions",
          currentSessionFirestoreId,
          "messages"
        ),
        {
          text: aiResponseText,
          sender: "bot",
          createdAt: serverTimestamp(),
        }
      );
      console.log("AI message added to Firestore.");
      // --- END RAG INTEGRATION ---
    } catch (error) {
      console.error(
        "Error in handleSendMessage (Firebase or initial setup) for existing session:",
        error
      );
      setIsLoadingResponse(false);
      setSendMessageError(
        "Failed to send message or save session. Please try again."
      );
      // Remove the optimistically added message if sending failed
      setMessages((prevMessages) =>
        prevMessages.filter((msg) => msg.id !== tempUserMessage.id)
      );
    }
  };

  // Render loading state, error state, or the chat UI
  if (isSessionLoading) {
    return <div className="p-4 text-gray-100">Loading session...</div>;
  }

  // Session load error takes precedence
  if (sessionLoadError) {
    return (
      <div className="p-4 text-red-400">
        Error: {sessionLoadError}
        <button
          onClick={() => navigate("/dashboard/new")}
          className="ml-4 bg-blue-600 hover:bg-blue-700 text-white py-1 px-3 rounded"
        >
          Start New Query
        </button>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full bg-gray-800 text-gray-100">
      {/* Session Title Bar */}
      <div className="p-4 bg-gray-900 shadow-md flex items-center justify-between">
        <h2 className="text-2xl font-bold text-blue-300">{sessionTitle}</h2>
      </div>

      {/* Messages Display Area */}
      <div className="flex-1 overflow-y-auto p-4 flex flex-col custom-scrollbar">
        {/* On-screen error message for sending */}
        {sendMessageError && (
          <div className="p-2 mb-4 bg-red-800 text-red-200 rounded-md text-center">
            {sendMessageError}
          </div>
        )}

        {messages.length === 0 &&
        !isLoadingResponse &&
        (queryId === "new" || !queryId) ? (
          <div className="text-center text-gray-400 mt-20">
            Start a new conversation...
          </div>
        ) : messages.length === 0 && !isLoadingResponse && queryId !== "new" ? (
          <div className="text-center text-gray-400 mt-20">
            No messages in this session yet. Type a message below!
          </div>
        ) : (
          messages.map((msg) => (
            <div
              key={msg.id} // Use Firestore document ID or temporary ID as key for messages
              className={`mb-4 p-3 rounded-lg max-w-[80%] ${
                msg.sender === "user"
                  ? "bg-blue-600 self-end ml-auto" // User messages on right
                  : "bg-gray-700 self-start mr-auto" // AI messages on left
              }`}
            >
              <p className="text-sm">{msg.text}</p>
            </div>
          ))
        )}
        {isLoadingResponse && (
          <div className="mb-4 p-3 rounded-lg bg-gray-700 self-start mr-auto max-w-[80%]">
            <p className="text-sm">VirLaw AI is typing...</p>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <form
        onSubmit={handleSendMessage}
        className="p-4 bg-gray-900 border-t border-gray-700 flex items-center flex-shrink-0"
      >
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Type your message..."
          className="flex-1 p-3 rounded-lg bg-gray-700 border border-gray-600 text-gray-100 focus:outline-none focus:ring-2 focus:ring-blue-500"
          // Input should only be disabled if AI is loading (isLoadingResponse is true) or no user
          disabled={isLoadingResponse || !currentUserId}
        />
        <button
          type="submit"
          className="ml-4 px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
          // Button should be disabled if AI is loading OR input is empty OR no user
          disabled={isLoadingResponse || !input.trim() || !currentUserId}
        >
          Send
        </button>
      </form>
    </div>
  );
};

export default QueryPage;
