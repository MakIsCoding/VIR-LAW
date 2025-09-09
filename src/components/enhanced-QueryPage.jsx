// FIXED enhanced-QueryPage.jsx - All advanced features preserved with fixes
import React, { useState, useEffect, useRef } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { auth, db } from "../firebase";
import {
  collection, addDoc, query, orderBy, onSnapshot, serverTimestamp, doc,
  updateDoc, getDoc
} from "firebase/firestore";
import axios from "axios";
import {
  DocumentTextIcon, TableCellsIcon, PhotoIcon, ScaleIcon, ClockIcon,
  CheckCircleIcon, ExclamationTriangleIcon, InformationCircleIcon,
  ChartBarIcon, CogIcon, ArrowPathIcon
} from "@heroicons/react/24/solid";

const QueryPage = () => {
  const { queryId } = useParams();
  const navigate = useNavigate();

  // Enhanced state management
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState([]);
  const [isLoadingResponse, setIsLoadingResponse] = useState(false);
  const [sessionTitle, setSessionTitle] = useState("New Chat");
  const [isSessionLoading, setIsSessionLoading] = useState(true);
  const [sessionLoadError, setSessionLoadError] = useState(null);
  const [sendMessageError, setSendMessageError] = useState(null);
  const [pendingMessage, setPendingMessage] = useState(null);

  // ULTIMATE new state for advanced features
  const [queryType, setQueryType] = useState("general_analysis");
  const [availableQueryTypes, setAvailableQueryTypes] = useState({});
  const [includeSourcesMode, setIncludeSourcesMode] = useState(true);
  const [lastQueryStats, setLastQueryStats] = useState(null);
  const [systemHealth, setSystemHealth] = useState(null);
  const [currentQueryId, setCurrentQueryId] = useState(null);
  const [confidenceThreshold, setConfidenceThreshold] = useState(0.7);
  const [showAdvancedOptions, setShowAdvancedOptions] = useState(false);
  const [realTimeMetrics, setRealTimeMetrics] = useState({});

  const messagesEndRef = useRef(null);
  const currentUserId = auth.currentUser?.uid;

  // Enhanced useEffect for advanced initialization
  useEffect(() => {
    initializeAdvancedFeatures();
    setupRealTimeMonitoring();
  }, []);

  // Load query types and system health
  const initializeAdvancedFeatures = async () => {
    try {
      // Try to load available query types - graceful degradation if not available
      try {
        const queryTypesResponse = await axios.get("http://localhost:8000/query-types");
        setAvailableQueryTypes(queryTypesResponse.data.query_types || {
          "general_analysis": { name: "General Analysis", description: "General legal analysis" },
          "case_law": { name: "Case Law", description: "Case law research" },
          "statutory": { name: "Statutory", description: "Statutory interpretation" }
        });
      } catch (error) {
        console.warn("Query types not available, using defaults");
        setAvailableQueryTypes({
          "general_analysis": { name: "General Analysis", description: "General legal analysis" },
          "case_law": { name: "Case Law", description: "Case law research" },
          "statutory": { name: "Statutory", description: "Statutory interpretation" }
        });
      }

      // Load system health
      try {
        const healthResponse = await axios.get("http://localhost:8000/health");
        setSystemHealth(healthResponse.data);
      } catch (error) {
        console.warn("System health check failed");
      }
    } catch (error) {
      console.error("Error initializing advanced features:", error);
    }
  };

  // Setup real-time monitoring
  const setupRealTimeMonitoring = () => {
    const interval = setInterval(async () => {
      try {
        const statsResponse = await axios.get("http://localhost:8000/system-stats");
        setRealTimeMetrics(statsResponse.data);
      } catch (error) {
        console.error("Error fetching real-time metrics:", error);
      }
    }, 30000); // Update every 30 seconds

    return () => clearInterval(interval);
  };

  // Original session management useEffect (enhanced)
  useEffect(() => {
    setMessages([]);
    setInput("");
    setIsLoadingResponse(false);
    setIsSessionLoading(true);
    setSessionLoadError(null);
    setSendMessageError(null);
    setLastQueryStats(null);

    if (!currentUserId) {
      setSessionLoadError("User not authenticated.");
      setIsSessionLoading(false);
      return;
    }

    if (queryId === "new") {
      setSessionTitle("New Chat");
      setMessages([]);
      setIsSessionLoading(false);
      return;
    }

    if (queryId) {
      const sessionDocRef = doc(db, "users", currentUserId, "querySessions", queryId);
      
      const unsubscribeSession = onSnapshot(sessionDocRef, (docSnap) => {
        if (docSnap.exists()) {
          setSessionTitle(docSnap.data().title || `Session ${queryId.substring(0, 8)}`);
          setSessionLoadError(null);
        } else {
          console.warn(`Firestore document for session ID "${queryId}" not found.`);
          setSessionLoadError("Session not found.");
          setSessionTitle("Session Not Found");
          setMessages([]);
          navigate("/dashboard/new");
        }
        setIsSessionLoading(false);
      }, (error) => {
        console.error("Error fetching session document:", error);
        setSessionLoadError("Failed to load session details.");
        setIsSessionLoading(false);
      });

      const messagesCollectionRef = collection(sessionDocRef, "messages");
      const q = query(messagesCollectionRef, orderBy("createdAt"));
      const unsubscribeMessages = onSnapshot(q, (snapshot) => {
        const loadedMessages = snapshot.docs.map((doc) => ({
          id: doc.id,
          ...doc.data(),
        }));
        setMessages(loadedMessages);
      }, (error) => {
        console.error("Error fetching messages for session:", error);
      });

      return () => {
        unsubscribeSession();
        unsubscribeMessages();
      };
    }

    setSessionTitle("New Chat");
    setMessages([]);
    setIsSessionLoading(false);
  }, [queryId, currentUserId, navigate]);

  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages]);

  // Enhanced pending message handler
  useEffect(() => {
    if (pendingMessage && queryId !== "new" && queryId !== undefined && !isSessionLoading && currentUserId) {
      const sendMessageToFirestore = async () => {
        try {
          const sessionDocRef = doc(db, "users", currentUserId, "querySessions", queryId);
          const docSnap = await getDoc(sessionDocRef);

          if (docSnap.exists()) {
            if (docSnap.data().title === "New Chat") {
              const updatedTitle = pendingMessage.substring(0, 50) +
                (pendingMessage.length > 50 ? "..." : "");
              await updateDoc(sessionDocRef, {
                title: updatedTitle,
                lastUpdated: serverTimestamp(),
              });
              setSessionTitle(updatedTitle);
            } else {
              await updateDoc(sessionDocRef, {
                lastUpdated: serverTimestamp()
              });
            }
          }

          await addDoc(collection(db, "users", currentUserId, "querySessions", queryId, "messages"), {
            text: pendingMessage,
            sender: "user",
            createdAt: serverTimestamp(),
          });

          // ULTIMATE RAG Integration with all advanced features
          await processUltimateQuery(pendingMessage, queryId);
        } catch (error) {
          console.error("Error processing pending message:", error);
          setSendMessageError("Failed to send message. Please try again.");
          setIsLoadingResponse(false);
        } finally {
          setPendingMessage(null);
        }
      };

      sendMessageToFirestore();
    }
  }, [pendingMessage, queryId, isSessionLoading, currentUserId]);

  // ULTIMATE query processing function
  const processUltimateQuery = async (userMessage, sessionId) => {
    setIsLoadingResponse(true);
    setSendMessageError(null);
    const queryStartTime = Date.now();
    
    let aiResponseText = "An error occurred while getting a response from VirLaw AI.";
    let queryStats = null;
    let sources = [];
    let metadata = {};

    try {
      console.log("🚀 Processing ULTIMATE query:", userMessage);

      // Use enhanced endpoint with all advanced features
      const requestData = {
        prompt: userMessage
      };

      // Add advanced parameters if supported
      if (includeSourcesMode) requestData.include_sources = true;
      if (queryType !== "general_analysis") requestData.query_type = queryType;
      if (confidenceThreshold !== 0.7) requestData.confidence_threshold = confidenceThreshold;

      const ragResponse = await axios.post("http://localhost:8000/gemini-rag", requestData);
      const data = ragResponse.data;

      aiResponseText = data.response;
      sources = data.sources || [];
      metadata = data.metadata || {};

      // Calculate query statistics
      const processingTime = Date.now() - queryStartTime;
      queryStats = {
        processing_time_ms: processingTime,
        confidence: data.confidence || 0.8,
        sources_found: sources.length,
        query_id: data.query_id,
        query_type: queryType,
        model_used: metadata.model_used || "unknown",
        timestamp: new Date().toISOString()
      };

      setLastQueryStats(queryStats);
      setCurrentQueryId(data.query_id);

      console.log("✅ ULTIMATE response received:", {
        response_length: aiResponseText.length,
        sources_count: sources.length,
        confidence: data.confidence,
        processing_time: processingTime
      });

    } catch (ragError) {
      console.error("Error calling ULTIMATE RAG API:", ragError);
      const processingTime = Date.now() - queryStartTime;
      
      queryStats = {
        processing_time_ms: processingTime,
        confidence: 0.0,
        sources_found: 0,
        error: true,
        error_type: ragError.response?.status || "network_error",
        timestamp: new Date().toISOString()
      };

      if (ragError.response) {
        aiResponseText = `VirLaw AI: Failed to get a response (Code: ${ragError.response.status}). Please check the backend connection.`;
      } else if (ragError.request) {
        aiResponseText = "VirLaw AI: No response from the AI server. Please ensure the backend is running.";
      } else {
        aiResponseText = `VirLaw AI: Error sending request: ${ragError.message}`;
      }

      setSendMessageError(aiResponseText);
      setLastQueryStats(queryStats);
    } finally {
      setIsLoadingResponse(false);
    }

    // Store enhanced AI response with metadata
    try {
      await addDoc(collection(db, "users", currentUserId, "querySessions", sessionId, "messages"), {
        text: aiResponseText,
        sender: "bot",
        createdAt: serverTimestamp(),
        metadata: {
          sources: sources,
          query_stats: queryStats,
          query_type: queryType,
          confidence: queryStats?.confidence || 0.0,
          query_id: currentQueryId
        }
      });
    } catch (error) {
      console.error("Error saving AI response to Firestore:", error);
    }
  };

  // Enhanced message sending handler
  const handleSendMessage = async (e) => {
    e.preventDefault();
    if (!input.trim() || !currentUserId) {
      if (!currentUserId) {
        setSendMessageError("You must be logged in to send messages.");
      }
      return;
    }

    const userMessageText = input;
    setInput("");
    setSendMessageError(null);

    // Optimistic UI update
    const tempUserMessage = {
      id: "temp-" + Date.now(),
      text: userMessageText,
      sender: "user",
      createdAt: new Date(),
    };
    setMessages((prevMessages) => [...prevMessages, tempUserMessage]);

    if (!queryId || queryId === "new") {
      try {
        const newSessionRef = await addDoc(
          collection(db, "users", currentUserId, "querySessions"),
          {
            title: "New Chat",
            createdAt: serverTimestamp(),
            lastUpdated: serverTimestamp(),
          }
        );
        const newSessionId = newSessionRef.id;
        setPendingMessage(userMessageText);
        navigate(`/dashboard/${newSessionId}`, { replace: true });
      } catch (error) {
        console.error("Failed to create new session:", error);
        setSendMessageError("Failed to create new session. Please try again.");
        setIsLoadingResponse(false);
        setMessages((prevMessages) =>
          prevMessages.filter((msg) => msg.id !== tempUserMessage.id)
        );
      }
      return;
    }

    // Handle existing session
    try {
      const sessionDocRef = doc(db, "users", currentUserId, "querySessions", queryId);
      const docSnap = await getDoc(sessionDocRef);

      if (docSnap.exists()) {
        if (docSnap.data().title === "New Chat") {
          const updatedTitle = userMessageText.substring(0, 50) +
            (userMessageText.length > 50 ? "..." : "");
          await updateDoc(sessionDocRef, {
            title: updatedTitle,
            lastUpdated: serverTimestamp(),
          });
          setSessionTitle(updatedTitle);
        } else {
          await updateDoc(sessionDocRef, {
            lastUpdated: serverTimestamp()
          });
        }
      }

      await addDoc(collection(db, "users", currentUserId, "querySessions", queryId, "messages"), {
        text: userMessageText,
        sender: "user",
        createdAt: serverTimestamp(),
      });

      // Process with ULTIMATE features
      await processUltimateQuery(userMessageText, queryId);
    } catch (error) {
      console.error("Error in handleSendMessage:", error);
      setIsLoadingResponse(false);
      setSendMessageError("Failed to send message. Please try again.");
      setMessages((prevMessages) =>
        prevMessages.filter((msg) => msg.id !== tempUserMessage.id)
      );
    }
  };

  // Enhanced message rendering with advanced features
  const renderMessage = (msg, index) => {
    const isUser = msg.sender === "user";
    const hasMetadata = msg.metadata && !isUser;
    const sources = hasMetadata ? msg.metadata.sources || [] : [];
    const queryStats = hasMetadata ? msg.metadata.query_stats : null;
    const confidence = hasMetadata ? msg.metadata.confidence || 0 : null;

    return (
      <div key={msg.id || index} className={`flex ${isUser ? "justify-end" : "justify-start"} mb-4`}>
        <div className={`max-w-3xl ${isUser ? "bg-blue-600 text-white" : "bg-gray-100 text-gray-800"} rounded-lg px-4 py-2`}>
          <p className="text-sm whitespace-pre-wrap">{msg.text}</p>
          
          {/* AI message with advanced features */}
          {!isUser && hasMetadata && (
            <div className="mt-3 pt-3 border-t border-gray-200">
              {/* Confidence Score */}
              {confidence !== null && (
                <div className="flex items-center space-x-2 mb-2">
                  <span className="text-xs font-medium text-gray-600">Confidence:</span>
                  <div className="flex items-center space-x-1">
                    <div className="w-16 bg-gray-200 rounded-full h-1.5">
                      <div
                        className={`h-1.5 rounded-full ${confidence >= 0.8 ? 'bg-green-500' : confidence >= 0.6 ? 'bg-yellow-500' : 'bg-red-500'}`}
                        style={{ width: `${confidence * 100}%` }}
                      />
                    </div>
                    <span className="text-xs text-gray-500">{Math.round(confidence * 100)}%</span>
                  </div>
                </div>
              )}

              {/* Sources */}
              {sources.length > 0 && (
                <div className="mb-2">
                  <span className="text-xs font-medium text-gray-600">Sources: </span>
                  <span className="text-xs text-gray-500">{sources.length} document(s)</span>
                </div>
              )}

              {/* Query Stats */}
              {queryStats && (
                <div className="flex items-center space-x-4 text-xs text-gray-500">
                  <span>⏱ {queryStats.processing_time_ms}ms</span>
                  <span>🔍 {queryStats.query_type}</span>
                  {queryStats.model_used && <span>🤖 {queryStats.model_used}</span>}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    );
  };

  // Loading and error states
  if (isSessionLoading) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <p className="text-gray-600">Loading session...</p>
        </div>
      </div>
    );
  }

  if (sessionLoadError) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <div className="text-center">
          <ExclamationTriangleIcon className="h-12 w-12 text-red-500 mx-auto mb-4" />
          <p className="text-red-600 mb-4">{sessionLoadError}</p>
          <button
            onClick={() => navigate("/dashboard/new")}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
          >
            Start New Query
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 flex flex-col h-full">
      {/* Enhanced Header */}
      <div className="bg-white border-b border-gray-200 px-6 py-4">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-xl font-semibold text-gray-800">{sessionTitle}</h1>
            <p className="text-sm text-gray-600">Enhanced Legal AI Assistant</p>
          </div>
          
          {/* Advanced Options Toggle */}
          <button
            onClick={() => setShowAdvancedOptions(!showAdvancedOptions)}
            className="flex items-center space-x-2 px-3 py-1 text-sm bg-gray-100 text-gray-600 rounded hover:bg-gray-200"
          >
            <CogIcon className="h-4 w-4" />
            <span>Advanced</span>
          </button>
        </div>

        {/* Advanced Options Panel */}
        {showAdvancedOptions && (
          <div className="mt-4 p-4 bg-gray-50 rounded-lg border">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {/* Query Type */}
              <div>
                <label className="block text-xs font-medium text-gray-700 mb-1">
                  Query Type
                </label>
                <select
                  value={queryType}
                  onChange={(e) => setQueryType(e.target.value)}
                  className="w-full text-sm border border-gray-300 rounded px-2 py-1 focus:ring-blue-500 focus:border-blue-500"
                >
                  {Object.entries(availableQueryTypes).map(([key, type]) => (
                    <option key={key} value={key}>{type.name}</option>
                  ))}
                </select>
              </div>

              {/* Confidence Threshold */}
              <div>
                <label className="block text-xs font-medium text-gray-700 mb-1">
                  Confidence Threshold: {Math.round(confidenceThreshold * 100)}%
                </label>
                <input
                  type="range"
                  min="0.1"
                  max="1.0"
                  step="0.1"
                  value={confidenceThreshold}
                  onChange={(e) => setConfidenceThreshold(parseFloat(e.target.value))}
                  className="w-full"
                />
              </div>

              {/* Include Sources */}
              <div className="flex items-center">
                <input
                  type="checkbox"
                  id="includeSources"
                  checked={includeSourcesMode}
                  onChange={(e) => setIncludeSourcesMode(e.target.checked)}
                  className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                />
                <label htmlFor="includeSources" className="ml-2 block text-xs text-gray-700">
                  Include source attribution
                </label>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-6 py-4">
        {messages.length === 0 && (
          <div className="text-center py-12">
            <div className="w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-4">
              <ScaleIcon className="w-8 h-8 text-blue-600" />
            </div>
            <h3 className="text-lg font-medium text-gray-800 mb-2">Welcome to VirLaw AI Ultimate</h3>
            <p className="text-gray-600 mb-4">Ask me any legal question to get started with advanced AI analysis.</p>
            
            {/* Quick Start Options */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 max-w-md mx-auto">
              <button
                onClick={() => setInput("What are the fundamental rights under Article 19?")}
                className="p-3 text-left bg-blue-50 hover:bg-blue-100 rounded-lg transition-colors"
              >
                <p className="text-sm font-medium text-blue-800">Constitutional Law</p>
                <p className="text-xs text-blue-600">Fundamental rights analysis</p>
              </button>
              <button
                onClick={() => setInput("Explain the procedure for filing a civil suit")}
                className="p-3 text-left bg-green-50 hover:bg-green-100 rounded-lg transition-colors"
              >
                <p className="text-sm font-medium text-green-800">Civil Procedure</p>
                <p className="text-xs text-green-600">Legal process guidance</p>
              </button>
            </div>
          </div>
        )}

        {messages.map((msg, index) => renderMessage(msg, index))}

        {isLoadingResponse && (
          <div className="flex justify-start mb-4">
            <div className="bg-gray-100 text-gray-800 rounded-lg px-4 py-2 flex items-center space-x-2">
              <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600"></div>
              <span className="text-sm">VirLaw AI is analyzing...</span>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* System Status */}
      {lastQueryStats && (
        <div className="px-6 py-2 bg-blue-50 border-t border-blue-200">
          <div className="flex items-center justify-between text-xs text-blue-600">
            <div className="flex items-center space-x-4">
              <span>⏱ {lastQueryStats.processing_time_ms}ms</span>
              <span>🎯 {Math.round(lastQueryStats.confidence * 100)}% confidence</span>
              <span>📄 {lastQueryStats.sources_found} sources</span>
            </div>
            {systemHealth && (
              <span className={systemHealth.status === "healthy" ? "text-green-600" : "text-red-600"}>
                ● {systemHealth.status === "healthy" ? "Backend Online" : "Backend Offline"}
              </span>
            )}
          </div>
        </div>
      )}

      {/* Error Message */}
      {sendMessageError && (
        <div className="px-6 py-2 bg-red-50 border-t border-red-200">
          <div className="flex items-center space-x-2">
            <ExclamationTriangleIcon className="h-4 w-4 text-red-500" />
            <p className="text-sm text-red-600">{sendMessageError}</p>
          </div>
        </div>
      )}

      {/* Input Form */}
      <div className="bg-white border-t border-gray-200 px-6 py-4">
        <form onSubmit={handleSendMessage} className="flex space-x-4">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask a legal question..."
            className="flex-1 border border-gray-300 rounded-lg px-4 py-2 focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            disabled={isLoadingResponse}
          />
          <button
            type="submit"
            disabled={isLoadingResponse || !input.trim()}
            className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isLoadingResponse ? "..." : "Send"}
          </button>
        </form>
      </div>
    </div>
  );
};

export default QueryPage;