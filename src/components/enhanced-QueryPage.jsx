import React, { useState, useEffect, useRef } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { auth, db } from "../firebase";
import {
  collection, addDoc, query, orderBy, onSnapshot, serverTimestamp, doc,
  updateDoc, getDoc
} from "firebase/firestore";
import ReactMarkdown from 'react-markdown';
import { apiClient, API_BASE_URL, HF_TOKEN } from '../config/api';

import {
  DocumentTextIcon, TableCellsIcon, PhotoIcon, ScaleIcon, ClockIcon,
  CheckCircleIcon, ExclamationTriangleIcon, InformationCircleIcon,
  ChartBarIcon, CogIcon, ArrowPathIcon, BookOpenIcon, AcademicCapIcon,
  ShieldCheckIcon, DocumentPlusIcon, CloudArrowUpIcon, TrashIcon
} from "@heroicons/react/24/solid";


const QueryPage = () => {
  const { queryId } = useParams();
  const navigate = useNavigate();

  // Your existing state management preserved
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState([]);
  const [isLoadingResponse, setIsLoadingResponse] = useState(false);
  const [sessionTitle, setSessionTitle] = useState("New Chat");
  const [isSessionLoading, setIsSessionLoading] = useState(true);
  const [sessionLoadError, setSessionLoadError] = useState(null);
  const [sendMessageError, setSendMessageError] = useState(null);
  const [pendingMessage, setPendingMessage] = useState(null);

  // FIXED: Only 4 backend-supported query types
  const [queryType, setQueryType] = useState("general_analysis");
  const [availableQueryTypes, setAvailableQueryTypes] = useState({
    general_analysis: { 
      name: "General Legal Analysis", 
      description: "Comprehensive legal analysis including constitutional law", 
      constitutional_capable: true 
    },
    case_analysis: { 
      name: "Case Law Analysis", 
      description: "Legal case analysis and judicial decisions", 
      constitutional_capable: true 
    },
    statutory_interpretation: { 
      name: "Statutory Interpretation", 
      description: "Statutory interpretation and legal provisions", 
      constitutional_capable: true 
    },
    procedure_guidance: { 
      name: "Legal Procedure Guidance", 
      description: "Legal procedure and process guidance", 
      constitutional_capable: false 
    }
  });

  const [includeSourcesMode, setIncludeSourcesMode] = useState(true);
  const [lastQueryStats, setLastQueryStats] = useState(null);
  const [systemHealth, setSystemHealth] = useState(null);
  const [currentQueryId, setCurrentQueryId] = useState(null);
  const [confidenceThreshold, setConfidenceThreshold] = useState(0.7);
  const [showAdvancedOptions, setShowAdvancedOptions] = useState(false);
  const [realTimeMetrics, setRealTimeMetrics] = useState({});

  // Enhanced: Query-with-file support
  const [queryWithFileMode, setQueryWithFileMode] = useState(false);
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [isUploadingFiles, setIsUploadingFiles] = useState(false);
  const [uploadProgress, setUploadProgress] = useState({});
  const [tempUploadedFiles, setTempUploadedFiles] = useState([]);
  const [dragActive, setDragActive] = useState(false);

  // Enhanced: Constitutional intelligence
  const [constitutionalReadiness, setConstitutionalReadiness] = useState(null);
  const [constitutionalSuggestions, setConstitutionalSuggestions] = useState([]);

  const messagesEndRef = useRef(null);
  const currentUserId = auth.currentUser?.uid;

  // FIXED: Offline-friendly initialization
  useEffect(() => {
    // Set UI defaults immediately
    setConstitutionalSuggestions([
      { query: "What are the fundamental rights under Article 19?", type: "general_analysis" },
      { query: "Explain Article 21 right to life and personal liberty", type: "general_analysis" },
      { query: "What is the basic structure doctrine?", type: "case_analysis" },
      { query: "Explain directive principles of state policy", type: "general_analysis" }
    ]);

    setSystemHealth({ system_status: "offline", message: "Working in offline mode" });
    setConstitutionalReadiness({ constitutional_ready: false });

    // Optional: Try backend connection in background (won't block UI)
    const checkBackend = async () => {
      try {
        await apiClient.get("/health", { timeout: 5000 });
        setSystemHealth({ system_status: "online", backend_available: true });
        console.log("✅ Backend available");
      } catch (error) {
        console.log("ℹ️ Continuing in offline mode");
      }
    };

    // Check backend in background after UI is ready
    setTimeout(checkBackend, 100);
  }, []);

  // FIXED: Firebase session loading useEffect with proper loading state management
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
          console.warn(`Firestore document for session ID ${queryId} not found.`);
          setSessionLoadError("Session not found.");
          setSessionTitle("Session Not Found");
          setMessages([]);
          navigate("/dashboard/new");
        }
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
          ...doc.data()
        }));
        setMessages(loadedMessages);
        setIsSessionLoading(false); // ← FIXED: This was missing!
      }, (error) => {
        console.error("Error fetching messages for session:", error);
        setIsSessionLoading(false);
      });

      return () => {
        unsubscribeSession();
        unsubscribeMessages();
      };
    }

    // Default fallback
    setSessionTitle("New Chat...");
    setMessages([]);
    setIsSessionLoading(false);

  }, [queryId, currentUserId, navigate]);

  // FIXED: Handle pending message from new session creation
  useEffect(() => {
    if (pendingMessage && queryId && queryId !== "new") {
      const processPendingMessage = async () => {
        try {
          await addDoc(collection(db, "users", currentUserId, "querySessions", queryId, "messages"), {
            text: pendingMessage,
            sender: "user",
            createdAt: serverTimestamp(),
          });

          await processConstitutionalQuery(pendingMessage, queryId);
          setPendingMessage(null);
        } catch (error) {
          console.error("Error processing pending message:", error);
          setSendMessageError("Failed to process message. Please try again.");
          setPendingMessage(null);
        }
      };

      processPendingMessage();
    }
  }, [pendingMessage, queryId, currentUserId]);

  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // UPDATED: processConstitutionalQuery - constitutional detection logic
  const processConstitutionalQuery = async (userMessage, sessionId) => {
    setIsLoadingResponse(true);
    setSendMessageError(null);
    const queryStartTime = Date.now();
    
    let aiResponseText = "An error occurred while getting a response from VirLaw AI.";
    let queryStats = null;
    let sources = [];
    let metadata = {};

    try {
      console.log("🏛️ Processing constitutional query:", userMessage);

      // Enhanced: Handle query-with-file mode
      if (queryWithFileMode && selectedFiles.length > 0) {
        await uploadFilesForQuery();
      }

      // UPDATED: Constitutional detection based on content, not query type
      const isConstitutionalQuery = userMessage.toLowerCase().includes('article') ||
                                  userMessage.toLowerCase().includes('constitution') ||
                                  userMessage.toLowerCase().includes('fundamental right') ||
                                  userMessage.toLowerCase().includes('directive principle');

      const endpoint = isConstitutionalQuery ? '/ultimate-query' : '/gemini-rag';

      // Enhanced: Prepare request data with chat history context
      const requestData = {
        [endpoint.includes('ultimate-query') ? 'question' : 'prompt']: userMessage,
        query_type: queryType,
        confidence_threshold: confidenceThreshold,
        include_sources: includeSourcesMode,
        // NEW: Add chat history for contextual understanding (last 10 messages)
        chat_history: messages.slice(-10).map(msg => ({
        role: msg.sender === "user" ? "user" : "assistant",
        content: msg.text
        })),
        // Preserve existing file functionality
        ...(tempUploadedFiles.length > 0 && { temp_files: tempUploadedFiles })
      };

      const ragResponse = await apiClient.post(endpoint, requestData);
      const data = ragResponse.data;

      aiResponseText = data.response;
      sources = data.sources || [];
      metadata = data.metadata || {};

      const constitutionalAnalysis = data.constitutional_analysis || data.legal_analysis || {};

      const processingTime = Date.now() - queryStartTime;
      queryStats = {
        processing_time_ms: processingTime,
        confidence: data.confidence || 0.8,
        sources_found: sources.length,
        query_id: data.query_id,
        query_type: queryType,
        model_used: metadata.model_used || "unknown",
        timestamp: new Date().toISOString(),
        constitutional_analysis: constitutionalAnalysis,
        constitutional_query: isConstitutionalQuery,
        chat_context_used: metadata.chat_context_used || false,
        context_messages_count: metadata.context_messages_count || 0
      };

      setLastQueryStats(queryStats);
      setCurrentQueryId(data.query_id);

      if (tempUploadedFiles.length > 0) {
        setTempUploadedFiles([]);
        setSelectedFiles([]);
      }

      console.log("✅ Constitutional response received:", {
        response_length: aiResponseText.length,
        sources_count: sources.length,
        confidence: data.confidence,
        constitutional_analysis: !!constitutionalAnalysis,
        processing_time: processingTime,
        chat_context_used: queryStats.chat_context_used,
        context_messages: queryStats.context_messages_count
      });

    } catch (ragError) {
      console.error("Error calling constitutional RAG API:", ragError);
      const processingTime = Date.now() - queryStartTime;
      
      queryStats = {
        processing_time_ms: processingTime,
        confidence: 0.0,
        sources_found: 0,
        error: true,
        error_type: ragError.response?.status || "network_error",
        timestamp: new Date().toISOString(),
        constitutional_query: isConstitutionalQuery,
        chat_context_used: false,
        context_messages_count: 0
      };

      if (ragError.response) {
        aiResponseText = `VirLaw AI: Failed to get constitutional analysis (Code: ${ragError.response.status}). Please check the backend connection.`;
      } else if (ragError.request) {
        aiResponseText = "VirLaw AI: No response from the constitutional AI server. Please ensure the backend is running.";
      } else {
        aiResponseText = `VirLaw AI: Error sending constitutional query: ${ragError.message}`;
      }

      setSendMessageError(aiResponseText);
      setLastQueryStats(queryStats);
    } finally {
      setIsLoadingResponse(false);
    }

    // Store enhanced AI response with constitutional metadata + context tracking
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
          query_id: currentQueryId,
          constitutional_analysis: queryStats?.constitutional_analysis || {},
          files_used: tempUploadedFiles.length > 0 ? tempUploadedFiles.map(f => f.filename) : [],
          chat_context_metadata: {
            context_used: queryStats?.chat_context_used || false,
            context_messages_count: queryStats?.context_messages_count || 0,
            conversation_continuity: messages.length > 1
          }
        }
      });
    } catch (error) {
      console.error("Error saving AI response to Firestore:", error);
    }
  };

  // ADDED: Missing file handling functions
  const uploadFilesForQuery = async () => {
    if (selectedFiles.length === 0) return;
  
    try {
      setIsUploadingFiles(true);
      setSendMessageError(null);
  
      const formData = new FormData();
      selectedFiles.forEach(file => {
        formData.append('files', file);
      });
      formData.append('temporary', 'true');
  
      selectedFiles.forEach(file => {
        setUploadProgress(prev => ({
          ...prev,
          [file.name]: { status: 'uploading', progress: 0 }
        }));
      });
  
      // ✅ FIXED: Use apiClient instead of axios
      const response = await apiClient.post('/upload-documents', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        onUploadProgress: (progressEvent) => {
          const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          selectedFiles.forEach(file => {
            setUploadProgress(prev => ({
              ...prev,
              [file.name]: { status: 'uploading', progress: percentCompleted }
            }));
          });
        },
        timeout: 120000, // 2 minutes for large files
      });
  
      const { uploaded } = response.data;
      setTempUploadedFiles(uploaded);
  
      uploaded.forEach(file => {
        setUploadProgress(prev => ({
          ...prev,
          [file.filename]: { status: 'success', progress: 100 }
        }));
      });
  
    } catch (error) {
      console.error("Error uploading files for query:", error);
      setSendMessageError("Failed to upload files for query. Please try again.");
      
      selectedFiles.forEach(file => {
        setUploadProgress(prev => ({
          ...prev,
          [file.name]: { status: 'error', progress: 0, error: error.message }
        }));
      });
    } finally {
      setIsUploadingFiles(false);
    }
  };
  

  // ADDED: Missing drag and drop handlers
  const handleDragEnter = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    const files = Array.from(e.dataTransfer.files);
    setSelectedFiles(files);
  };

  const handleFileSelect = (e) => {
    const files = Array.from(e.target.files);
    setSelectedFiles(files);
  };

  const removeFile = (index) => {
    setSelectedFiles(prev => prev.filter((_, i) => i !== index));
  };

  // Your existing handleSendMessage preserved with enhancements
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

      await processConstitutionalQuery(userMessageText, queryId);
    } catch (error) {
      console.error("Error in handleSendMessage:", error);
      setIsLoadingResponse(false);
      setSendMessageError("Failed to send message. Please try again.");
      setMessages((prevMessages) =>
        prevMessages.filter((msg) => msg.id !== tempUserMessage.id)
      );
    }
  };

// Enhanced: Message rendering with constitutional intelligence
const renderMessage = (msg, index) => {
  const isUser = msg.sender === "user";
  const hasMetadata = msg.metadata && !isUser;
  const sources = hasMetadata ? msg.metadata.sources || [] : [];
  const queryStats = hasMetadata ? msg.metadata.query_stats : null;
  const confidence = hasMetadata ? msg.metadata.confidence || 0 : null;
  const constitutionalAnalysis = hasMetadata ? msg.metadata.constitutional_analysis || {} : {};
  const filesUsed = hasMetadata ? msg.metadata.files_used || [] : [];

  return (
    <div key={msg.id || index} className={`flex ${isUser ? "justify-end" : "justify-start"} mb-4`}>
      <div className={`max-w-3xl ${isUser ? "bg-blue-600 text-white" : "bg-gray-100 text-gray-800"} rounded-lg px-4 py-2`}>
        <div className="text-sm prose prose-sm max-w-none">
          <ReactMarkdown 
            components={{
              strong: ({children}) => (
                <strong className={`font-semibold ${isUser ? 'text-white' : 'text-gray-900'}`}>
                  {children}
                </strong>
              ),
              em: ({children}) => (
                <em className={`italic ${isUser ? 'text-gray-200' : 'text-gray-700'}`}>
                  {children}
                </em>
              ),
              ul: ({children}) => <ul className="list-disc ml-4 my-2 space-y-1">{children}</ul>,
              li: ({children}) => <li className="text-sm">{children}</li>,
              p: ({children}) => <p className="my-2">{children}</p>
            }}
          >
            {msg.text}
          </ReactMarkdown>
        </div>
        
        {/* Enhanced AI message with constitutional intelligence */}
        {!isUser && hasMetadata && (
          <div className="mt-3 pt-3 border-t border-gray-200">
            {/* Constitutional Analysis Indicator */}
            {constitutionalAnalysis.constitutional_query && (
              <div className="flex items-center space-x-2 mb-2">
                <BookOpenIcon className="h-4 w-4 text-amber-600" />
                <span className="text-xs font-medium text-amber-700 bg-amber-50 px-2 py-1 rounded">
                  Constitutional Analysis
                </span>
                {constitutionalAnalysis.constitutional_articles_referenced && (
                  <span className="text-xs text-amber-600">
                    Articles: {constitutionalAnalysis.constitutional_articles_referenced.join(', ')}
                  </span>
                )}
              </div>
            )}

            {/* Files Used Indicator */}
            {filesUsed.length > 0 && (
              <div className="flex items-center space-x-2 mb-2">
                <DocumentPlusIcon className="h-4 w-4 text-purple-600" />
                <span className="text-xs font-medium text-purple-700">
                  Analysis based on uploaded files: {filesUsed.join(', ')}
                </span>
              </div>
            )}

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
                {sources.filter(s => s.document?.toLowerCase().includes('constitution')).length > 0 && (
                  <span className="text-xs text-amber-600 ml-2">
                    ({sources.filter(s => s.document?.toLowerCase().includes('constitution')).length} constitutional)
                  </span>
                )}
              </div>
            )}

            {/* Enhanced Query Stats */}
            {queryStats && (
              <div className="flex items-center space-x-4 text-xs text-gray-500">
                <span>⏱ {queryStats.processing_time_ms}ms</span>
                <span className={queryStats.constitutional_query ? "text-amber-600" : ""}>
                  🔍 {queryStats.query_type}
                </span>
                {queryStats.model_used && <span>🤖 {queryStats.model_used}</span>}
                {queryStats.constitutional_query && (
                  <span className="text-amber-600">🏛️ Constitutional</span>
                )}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};


  // Your existing loading/error states preserved
  if (isSessionLoading) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <p className="text-gray-600">Loading constitutional session...</p>
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
            Start New Constitutional Query
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 flex flex-col h-full">
      {/* Enhanced Header with Constitutional Intelligence */}
      <div className="bg-white border-b border-gray-200 px-6 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div>
              <h1 className="text-xl font-semibold text-gray-800">{sessionTitle}</h1>
              <p className="text-sm text-gray-600">Constitutional Legal AI Assistant</p>
            </div>
            
            {/* Constitutional Readiness Indicator */}
            {constitutionalReadiness?.constitutional_ready && (
              <div className="flex items-center space-x-2 px-2 py-1 bg-amber-50 rounded">
                <ShieldCheckIcon className="h-4 w-4 text-amber-600" />
                <span className="text-xs text-amber-700">Constitutional Ready</span>
              </div>
            )}
          </div>
          
          {/* Enhanced Controls */}
          <div className="flex items-center space-x-2">
            {/* Query-with-File Toggle */}
            <button
              onClick={() => setQueryWithFileMode(!queryWithFileMode)}
              className={`px-3 py-1 text-sm rounded transition-colors ${
                queryWithFileMode 
                  ? 'bg-purple-100 text-purple-700 hover:bg-purple-200' 
                  : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
              }`}
            >
              <DocumentPlusIcon className="h-4 w-4 inline mr-1" />
              Query with File
            </button>

            {/* Advanced Options Toggle */}
            <button
              onClick={() => setShowAdvancedOptions(!showAdvancedOptions)}
              className="flex items-center space-x-2 px-3 py-1 text-sm bg-gray-100 text-gray-600 rounded hover:bg-gray-200"
            >
              <CogIcon className="h-4 w-4" />
              <span>Advanced</span>
            </button>
          </div>
        </div>

        {/* Enhanced Advanced Options Panel */}
        {showAdvancedOptions && (
          <div className="mt-4 p-4 bg-gray-50 rounded-lg border">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {/* Enhanced Query Type with Constitutional Options */}
              <div>
                <label className="block text-xs font-medium text-gray-700 mb-1">
                  Query Type
                </label>
                <select
                  value={queryType}
                  onChange={e => setQueryType(e.target.value)}
                  className="w-full px-2 py-1 text-sm border border-gray-300 rounded focus:ring-blue-500 focus:border-blue-500"
                >
                  {Object.entries(availableQueryTypes).map(([key, typeInfo]) => (
                    <option key={key} value={key}>
                      {typeInfo.name}
                    </option>
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

        {/* Enhanced: Query-with-File Panel */}
        {queryWithFileMode && (
          <div className="mt-4 p-4 bg-purple-50 rounded-lg border border-purple-200">
            <div className="mb-3">
              <h4 className="text-sm font-medium text-purple-800 mb-1">Query with Files</h4>
              <p className="text-xs text-purple-600">Upload documents to analyze alongside your query</p>
            </div>

            {/* File Upload Area */}
            <div
              className={`border-2 border-dashed rounded-lg p-4 text-center transition-colors ${
                dragActive 
                  ? "border-purple-500 bg-purple-100" 
                  : "border-purple-300 hover:border-purple-400"
              }`}
              onDragEnter={handleDragEnter}
              onDragLeave={handleDragLeave}
              onDragOver={handleDragOver}
              onDrop={handleDrop}
            >
              <CloudArrowUpIcon className="h-8 w-8 text-purple-400 mx-auto mb-2" />
              <p className="text-sm text-purple-700 mb-2">
                Drag files here or click to select
              </p>
              
              <input
                type="file"
                multiple
                onChange={handleFileSelect}
                className="hidden"
                id="query-file-upload"
                accept=".pdf,.docx,.doc,.txt,.rtf"
              />
              <label
                htmlFor="query-file-upload"
                className="inline-flex items-center px-3 py-1 bg-purple-600 text-white text-sm rounded hover:bg-purple-700 cursor-pointer"
              >
                <DocumentPlusIcon className="h-4 w-4 mr-1" />
                Select Files
              </label>
            </div>

            {/* Selected Files Display */}
            {selectedFiles.length > 0 && (
              <div className="mt-3 space-y-2">
                {selectedFiles.map((file, index) => (
                  <div key={index} className="flex items-center justify-between p-2 bg-white rounded border">
                    <div className="flex items-center space-x-2">
                      <DocumentTextIcon className="h-4 w-4 text-purple-600" />
                      <span className="text-sm text-gray-700">{file.name}</span>
                      <span className="text-xs text-gray-500">({Math.round(file.size / 1024)}KB)</span>
                    </div>
                    <button
                      onClick={() => removeFile(index)}
                      className="p-1 text-red-500 hover:bg-red-50 rounded"
                    >
                      <TrashIcon className="h-4 w-4" />
                    </button>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>

      {/* Enhanced Messages Area */}
      <div className="flex-1 overflow-y-auto px-6 py-4">
        {messages.length === 0 && (
          <div className="text-center py-12">
            <div className="w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-4">
              <BookOpenIcon className="w-8 h-8 text-blue-600" />
            </div>
            <h3 className="text-lg font-medium text-gray-800 mb-2">Welcome to Constitutional VirLaw AI</h3>
            <p className="text-gray-600 mb-6">Ask me any legal question to get started with advanced constitutional AI analysis.</p>
            
            {/* Enhanced Constitutional Quick Start Options */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 max-w-2xl mx-auto mb-6">
              {constitutionalSuggestions.map((suggestion, index) => (
                <button
                  key={index}
                  onClick={() => {
                    setInput(suggestion.query);
                    setQueryType(suggestion.type);
                  }}
                  className="p-4 text-left bg-amber-50 hover:bg-amber-100 rounded-lg transition-colors border border-amber-200"
                >
                  <div className="flex items-center space-x-2 mb-2">
                    <BookOpenIcon className="h-4 w-4 text-amber-600" />
                    <span className="text-sm font-medium text-amber-800">Constitutional Law</span>
                  </div>
                  <p className="text-sm text-amber-700">{suggestion.query}</p>
                </button>
              ))}
            </div>

            {/* Constitutional Readiness Status */}
            {constitutionalReadiness && (
              <div className="max-w-md mx-auto">
                {constitutionalReadiness.constitutional_ready ? (
                  <div className="flex items-center justify-center space-x-2 text-sm text-green-700 bg-green-50 py-2 px-4 rounded">
                    <CheckCircleIcon className="h-4 w-4" />
                    <span>Constitutional database ready - Full constitutional analysis available</span>
                  </div>
                ) : (
                  <div className="flex items-center justify-center space-x-2 text-sm text-amber-700 bg-amber-50 py-2 px-4 rounded">
                    <InformationCircleIcon className="h-4 w-4" />
                    <span>Upload the Indian Constitution for enhanced constitutional analysis</span>
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {messages.map((msg, index) => renderMessage(msg, index))}

        {isLoadingResponse && (
          <div className="flex justify-start mb-4">
            <div className="bg-gray-100 text-gray-800 rounded-lg px-4 py-2 flex items-center space-x-2">
              <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600"></div>
              <span className="text-sm">VirLaw AI analyzing...</span>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Enhanced System Status with Constitutional Intelligence */}
      {lastQueryStats && (
        <div className={`px-6 py-2 border-t ${lastQueryStats.constitutional_query ? 'bg-amber-50 border-amber-200' : 'bg-blue-50 border-blue-200'}`}>
          <div className="flex items-center justify-between text-xs">
            <div className="flex items-center space-x-4">
              <span className={lastQueryStats.constitutional_query ? 'text-amber-600' : 'text-blue-600'}>
                ⏱ {lastQueryStats.processing_time_ms}ms
              </span>
              <span className={lastQueryStats.constitutional_query ? 'text-amber-600' : 'text-blue-600'}>
                🎯 {Math.round(lastQueryStats.confidence * 100)}% confidence
              </span>
              <span className={lastQueryStats.constitutional_query ? 'text-amber-600' : 'text-blue-600'}>
                📄 {lastQueryStats.sources_found} sources
              </span>
              {lastQueryStats.constitutional_query && (
                <span className="text-amber-600">🏛️ Constitutional Analysis</span>
              )}
            </div>
            {systemHealth && (
              <span className={systemHealth.system_status === "online" ? "text-green-600" : "text-red-600"}>
                ● {systemHealth.system_status === "online" ? "Backend Online" : "Backend Offline"}
              </span>
            )}
          </div>
        </div>
      )}

      {/* Your existing error message preserved */}
      {sendMessageError && (
        <div className="px-6 py-2 bg-red-50 border-t border-red-200">
          <div className="flex items-center space-x-2">
            <ExclamationTriangleIcon className="h-4 w-4 text-red-500" />
            <p className="text-sm text-red-600">{sendMessageError}</p>
          </div>
        </div>
      )}

      {/* Enhanced Input Form with File Support */}
      <div className="bg-white border-t border-gray-200 px-6 py-4">
        {/* File Upload Status */}
        {isUploadingFiles && (
          <div className="mb-2 text-sm text-purple-600 flex items-center space-x-2">
            <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-purple-600"></div>
            <span>Uploading files for analysis...</span>
          </div>
        )}

        <form onSubmit={handleSendMessage} className="flex space-x-4">
          <div className="flex-1 relative">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder={queryWithFileMode ? "Ask a question about your uploaded files..." : "Ask a constitutional law question..."}
              className="w-full border border-gray-300 rounded-lg px-4 py-2 pr-24 focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              disabled={isLoadingResponse || isUploadingFiles}
            />
            {/* File Count Indicator */}
            {selectedFiles.length > 0 && (
              <div className="absolute right-2 top-1/2 transform -translate-y-1/2 flex items-center space-x-1">
                <DocumentPlusIcon className="h-4 w-4 text-purple-600" />
                <span className="text-xs text-purple-600">{selectedFiles.length}</span>
              </div>
            )}
          </div>
          <button
            type="submit"
            disabled={isLoadingResponse || !input.trim() || isUploadingFiles}
            className={`px-6 py-2 text-white rounded-lg disabled:opacity-50 disabled:cursor-not-allowed bg-blue-600 hover:bg-blue-700`}
          >
            {isLoadingResponse || isUploadingFiles ? "..." : "Send"}
          </button>
        </form>
      </div>
    </div>
  );
};

export default QueryPage;
