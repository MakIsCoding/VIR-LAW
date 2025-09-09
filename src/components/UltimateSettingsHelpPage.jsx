// 🚀 ULTIMATE Enhanced Settings & Help Page - All Features Unlocked
// Complete configuration, advanced settings, and comprehensive help system

import React, { useState, useEffect } from "react";
import { auth } from "../firebase";
import axios from "axios";
import {
  CogIcon, QuestionMarkCircleIcon, DocumentTextIcon,
  BeakerIcon, BoltIcon, ShieldCheckIcon, ClockIcon,
  ServerIcon, ChatBubbleLeftRightIcon, AcademicCapIcon,
  ExclamationTriangleIcon, InformationCircleIcon,
  CheckCircleIcon, ArrowPathIcon, CloudIcon,
  WrenchScrewdriverIcon, BookOpenIcon, ChartBarIcon,
  LightBulbIcon, RocketLaunchIcon, UserCircleIcon
} from "@heroicons/react/24/solid";

const UltimateSettingsHelpPage = () => {
  const [activeTab, setActiveTab] = useState("settings");
  const [systemHealth, setSystemHealth] = useState(null);
  const [userPreferences, setUserPreferences] = useState({
    defaultQueryType: "general_analysis",
    confidenceThreshold: 0.7,
    includeSourcesDefault: true,
    autoProcessDocuments: true,
    batchSize: 5,
    maxResults: 10,
    theme: "light",
    notifications: true,
    advancedMode: false
  });
  const [availableQueryTypes, setAvailableQueryTypes] = useState({});
  const [isLoading, setIsLoading] = useState(true);
  const [saveStatus, setSaveStatus] = useState(null);
  
  const currentUser = auth.currentUser;

  // Initialize component
  useEffect(() => {
    loadInitialData();
  }, []);

  // Load initial data
  const loadInitialData = async () => {
    try {
      setIsLoading(true);
      
      const [healthResponse, queryTypesResponse] = await Promise.all([
        axios.get("http://localhost:8000/health"),
        axios.get("http://localhost:8000/query-types")
      ]);
      
      setSystemHealth(healthResponse.data);
      setAvailableQueryTypes(queryTypesResponse.data.query_types);
      
      // Load user preferences from localStorage
      const saved = localStorage.getItem('virlaw_preferences');
      if (saved) {
        setUserPreferences({...userPreferences, ...JSON.parse(saved)});
      }
      
    } catch (error) {
      console.error("Error loading initial data:", error);
    } finally {
      setIsLoading(false);
    }
  };

  // Save user preferences
  const savePreferences = () => {
    try {
      localStorage.setItem('virlaw_preferences', JSON.stringify(userPreferences));
      setSaveStatus("success");
      setTimeout(() => setSaveStatus(null), 3000);
    } catch (error) {
      console.error("Error saving preferences:", error);
      setSaveStatus("error");
      setTimeout(() => setSaveStatus(null), 3000);
    }
  };

  // Update preference
  const updatePreference = (key, value) => {
    setUserPreferences(prev => ({
      ...prev,
      [key]: value
    }));
  };

  // Reset to defaults
  const resetToDefaults = () => {
    if (confirm("Are you sure you want to reset all settings to defaults?")) {
      setUserPreferences({
        defaultQueryType: "general_analysis",
        confidenceThreshold: 0.7,
        includeSourcesDefault: true,
        autoProcessDocuments: true,
        batchSize: 5,
        maxResults: 10,
        theme: "light",
        notifications: true,
        advancedMode: false
      });
    }
  };

  if (!currentUser) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <div className="text-center">
          <ExclamationTriangleIcon className="h-12 w-12 text-red-500 mx-auto mb-4" />
          <h2 className="text-xl font-semibold text-gray-800 mb-2">Authentication Required</h2>
          <p className="text-gray-600">Please sign in to access settings and help.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 p-6 bg-gray-50 overflow-y-auto">
      {/* Header */}
      <div className="bg-white rounded-lg shadow-md p-6 mb-6">
        <h1 className="text-2xl font-bold text-gray-800 mb-2">Settings & Help</h1>
        <p className="text-gray-600">Configure your experience and get help with VirLaw AI</p>
      </div>

      {/* Navigation Tabs */}
      <div className="bg-white rounded-lg shadow-md mb-6">
        <div className="border-b border-gray-200">
          <nav className="flex space-x-8 px-6">
            {[
              { id: "settings", name: "Settings", icon: CogIcon },
              { id: "help", name: "Help & Guide", icon: QuestionMarkCircleIcon },
              { id: "about", name: "About System", icon: InformationCircleIcon },
              { id: "api", name: "API Reference", icon: BookOpenIcon }
            ].map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`py-4 px-2 border-b-2 font-medium text-sm ${
                  activeTab === tab.id
                    ? "border-blue-500 text-blue-600"
                    : "border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300"
                }`}
              >
                <div className="flex items-center space-x-2">
                  <tab.icon className="h-5 w-5" />
                  <span>{tab.name}</span>
                </div>
              </button>
            ))}
          </nav>
        </div>

        <div className="p-6">
          {/* Settings Tab */}
          {activeTab === "settings" && (
            <div className="space-y-6">
              {/* Query Settings */}
              <div className="bg-gray-50 p-6 rounded-lg">
                <div className="flex items-center space-x-2 mb-4">
                  <ChatBubbleLeftRightIcon className="h-6 w-6 text-blue-600" />
                  <h3 className="text-lg font-semibold text-gray-800">Query Settings</h3>
                </div>
                
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Default Query Type
                    </label>
                    <select
                      value={userPreferences.defaultQueryType}
                      onChange={(e) => updatePreference("defaultQueryType", e.target.value)}
                      className="w-full border border-gray-300 rounded-md px-3 py-2 focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                    >
                      {Object.entries(availableQueryTypes).map(([key, type]) => (
                        <option key={key} value={key}>
                          {type.name}
                        </option>
                      ))}
                    </select>
                    <p className="text-xs text-gray-500 mt-1">
                      Choose the default analysis type for new queries
                    </p>
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Confidence Threshold: {userPreferences.confidenceThreshold.toFixed(1)}
                    </label>
                    <input
                      type="range"
                      min="0.1"
                      max="1.0"
                      step="0.1"
                      value={userPreferences.confidenceThreshold}
                      onChange={(e) => updatePreference("confidenceThreshold", parseFloat(e.target.value))}
                      className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                    />
                    <p className="text-xs text-gray-500 mt-1">
                      Minimum confidence level for displaying results
                    </p>
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Maximum Results per Query
                    </label>
                    <select
                      value={userPreferences.maxResults}
                      onChange={(e) => updatePreference("maxResults", parseInt(e.target.value))}
                      className="w-full border border-gray-300 rounded-md px-3 py-2 focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                    >
                      <option value={5}>5 Results (Fast)</option>
                      <option value={10}>10 Results (Balanced)</option>
                      <option value={15}>15 Results (Comprehensive)</option>
                      <option value={20}>20 Results (Maximum)</option>
                    </select>
                  </div>
                  
                  <div className="space-y-3">
                    <div className="flex items-center">
                      <input
                        type="checkbox"
                        id="includeSources"
                        checked={userPreferences.includeSourcesDefault}
                        onChange={(e) => updatePreference("includeSourcesDefault", e.target.checked)}
                        className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                      />
                      <label htmlFor="includeSources" className="ml-2 text-sm text-gray-700">
                        Include sources by default
                      </label>
                    </div>
                    
                    <div className="flex items-center">
                      <input
                        type="checkbox"
                        id="advancedMode"
                        checked={userPreferences.advancedMode}
                        onChange={(e) => updatePreference("advancedMode", e.target.checked)}
                        className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                      />
                      <label htmlFor="advancedMode" className="ml-2 text-sm text-gray-700">
                        Enable advanced mode
                      </label>
                    </div>
                  </div>
                </div>
              </div>
              
              {/* Document Processing Settings */}
              <div className="bg-gray-50 p-6 rounded-lg">
                <div className="flex items-center space-x-2 mb-4">
                  <DocumentTextIcon className="h-6 w-6 text-green-600" />
                  <h3 className="text-lg font-semibold text-gray-800">Document Processing</h3>
                </div>
                
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Processing Batch Size
                    </label>
                    <select
                      value={userPreferences.batchSize}
                      onChange={(e) => updatePreference("batchSize", parseInt(e.target.value))}
                      className="w-full border border-gray-300 rounded-md px-3 py-2 focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                    >
                      <option value={1}>1 Document (Highest Quality)</option>
                      <option value={3}>3 Documents (High Quality)</option>
                      <option value={5}>5 Documents (Balanced)</option>
                      <option value={10}>10 Documents (Fast)</option>
                    </select>
                    <p className="text-xs text-gray-500 mt-1">
                      Number of documents to process simultaneously
                    </p>
                  </div>
                  
                  <div className="flex items-center">
                    <input
                      type="checkbox"
                      id="autoProcess"
                      checked={userPreferences.autoProcessDocuments}
                      onChange={(e) => updatePreference("autoProcessDocuments", e.target.checked)}
                      className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                    />
                    <label htmlFor="autoProcess" className="ml-2 text-sm text-gray-700">
                      Auto-process documents after upload
                    </label>
                  </div>
                </div>
              </div>
              
              {/* User Interface Settings */}
              <div className="bg-gray-50 p-6 rounded-lg">
                <div className="flex items-center space-x-2 mb-4">
                  <UserCircleIcon className="h-6 w-6 text-purple-600" />
                  <h3 className="text-lg font-semibold text-gray-800">User Interface</h3>
                </div>
                
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Theme
                    </label>
                    <select
                      value={userPreferences.theme}
                      onChange={(e) => updatePreference("theme", e.target.value)}
                      className="w-full border border-gray-300 rounded-md px-3 py-2 focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                    >
                      <option value="light">Light Theme</option>
                      <option value="dark">Dark Theme</option>
                      <option value="auto">Auto (System)</option>
                    </select>
                  </div>
                  
                  <div className="flex items-center">
                    <input
                      type="checkbox"
                      id="notifications"
                      checked={userPreferences.notifications}
                      onChange={(e) => updatePreference("notifications", e.target.checked)}
                      className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                    />
                    <label htmlFor="notifications" className="ml-2 text-sm text-gray-700">
                      Enable notifications
                    </label>
                  </div>
                </div>
              </div>
              
              {/* Action Buttons */}
              <div className="flex items-center justify-between pt-4">
                <button
                  onClick={resetToDefaults}
                  className="px-4 py-2 text-gray-600 border border-gray-300 rounded-md hover:bg-gray-50"
                >
                  Reset to Defaults
                </button>
                
                <div className="flex items-center space-x-3">
                  {saveStatus && (
                    <div className={`flex items-center space-x-1 text-sm ${
                      saveStatus === "success" ? "text-green-600" : "text-red-600"
                    }`}>
                      {saveStatus === "success" ? (
                        <CheckCircleIcon className="h-4 w-4" />
                      ) : (
                        <ExclamationTriangleIcon className="h-4 w-4" />
                      )}
                      <span>
                        {saveStatus === "success" ? "Settings saved!" : "Failed to save"}
                      </span>
                    </div>
                  )}
                  
                  <button
                    onClick={savePreferences}
                    className="px-6 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700"
                  >
                    Save Settings
                  </button>
                </div>
              </div>
            </div>
          )}

          {/* Help Tab */}
          {activeTab === "help" && (
            <div className="space-y-8">
              {/* Getting Started */}
              <div>
                <div className="flex items-center space-x-2 mb-4">
                  <RocketLaunchIcon className="h-6 w-6 text-blue-600" />
                  <h3 className="text-lg font-semibold text-gray-800">Getting Started</h3>
                </div>
                
                <div className="space-y-4">
                  <div className="bg-blue-50 p-4 rounded-lg">
                    <h4 className="font-medium text-blue-900 mb-2">1. Upload Legal Documents</h4>
                    <p className="text-blue-800 text-sm">
                      Go to Document Management and upload your legal documents (PDF, DOCX, etc.). 
                      The system will automatically extract text, tables, images, and legal citations.
                    </p>
                  </div>
                  
                  <div className="bg-green-50 p-4 rounded-lg">
                    <h4 className="font-medium text-green-900 mb-2">2. Ask Legal Questions</h4>
                    <p className="text-green-800 text-sm">
                      Use the chat interface to ask specific legal questions. Choose from different 
                      query types like case analysis, statutory interpretation, or procedural guidance.
                    </p>
                  </div>
                  
                  <div className="bg-purple-50 p-4 rounded-lg">
                    <h4 className="font-medium text-purple-900 mb-2">3. Review Sources & Citations</h4>
                    <p className="text-purple-800 text-sm">
                      Each response includes source attribution, confidence scores, and relevant 
                      legal citations extracted from your documents.
                    </p>
                  </div>
                </div>
              </div>
              
              {/* Query Types */}
              <div>
                <div className="flex items-center space-x-2 mb-4">
                  <AcademicCapIcon className="h-6 w-6 text-green-600" />
                  <h3 className="text-lg font-semibold text-gray-800">Query Types Explained</h3>
                </div>
                
                <div className="space-y-3">
                  {Object.entries(availableQueryTypes).map(([key, type]) => (
                    <div key={key} className="border border-gray-200 p-4 rounded-lg">
                      <h4 className="font-medium text-gray-900 mb-1">{type.name}</h4>
                      <p className="text-gray-600 text-sm">{type.description}</p>
                    </div>
                  ))}
                </div>
              </div>
              
              {/* Tips & Best Practices */}
              <div>
                <div className="flex items-center space-x-2 mb-4">
                  <LightBulbIcon className="h-6 w-6 text-yellow-600" />
                  <h3 className="text-lg font-semibold text-gray-800">Tips & Best Practices</h3>
                </div>
                
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="bg-yellow-50 p-4 rounded-lg">
                    <h4 className="font-medium text-yellow-900 mb-2">📄 Document Quality</h4>
                    <ul className="text-yellow-800 text-sm space-y-1">
                      <li>• Use high-quality, searchable PDFs</li>
                      <li>• Ensure text is readable and not image-only</li>
                      <li>• Include complete legal documents</li>
                    </ul>
                  </div>
                  
                  <div className="bg-green-50 p-4 rounded-lg">
                    <h4 className="font-medium text-green-900 mb-2">❓ Query Formatting</h4>
                    <ul className="text-green-800 text-sm space-y-1">
                      <li>• Be specific about legal provisions</li>
                      <li>• Include relevant case names or sections</li>
                      <li>• Use proper legal terminology</li>
                    </ul>
                  </div>
                  
                  <div className="bg-blue-50 p-4 rounded-lg">
                    <h4 className="font-medium text-blue-900 mb-2">🎯 Accuracy Tips</h4>
                    <ul className="text-blue-800 text-sm space-y-1">
                      <li>• Review confidence scores</li>
                      <li>• Check source attributions</li>
                      <li>• Cross-reference with original documents</li>
                    </ul>
                  </div>
                  
                  <div className="bg-purple-50 p-4 rounded-lg">
                    <h4 className="font-medium text-purple-900 mb-2">⚡ Performance</h4>
                    <ul className="text-purple-800 text-sm space-y-1">
                      <li>• Process documents in smaller batches</li>
                      <li>• Use appropriate confidence thresholds</li>
                      <li>• Enable auto-processing for efficiency</li>
                    </ul>
                  </div>
                </div>
              </div>
              
              {/* FAQ */}
              <div>
                <div className="flex items-center space-x-2 mb-4">
                  <QuestionMarkCircleIcon className="h-6 w-6 text-orange-600" />
                  <h3 className="text-lg font-semibold text-gray-800">Frequently Asked Questions</h3>
                </div>
                
                <div className="space-y-3">
                  {[
                    {
                      q: "How accurate are the AI responses?",
                      a: "Responses include confidence scores and source attribution. Always verify important legal information with original documents."
                    },
                    {
                      q: "What file formats are supported?",
                      a: "PDF, DOCX, DOC, TXT, RTF for documents; PNG, JPG, JPEG, TIFF for images."
                    },
                    {
                      q: "Can I use this for legal advice?",
                      a: "No, this tool provides legal information only. Always consult qualified legal professionals for advice."
                    },
                    {
                      q: "How are my documents processed?",
                      a: "Documents are processed locally and securely. Text, tables, images, and citations are extracted and analyzed."
                    }
                  ].map((faq, index) => (
                    <div key={index} className="bg-gray-50 p-4 rounded-lg">
                      <h4 className="font-medium text-gray-900 mb-2">{faq.q}</h4>
                      <p className="text-gray-700 text-sm">{faq.a}</p>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* About Tab */}
          {activeTab === "about" && (
            <div className="space-y-6">
              <div className="text-center">
                <div className="bg-blue-100 w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-4">
                  <BeakerIcon className="h-8 w-8 text-blue-600" />
                </div>
                <h3 className="text-xl font-semibold text-gray-800 mb-2">VirLaw AI</h3>
                <p className="text-gray-600">Ultimate Virtual Legal Assistant</p>
                <p className="text-sm text-gray-500 mt-1">Version 3.0 - ULTIMATE Edition</p>
              </div>
              
              {/* System Status */}
              {systemHealth && (
                <div className="bg-gray-50 p-6 rounded-lg">
                  <h4 className="font-semibold text-gray-800 mb-4">System Status</h4>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="flex items-center justify-between">
                      <span className="text-gray-700">System Status</span>
                      <div className="flex items-center space-x-2">
                        <div className={`w-2 h-2 rounded-full ${
                          systemHealth.system_status === "operational" ? "bg-green-500" : "bg-red-500"
                        }`}></div>
                        <span className="text-sm font-medium capitalize">{systemHealth.system_status}</span>
                      </div>
                    </div>
                    
                    <div className="flex items-center justify-between">
                      <span className="text-gray-700">Documents Processed</span>
                      <span className="text-sm font-medium">{systemHealth.uptime_info?.total_documents || 0}</span>
                    </div>
                    
                    <div className="flex items-center justify-between">
                      <span className="text-gray-700">Version</span>
                      <span className="text-sm font-medium">{systemHealth.version}</span>
                    </div>
                    
                    <div className="flex items-center justify-between">
                      <span className="text-gray-700">Last Processing</span>
                      <span className="text-sm font-medium">
                        {systemHealth.uptime_info?.last_processing || "Never"}
                      </span>
                    </div>
                  </div>
                </div>
              )}
              
              {/* Features */}
              <div className="bg-gray-50 p-6 rounded-lg">
                <h4 className="font-semibold text-gray-800 mb-4">Advanced Features</h4>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                  {[
                    "Multi-modal RAG (text, tables, images)",
                    "Google Gemini 2.0 Flash + Pro integration",
                    "Groq Llama 3.3 70B ultra-fast inference",
                    "Advanced legal citation extraction",
                    "Source attribution with confidence scoring",
                    "Real-time performance monitoring",
                    "Batch document processing",
                    "Multiple query analysis types",
                    "ChromaDB vector storage",
                    "Comprehensive error handling",
                    "Multi-language support",
                    "Production-ready API endpoints"
                  ].map((feature, index) => (
                    <div key={index} className="flex items-center space-x-2">
                      <CheckCircleIcon className="h-4 w-4 text-green-500" />
                      <span className="text-sm text-gray-700">{feature}</span>
                    </div>
                  ))}
                </div>
              </div>
              
              {/* Contact Information */}
              <div className="bg-blue-50 p-6 rounded-lg">
                <h4 className="font-semibold text-blue-900 mb-4">Support & Contact</h4>
                <div className="space-y-2">
                  <p className="text-blue-800 text-sm">
                    <strong>Technical Support:</strong> support@virlaw.com
                  </p>
                  <p className="text-blue-800 text-sm">
                    <strong>Documentation:</strong> docs.virlaw.com
                  </p>
                  <p className="text-blue-800 text-sm">
                    <strong>Legal Disclaimer:</strong> This tool provides legal information, not legal advice. 
                    Always consult qualified legal professionals for specific legal matters.
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* API Reference Tab */}
          {activeTab === "api" && (
            <div className="space-y-6">
              <div>
                <h3 className="text-lg font-semibold text-gray-800 mb-4">API Endpoints</h3>
                <div className="space-y-4">
                  {[
                    {
                      method: "POST",
                      endpoint: "/gemini-rag",
                      description: "Enhanced chat endpoint with all advanced features",
                      params: "prompt, include_sources, query_type, confidence_threshold"
                    },
                    {
                      method: "POST", 
                      endpoint: "/ultimate-query",
                      description: "Full advanced query with comprehensive response",
                      params: "question, query_type, max_results, include_citations"
                    },
                    {
                      method: "POST",
                      endpoint: "/process-documents-ultimate",
                      description: "Ultimate document processing with all features",
                      params: "documents_dir, batch_size"
                    },
                    {
                      method: "POST",
                      endpoint: "/upload-documents",
                      description: "Upload multiple documents with progress tracking",
                      params: "files (multipart/form-data)"
                    },
                    {
                      method: "GET",
                      endpoint: "/health",
                      description: "Comprehensive system health and capabilities",
                      params: "None"
                    },
                    {
                      method: "GET",
                      endpoint: "/system-stats",
                      description: "Detailed performance metrics and statistics",
                      params: "None"
                    },
                    {
                      method: "GET",
                      endpoint: "/query-types",
                      description: "Available query types and descriptions",
                      params: "None"
                    },
                    {
                      method: "POST",
                      endpoint: "/search-citations",
                      description: "Search for specific legal citations",
                      params: "query"
                    },
                    {
                      method: "GET",
                      endpoint: "/document-list",
                      description: "List of uploaded and processed documents",
                      params: "None"
                    }
                  ].map((api, index) => (
                    <div key={index} className="border border-gray-200 p-4 rounded-lg">
                      <div className="flex items-center space-x-3 mb-2">
                        <span className={`px-2 py-1 text-xs font-medium rounded ${
                          api.method === "GET" ? "bg-green-100 text-green-800" : "bg-blue-100 text-blue-800"
                        }`}>
                          {api.method}
                        </span>
                        <code className="text-sm font-mono text-gray-800">{api.endpoint}</code>
                      </div>
                      <p className="text-sm text-gray-600 mb-2">{api.description}</p>
                      <p className="text-xs text-gray-500">
                        <strong>Parameters:</strong> {api.params}
                      </p>
                    </div>
                  ))}
                </div>
              </div>
              
              <div className="bg-yellow-50 p-4 rounded-lg">
                <div className="flex items-center space-x-2 mb-2">
                  <InformationCircleIcon className="h-5 w-5 text-yellow-600" />
                  <h4 className="font-medium text-yellow-900">API Usage Notes</h4>
                </div>
                <ul className="text-yellow-800 text-sm space-y-1">
                  <li>• All endpoints require proper CORS headers</li>
                  <li>• Backend runs on http://localhost:8000 by default</li>
                  <li>• Responses include comprehensive metadata and error handling</li>
                  <li>• Upload endpoints support multiple file formats</li>
                  <li>• Real-time endpoints provide streaming capabilities</li>
                </ul>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default UltimateSettingsHelpPage;