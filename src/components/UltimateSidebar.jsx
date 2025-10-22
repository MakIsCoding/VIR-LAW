// FIXED UltimateSidebar.jsx - All advanced features preserved with fixes
import React, { useState, useEffect, useRef } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import { signOut } from "firebase/auth";
import { auth, db } from "../firebase";
import {
  collection, query, orderBy, onSnapshot, addDoc, serverTimestamp, doc,
  updateDoc, deleteDoc, writeBatch,
} from "firebase/firestore";
import axios from "axios";
import { API_BASE_URL } from '../config/api';
import {
  ChatBubbleLeftRightIcon, HomeIcon, Cog6ToothIcon, XMarkIcon,
  EllipsisVerticalIcon, ShareIcon, MapPinIcon, PencilIcon, TrashIcon,
  DocumentTextIcon, ServerIcon, ChartBarIcon, CloudArrowUpIcon,
  BeakerIcon, RocketLaunchIcon, ExclamationTriangleIcon, CheckCircleIcon,
  ClockIcon, BoltIcon, CpuChipIcon
} from "@heroicons/react/24/solid";

// Define constants for routes
const ROUTES = {
  DASHBOARD: "/dashboard",
  SIGN_IN: "/signin",
  WELCOME: "/dashboard/welcome",
  SETTINGS_HELP: "/dashboard/settings-help",
  DOCUMENT_MANAGEMENT: "/dashboard/documents",
  SYSTEM_DASHBOARD: "/dashboard/system",
  NEW_QUERY_ID: "new",
};

const UltimateSidebar = ({
  user,
  isSidebarOpen,
  setIsSidebarOpen,
  onNewQueryClick,
  recentQueries,
  onRecentQueryClick,
  loadingRecentQueries,
  activeQueryIdFromUrl,
}) => {
  const navigate = useNavigate();
  const location = useLocation();
  
  // Enhanced state for advanced features
  const [contextMenu, setContextMenu] = useState({
    isVisible: false,
    x: 0,
    y: 0,
    queryId: null,
  });
  const [hoveredQueryId, setHoveredQueryId] = useState(null);
  const [showSystemStatus, setShowSystemStatus] = useState(true);
  const [systemHealth, setSystemHealth] = useState(null);
  const [processingStats, setProcessingStats] = useState(null);
  const [realTimeMetrics, setRealTimeMetrics] = useState({});
  const [selectionMode, setSelectionMode] = useState(false);
  const [selectedQueryIds, setSelectedQueryIds] = useState(new Set());

  const contextMenuRef = useRef(null);
  const isUserAuthenticated = !!user;

  // Initialize system monitoring
  useEffect(() => {
    if (isUserAuthenticated) {
      loadSystemHealth();
      setupRealTimeMonitoring();
    }
  }, [isUserAuthenticated]);

  // Load system health
  const loadSystemHealth = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/health`);
      setSystemHealth(response.data);
    } catch (error) {
      console.error("Error loading system health:", error);
      setSystemHealth({ status: "offline", message: "Backend unavailable" });
    }
  };

  // Setup real-time monitoring
  const setupRealTimeMonitoring = () => {
    const interval = setInterval(async () => {
      try {
        const [healthResponse, statsResponse] = await Promise.all([
          axios.get(`${API_BASE_URL}/health`),
          axios.get(`${API_BASE_URL}/system-stats`)          
        ]);
        setSystemHealth(healthResponse.data);
        setProcessingStats(statsResponse.data);
      } catch (error) {
        console.error("Error in real-time monitoring:", error);
      }
    }, 30000);

    return () => clearInterval(interval);
  };

  // Handle clicks outside context menu
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (contextMenuRef.current && !contextMenuRef.current.contains(event.target)) {
        setContextMenu({ ...contextMenu, isVisible: false });
      }
    };

    if (contextMenu.isVisible) {
      document.addEventListener("mousedown", handleClickOutside);
    }
    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, [contextMenu]);

  // Handle navigation
  const handleLinkClick = (path) => {
    navigate(path);
    if (window.innerWidth < 1024) {
      setIsSidebarOpen(false);
    }
  };

  // Handle sign out
  const handleSignOut = async () => {
    try {
      await signOut(auth);
      navigate("/signin");
      setIsSidebarOpen(false);
    } catch (error) {
      console.error("Error signing out:", error);
    }
  };

  // Enhanced query management functions
  const handleShareQuery = (queryId) => {
    // Copy share link to clipboard
    const shareUrl = `${window.location.origin}/dashboard/${queryId}`;
    navigator.clipboard.writeText(shareUrl).then(() => {
      console.log("Share link copied to clipboard");
    });
  };

  const handlePinQuery = async (queryId, pin) => {
    try {
      const queryDocRef = doc(db, "users", user.uid, "querySessions", queryId);
      await updateDoc(queryDocRef, {
        pinned: pin,
        lastUpdated: serverTimestamp()
      });
    } catch (error) {
      console.error("Error pinning query:", error);
    }
  };

  const handleRenameQuery = async (queryId) => {
    const newTitle = prompt("Enter new title:");
    if (newTitle && newTitle.trim()) {
      try {
        const queryDocRef = doc(db, "users", user.uid, "querySessions", queryId);
        await updateDoc(queryDocRef, {
          title: newTitle.trim(),
          lastUpdated: serverTimestamp()
        });
      } catch (error) {
        console.error("Error renaming query:", error);
      }
    }
  };

  const handleDeleteQuery = async (queryId) => {
    if (confirm("Are you sure you want to delete this query?")) {
      try {
        const queryDocRef = doc(db, "users", user.uid, "querySessions", queryId);
        await deleteDoc(queryDocRef);
      } catch (error) {
        console.error("Error deleting query:", error);
      }
    }
  };

  const handleDeleteSelectedQueries = async () => {
    if (selectedQueryIds.size === 0) return;
    
    if (confirm(`Are you sure you want to delete ${selectedQueryIds.size} selected queries?`)) {
      try {
        const batch = writeBatch(db);
        selectedQueryIds.forEach(queryId => {
          const queryDocRef = doc(db, "users", user.uid, "querySessions", queryId);
          batch.delete(queryDocRef);
        });
        await batch.commit();
        setSelectedQueryIds(new Set());
        setSelectionMode(false);
      } catch (error) {
        console.error("Error deleting selected queries:", error);
      }
    }
  };

  // Handle ellipsis click for context menu
  const handleEllipsisClick = (event, queryId) => {
    event.stopPropagation();
    const buttonRect = event.currentTarget.getBoundingClientRect();
    setContextMenu({
      isVisible: true,
      x: buttonRect.right + 5,
      y: buttonRect.top,
      queryId: queryId,
    });
  };

  // Handle context menu item clicks
  const handleMenuItemClick = (action, queryId) => {
    setContextMenu({ ...contextMenu, isVisible: false });
    switch (action) {
      case "share":
        handleShareQuery(queryId);
        break;
      case "pin":
        handlePinQuery(queryId, true);
        break;
      case "unpin":
        handlePinQuery(queryId, false);
        break;
      case "rename":
        handleRenameQuery(queryId);
        break;
      case "delete":
        handleDeleteQuery(queryId);
        break;
      default:
        break;
    }
  };

  // Get status indicator color
  const getStatusColor = (status) => {
    switch (status) {
      case 'operational':
      case 'healthy':
      case 'ready':
        return 'green';
      case 'processing':
      case 'warning':
        return 'yellow';
      case 'error':
      case 'failed':
      case 'offline':
        return 'red';
      default:
        return 'gray';
    }
  };

  // Format number with commas
  const formatNumber = (num) => {
    return new Intl.NumberFormat().format(num || 0);
  };

  return (
    <>
      {/* Mobile Backdrop */}
      {isSidebarOpen && (
        <div
          className="fixed inset-0 bg-black bg-opacity-50 z-40 lg:hidden"
          onClick={() => setIsSidebarOpen(false)}
        />
      )}

      {/* Sidebar */}
      <div
        className={`fixed lg:static inset-y-0 left-0 z-50 lg:z-auto w-80 bg-white border-r border-gray-200 transform transition-transform duration-300 ease-in-out flex flex-col ${
          isSidebarOpen ? "translate-x-0" : "-translate-x-full lg:translate-x-0"
        }`}
      >
        {/* Mobile Close Button */}
        <div className="lg:hidden absolute top-4 right-4">
          <button
            onClick={() => setIsSidebarOpen(false)}
            className="p-2 text-gray-600 hover:text-gray-800 hover:bg-gray-100 rounded-md"
          >
            <XMarkIcon className="h-5 w-5" />
          </button>
        </div>

        {isUserAuthenticated ? (
          <>
            {/* Enhanced Header with System Status */}
            <div className="p-4 border-b border-gray-200">
              <div className="flex items-center justify-between mb-2">
                <div>
                  <h2 className="text-lg font-semibold text-gray-800">VirLaw AI</h2>
                  <p className="text-sm text-gray-600">Ultimate Virtual Legal Assistant</p>
                </div>
                
                {/* System Status Indicator */}
                {systemHealth && showSystemStatus && (
                  <div className={`w-3 h-3 rounded-full ${
                    systemHealth.status === 'healthy' ? 'bg-green-500' : 
                    systemHealth.status === 'offline' ? 'bg-red-500' : 'bg-yellow-500'
                  } animate-pulse`} title={`Backend: ${systemHealth.status}`} />
                )}
              </div>

              {/* User Info */}
              <div className="flex items-center space-x-2 text-sm text-gray-600">
                <div className="w-6 h-6 bg-blue-600 rounded-full flex items-center justify-center">
                  <span className="text-white text-xs font-medium">
                    {(user.displayName || user.email)?.[0]?.toUpperCase()}
                  </span>
                </div>
                <span className="truncate">{user.displayName || user.email}</span>
              </div>
            </div>

            {/* Navigation */}
            <nav className="flex-1 px-4 py-6 space-y-2 overflow-y-auto">
              {/* New Query Button */}
              <button
                onClick={onNewQueryClick}
                className="w-full flex items-center space-x-3 px-3 py-2 text-left bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
              >
                <ChatBubbleLeftRightIcon className="h-5 w-5" />
                <span className="font-medium">New Query</span>
              </button>

              {/* Navigation Links */}
              <div className="space-y-1 mt-4">
                <button
                  onClick={() => handleLinkClick(ROUTES.WELCOME)}
                  className={`w-full flex items-center space-x-3 px-3 py-2 rounded-lg transition-colors ${
                    location.pathname === ROUTES.WELCOME
                      ? "bg-blue-100 text-blue-700"
                      : "text-gray-700 hover:bg-gray-100"
                  }`}
                >
                  <HomeIcon className="h-5 w-5" />
                  <span>Welcome</span>
                </button>
                
                <button
                  onClick={() => handleLinkClick(ROUTES.DOCUMENT_MANAGEMENT)}
                  className={`w-full flex items-center space-x-3 px-3 py-2 rounded-lg transition-colors ${
                    location.pathname === ROUTES.DOCUMENT_MANAGEMENT
                      ? "bg-blue-100 text-blue-700"
                      : "text-gray-700 hover:bg-gray-100"
                  }`}
                >
                  <DocumentTextIcon className="h-5 w-5" />
                  <span>Documents</span>
                </button>
                
                <button
                  onClick={() => handleLinkClick(ROUTES.SYSTEM_DASHBOARD)}
                  className={`w-full flex items-center space-x-3 px-3 py-2 rounded-lg transition-colors ${
                    location.pathname === ROUTES.SYSTEM_DASHBOARD
                      ? "bg-blue-100 text-blue-700"
                      : "text-gray-700 hover:bg-gray-100"
                  }`}
                >
                  <ChartBarIcon className="h-5 w-5" />
                  <span>System Dashboard</span>
                </button>
                
                <button
                  onClick={() => handleLinkClick(ROUTES.SETTINGS_HELP)}
                  className={`w-full flex items-center space-x-3 px-3 py-2 rounded-lg transition-colors ${
                    location.pathname === ROUTES.SETTINGS_HELP
                      ? "bg-blue-100 text-blue-700"
                      : "text-gray-700 hover:bg-gray-100"
                  }`}
                >
                  <Cog6ToothIcon className="h-5 w-5" />
                  <span>Settings & Help</span>
                </button>
              </div>

              {/* System Status Panel */}
              {systemHealth && showSystemStatus && (
                <div className="mt-6 p-3 bg-gray-50 rounded-lg">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-xs font-medium text-gray-600">System Status</span>
                    <button
                      onClick={() => setShowSystemStatus(false)}
                      className="text-gray-400 hover:text-gray-600"
                    >
                      <XMarkIcon className="h-3 w-3" />
                    </button>
                  </div>
                  <div className="space-y-1 text-xs">
                    <div className={`flex items-center space-x-2 text-${getStatusColor(systemHealth.status)}-600`}>
                      <div className={`w-2 h-2 rounded-full bg-${getStatusColor(systemHealth.status)}-500`} />
                      <span>Backend: {systemHealth.status}</span>
                    </div>
                    {processingStats && (
                      <>
                        <div className="text-gray-600">
                          Documents: {formatNumber(processingStats.processing_stats?.total_documents)}
                        </div>
                        <div className="text-gray-600">
                          Chunks: {formatNumber(processingStats.processing_stats?.total_chunks)}
                        </div>
                      </>
                    )}
                  </div>
                </div>
              )}

              {/* Recent Cases */}
              <div className="mt-6">
                <div className="flex items-center justify-between mb-3">
                  <h3 className="text-sm font-medium text-gray-600">Recent Cases</h3>
                  {recentQueries.length > 0 && (
                    <div className="flex items-center space-x-1">
                      <button
                        onClick={() => setSelectionMode(!selectionMode)}
                        className="p-1 text-gray-400 hover:text-gray-600 text-xs"
                      >
                        {selectionMode ? "Cancel" : "Select"}
                      </button>
                      {selectionMode && selectedQueryIds.size > 0 && (
                        <button
                          onClick={handleDeleteSelectedQueries}
                          className="p-1 text-red-500 hover:text-red-600"
                        >
                          <TrashIcon className="h-3 w-3" />
                        </button>
                      )}
                    </div>
                  )}
                </div>

                <div className="space-y-1 max-h-96 overflow-y-auto">
                  {loadingRecentQueries ? (
                    <div className="text-center py-4">
                      <ClockIcon className="h-5 w-5 text-gray-300 mx-auto mb-1 animate-spin" />
                      <p className="text-xs text-gray-500">Loading...</p>
                    </div>
                  ) : recentQueries.length > 0 ? (
                    recentQueries.map((queryItem) => {
                      const isActive = activeQueryIdFromUrl === queryItem.id;
                      const isPinned = queryItem.pinned;
                      const isSelected = selectedQueryIds.has(queryItem.id);
                      
                      return (
                        <div
                          key={queryItem.id}
                          className={`group relative ${
                            isActive
                              ? "bg-blue-100 border-blue-300"
                              : "bg-gray-50 hover:bg-gray-100 border-gray-200"
                          } border rounded-lg transition-colors`}
                          onMouseEnter={() => setHoveredQueryId(queryItem.id)}
                          onMouseLeave={() => setHoveredQueryId(null)}
                        >
                          <div className="flex items-center">
                            {selectionMode && (
                              <div className="p-2">
                                <input
                                  type="checkbox"
                                  checked={isSelected}
                                  onChange={(e) => {
                                    const newSelected = new Set(selectedQueryIds);
                                    if (e.target.checked) {
                                      newSelected.add(queryItem.id);
                                    } else {
                                      newSelected.delete(queryItem.id);
                                    }
                                    setSelectedQueryIds(newSelected);
                                  }}
                                  className="h-3 w-3"
                                />
                              </div>
                            )}
                            
                            <button
                              onClick={() => onRecentQueryClick(queryItem.id)}
                              className="flex-1 text-left p-3 focus:outline-none"
                            >
                              <div className="flex items-start justify-between">
                                <div className="flex-1 min-w-0">
                                  <p className={`text-sm font-medium truncate ${
                                    isActive ? "text-blue-900" : "text-gray-800"
                                  }`}>
                                    {isPinned && <MapPinIcon className="h-3 w-3 inline mr-1" />}
                                    {queryItem.title}
                                  </p>
                                  <p className={`text-xs truncate ${
                                    isActive ? "text-blue-700" : "text-gray-600"
                                  }`}>
                                    {queryItem.lastUpdated
                                      ? new Date(queryItem.lastUpdated.toDate()).toLocaleDateString()
                                      : "No date"}
                                  </p>
                                </div>
                                
                                {(hoveredQueryId === queryItem.id || contextMenu.queryId === queryItem.id) && (
                                  <button
                                    onClick={(e) => handleEllipsisClick(e, queryItem.id)}
                                    className="p-1 text-gray-400 hover:text-gray-600 hover:bg-white rounded"
                                  >
                                    <EllipsisVerticalIcon className="h-4 w-4" />
                                  </button>
                                )}
                              </div>
                            </button>
                          </div>
                        </div>
                      );
                    })
                  ) : (
                    <div className="text-center py-6">
                      <ChatBubbleLeftRightIcon className="h-8 w-8 text-gray-300 mx-auto mb-2" />
                      <p className="text-sm text-gray-500">No recent cases yet</p>
                      <p className="text-xs text-gray-400">Start a new query to begin</p>
                    </div>
                  )}
                </div>
              </div>
            </nav>

            {/* Footer */}
            <div className="border-t border-gray-200 p-4">
              <div className="text-xs text-gray-500 mb-3">
                Version 3.0 Ultimate • All Features Unlocked
              </div>
              <button
                onClick={handleSignOut}
                className="w-full px-3 py-2 text-sm text-gray-700 hover:bg-gray-100 rounded-lg transition-colors text-left"
              >
                Sign Out
              </button>
            </div>
          </>
        ) : (
          /* Guest State */
          <div className="flex flex-col h-full">
            <div className="p-4 border-b border-gray-200">
              <h2 className="text-lg font-semibold text-gray-800">VirLaw AI</h2>
              <p className="text-sm text-gray-600">Ultimate Virtual Legal Assistant</p>
            </div>
            
            <div className="flex-1 flex items-center justify-center p-4">
              <div className="text-center">
                <ChatBubbleLeftRightIcon className="h-12 w-12 text-blue-500 mx-auto mb-4" />
                <h3 className="text-lg font-medium text-gray-800 mb-2">Welcome to VirLaw AI</h3>
                <p className="text-sm text-gray-600 mb-4">
                  Sign in to access advanced legal AI features, document processing, and case management.
                </p>
                <button
                  onClick={() => handleLinkClick("/signin")}
                  className="w-full px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
                >
                  Sign In
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Context Menu */}
        {contextMenu.isVisible && (
          <div
            ref={contextMenuRef}
            className="fixed bg-white border border-gray-200 rounded-lg shadow-lg py-2 z-50"
            style={{ left: contextMenu.x, top: contextMenu.y }}
          >
            <button
              onClick={() => handleMenuItemClick("share", contextMenu.queryId)}
              className="w-full text-left px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 flex items-center space-x-2"
            >
              <ShareIcon className="h-4 w-4" />
              <span>Share</span>
            </button>
            <button
              onClick={() => handleMenuItemClick("pin", contextMenu.queryId)}
              className="w-full text-left px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 flex items-center space-x-2"
            >
              <MapPinIcon className="h-4 w-4" />
              <span>Pin</span>
            </button>
            <button
              onClick={() => handleMenuItemClick("rename", contextMenu.queryId)}
              className="w-full text-left px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 flex items-center space-x-2"
            >
              <PencilIcon className="h-4 w-4" />
              <span>Rename</span>
            </button>
            <button
              onClick={() => handleMenuItemClick("delete", contextMenu.queryId)}
              className="w-full text-left px-4 py-2 text-sm text-red-700 hover:bg-red-50 flex items-center space-x-2"
            >
              <TrashIcon className="h-4 w-4" />
              <span>Delete</span>
            </button>
          </div>
        )}
      </div>
    </>
  );
};

export default UltimateSidebar;