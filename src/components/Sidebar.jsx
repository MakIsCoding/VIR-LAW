// src/components/Sidebar.jsx

import React, { useState, useEffect, useRef } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import { signOut } from "firebase/auth";
import { auth, db } from "../firebase";

import {
  collection,
  query,
  orderBy,
  onSnapshot,
  addDoc,
  serverTimestamp,
  doc, // Import doc for specific document operations
  updateDoc, // Import updateDoc for updating documents (pin, rename)
  deleteDoc, // Import deleteDoc for deleting documents
  writeBatch, // Import writeBatch for batch deletions
} from "firebase/firestore";

import {
  ChatBubbleLeftRightIcon,
  HomeIcon,
  Cog6ToothIcon,
  XMarkIcon,
  EllipsisVerticalIcon, // For the context menu trigger
  ShareIcon, // For the Share option in context menu
  MapPinIcon, // THIS IS THE ICON TO USE FOR PINNING
  PencilIcon, // For the Rename option in context menu
  TrashIcon, // For the Delete option in context menu - Also used for hover delete
  // CheckIcon, // For a "select all" or confirmation icon (optional, not strictly used yet) // Removed as not used
  // REMOVED: PushPinIcon - it does not exist in Heroicons v2 by this name
} from "@heroicons/react/24/solid";

// No separate import for PushPinIcon from outline needed, as MapPinIcon will be used

// Define constants for routes for better maintainability and to avoid magic strings
const ROUTES = {
  DASHBOARD: "/dashboard",
  SIGN_IN: "/signin",
  WELCOME: "/dashboard/welcome",
  SETTINGS_HELP: "/dashboard/settings-help",
  NEW_QUERY_ID: "new", // This refers to the URL segment for a new query
};

// Helper component to avoid duplicating the sidebar's internal content
const SidebarContent = ({
  onNewQueryClick,
  recentQueries,
  activeQueryIdFromUrl,
  onRecentQueryClick,
  user,
  handleLinkClick,
  handleSignOut,
  fetchError, // Prop to display fetch errors
  // Props for chat history actions, passed from parent Sidebar component
  handleShareQuery,
  handlePinQuery,
  handleRenameQuery,
  handleDeleteQuery,
  // New props for multiple selection
  selectionMode,
  toggleSelectionMode,
  selectedQueryIds,
  handleSelectQuery,
  handleDeleteSelectedQueries,
}) => {
  const isUserAuthenticated = !!user;

  // State for managing the context menu's visibility, position, and the ID of the query it's opened for
  const [contextMenu, setContextMenu] = useState({
    isVisible: false,
    x: 0,
    y: 0,
    queryId: null,
  });

  // State to track which query item is currently being hovered over for the delete icon
  const [hoveredQueryId, setHoveredQueryId] = useState(null);

  // Ref for the context menu element to detect clicks outside of it
  const contextMenuRef = useRef(null);

  // Effect to add/remove event listener for closing the context menu when clicking outside
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (
        contextMenuRef.current &&
        !contextMenuRef.current.contains(event.target)
      ) {
        setContextMenu({ ...contextMenu, isVisible: false });
      }
    };

    if (contextMenu.isVisible) {
      document.addEventListener("mousedown", handleClickOutside);
    }
    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, [contextMenu]); // Re-run effect if contextMenu state changes

  // Handler for opening the context menu when the ellipsis icon is clicked
  const handleEllipsisClick = (event, queryId) => {
    event.stopPropagation(); // Prevent the parent button's onClick (onRecentQueryClick) from firing
    const buttonRect = event.currentTarget.getBoundingClientRect();
    setContextMenu({
      isVisible: true,
      x: buttonRect.right + 5, // Position menu slightly to the right of the button
      y: buttonRect.top, // Align top of menu with top of the button
      queryId: queryId,
    });
  };

  // Handler for performing an action when a context menu item is clicked
  const handleMenuItemClick = (action, queryId) => {
    setContextMenu({ ...contextMenu, isVisible: false }); // Close the menu after an action is selected
    switch (action) {
      case "share":
        handleShareQuery(queryId);
        break;
      // MODIFICATION: Pass true/false to handlePinQuery for explicit pin/unpin
      case "pin":
        handlePinQuery(queryId, true); // Explicitly pin
        break;
      case "unpin":
        handlePinQuery(queryId, false); // Explicitly unpin
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

  return (
    <>
      {isUserAuthenticated ? (
        <>
          <nav className="flex-grow overflow-y-auto pr-2 custom-scrollbar">
            <ul>
              <li className="mb-2">
                <button
                  onClick={onNewQueryClick}
                  className={`flex items-center w-full p-3 rounded-md transition-colors duration-200 text-blue-300
                    ${
                      activeQueryIdFromUrl === ROUTES.NEW_QUERY_ID
                        ? "bg-blue-700"
                        : "hover:bg-blue-600 bg-blue-700"
                    }`}
                  aria-current={
                    activeQueryIdFromUrl === ROUTES.NEW_QUERY_ID
                      ? "page"
                      : undefined
                  }
                >
                  <ChatBubbleLeftRightIcon className="w-5 h-5 mr-3" />
                  New Query
                </button>
              </li>
              <li className="mb-2">
                <button
                  onClick={() => handleLinkClick(ROUTES.DASHBOARD)}
                  className={`flex items-center w-full p-3 rounded-md hover:bg-gray-700 transition-colors duration-200
                    ${
                      ["", "dashboard", "welcome"].includes(activeQueryIdFromUrl)
                        ? "bg-gray-700"
                        : ""
                    }`}
                  aria-current={
                    ["", "dashboard", "welcome"].includes(activeQueryIdFromUrl)
                      ? "page"
                      : undefined
                  }
                >
                  <HomeIcon className="w-5 h-5 mr-3" />
                  Dashboard
                </button>
              </li>
              <li className="mb-2">
                <button
                  onClick={() => handleLinkClick(ROUTES.WELCOME)}
                  className={`flex items-center w-full p-3 rounded-md hover:bg-gray-700 transition-colors duration-200
                    ${activeQueryIdFromUrl === "welcome" ? "bg-gray-700" : ""}`}
                  aria-current={
                    activeQueryIdFromUrl === "welcome" ? "page" : undefined
                  }
                >
                  <HomeIcon className="w-5 h-5 mr-3" />
                  Welcome
                </button>
              </li>
            </ul>
            <div className="border-t border-gray-600 pt-4 mt-4">
              <div className="flex justify-between items-center mb-3">
                <h3 className="text-sm font-semibold text-gray-400">
                  Recent Cases
                </h3>
                {recentQueries.length > 0 && (
                  <button
                    onClick={toggleSelectionMode}
                    className="text-blue-400 hover:text-blue-300 text-xs px-2 py-1 rounded transition-colors duration-200"
                  >
                    {selectionMode ? "Cancel" : "Select"}
                  </button>
                )}
              </div>

              {/* Delete Selected Button - Moved to the top and centered */}
              {selectionMode && (
                <div className="mb-4">
                  {" "}
                  {/* Add margin-bottom for spacing */}
                  <button
                    onClick={handleDeleteSelectedQueries}
                    className={`w-full bg-red-600 hover:bg-red-700 text-white font-bold py-2 px-4 rounded transition duration-200 flex items-center justify-center gap-2
                      ${
                        selectedQueryIds.length === 0
                          ? "opacity-50 cursor-not-allowed"
                          : ""
                      }`}
                    disabled={selectedQueryIds.length === 0}
                  >
                    <TrashIcon className="w-5 h-5" />
                    Delete Selected ({selectedQueryIds.length})
                  </button>
                </div>
              )}

              {fetchError && (
                // Display fetchError if present
                <p className="text-red-400 text-sm mb-2">{fetchError}</p>
              )}
              {recentQueries.length > 0 ? (
                <ul>
                  {recentQueries.map((queryItem) => (
                    <li
                      key={queryItem.id}
                      className="mb-2"
                      onMouseEnter={() => setHoveredQueryId(queryItem.id)}
                      onMouseLeave={() => setHoveredQueryId(null)}
                    >
                      <div
                        className={`flex items-center justify-between text-sm w-full p-2 rounded-md truncate group
                          ${
                            activeQueryIdFromUrl === queryItem.id
                              ? "bg-gray-700"
                              : "hover:bg-gray-700"
                          }
                          ${selectionMode ? "pl-0" : ""}`}
                      >
                        {selectionMode && (
                          <input
                            type="checkbox"
                            className="form-checkbox h-4 w-4 text-blue-600 bg-gray-800 border-gray-600 rounded mr-2 ml-2 cursor-pointer"
                            checked={selectedQueryIds.includes(queryItem.id)}
                            onChange={() => handleSelectQuery(queryItem.id)}
                            onClick={(e) => e.stopPropagation()} // Prevent parent div's click from selecting chat
                          />
                        )}
                        <button
                          onClick={() =>
                            !selectionMode && onRecentQueryClick(queryItem.id)
                          }
                          className={`flex-grow text-left truncate flex items-center gap-2 ${
                            selectionMode ? "cursor-default" : ""
                          }`}
                          aria-current={
                            activeQueryIdFromUrl === queryItem.id
                              ? "page"
                              : undefined
                          }
                          disabled={selectionMode} // Disable navigation when in selection mode
                        >
                          {/* CORRECTED: Conditionally render MapPinIcon for pinned state */}
                          {queryItem.pinned && (
                            <MapPinIcon className="w-4 h-4 text-blue-400 transform rotate-45" />
                          )}
                          {queryItem.title || "Untitled Query"}
                        </button>
                        <div className="flex items-center">
                          {/* Delete Icon (appears on hover, but not for the currently active query or in selection mode) */}
                          {!selectionMode &&
                            hoveredQueryId === queryItem.id &&
                            activeQueryIdFromUrl !== queryItem.id && (
                              <button
                                onClick={(e) => {
                                  e.stopPropagation(); // Prevent onRecentQueryClick from triggering
                                  handleDeleteQuery(queryItem.id);
                                }}
                                className="ml-2 p-1 rounded-md hover:bg-gray-600 focus:outline-none focus:ring-2 focus:ring-red-500"
                                aria-label={`Delete ${
                                  queryItem.title || "Untitled Query"
                                }`}
                              >
                                <TrashIcon className="w-4 h-4 text-red-400" />
                              </button>
                            )}
                          {/* Ellipsis (More Options) button - hidden in selection mode */}
                          {!selectionMode && (
                            <button
                              onClick={(e) =>
                                handleEllipsisClick(e, queryItem.id)
                              }
                              className={`ml-2 p-1 rounded-md hover:bg-gray-600 focus:outline-none focus:ring-2 focus:ring-blue-500
                                ${
                                  (hoveredQueryId === queryItem.id &&
                                    activeQueryIdFromUrl !== queryItem.id) ||
                                  activeQueryIdFromUrl === queryItem.id
                                    ? "opacity-100"
                                    : "opacity-0 group-hover:opacity-100"
                                }
                                transition-opacity duration-200`}
                              aria-label={`Options for ${
                                queryItem.title || "Untitled Query"
                              }`}
                            >
                              <EllipsisVerticalIcon className="w-4 h-4 text-gray-400" />
                            </button>
                          )}
                        </div>
                      </div>
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="text-gray-500 text-sm">No recent queries.</p>
              )}
            </div>
          </nav>
          <div className="mt-auto pt-4 border-t border-gray-700">
            <ul className="mb-2">
              <li className="mb-2">
                <button
                  onClick={() => handleLinkClick(ROUTES.SETTINGS_HELP)}
                  className={`flex items-center w-full p-3 rounded-md hover:bg-gray-700 transition-colors duration-200
                    ${
                      activeQueryIdFromUrl === "settings-help"
                        ? "bg-gray-700"
                        : ""
                    }`}
                  aria-current={
                    activeQueryIdFromUrl === "settings-help"
                      ? "page"
                      : undefined
                  }
                >
                  <Cog6ToothIcon className="w-5 h-5 mr-3" />
                  Settings & Help
                </button>
              </li>
            </ul>
            <button
              onClick={handleSignOut}
              className="w-full bg-red-600 hover:bg-red-700 text-white font-bold py-2 px-4 rounded transition duration-200 flex items-center justify-center"
            >
              Sign Out
            </button>
            <p className="text-gray-500 text-xs text-center mt-2">
              Version 1.0
            </p>
          </div>
        </>
      ) : (
        <div className="flex-grow flex flex-col justify-between p-4 text-gray-400 text-center text-sm">
          <p>Sign in to access features like new queries and recent cases.</p>
          <div className="mt-auto pt-4 border-t border-gray-700">
            <button
              onClick={() => handleLinkClick(ROUTES.SIGN_IN)}
              className="w-full bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded transition duration-200 flex items-center justify-center mb-2"
            >
              Sign In
            </button>
            <button
              onClick={() => handleLinkClick(ROUTES.SETTINGS_HELP)}
              className="flex items-center w-full p-2 rounded hover:bg-gray-700 text-sm mb-2"
            >
              <Cog6ToothIcon className="w-5 h-5 mr-3" /> Settings & Help
            </button>
          </div>
        </div>
      )}
      {/* Context Menu for Chat History Options */}
      {contextMenu.isVisible && (
        <div
          ref={contextMenuRef}
          className="absolute z-50 bg-gray-700 rounded-md shadow-lg py-1"
          style={{ top: contextMenu.y, left: contextMenu.x }}
        >
          <button
            onClick={() => handleMenuItemClick("share", contextMenu.queryId)}
            className="flex items-center w-full px-4 py-2 text-sm text-gray-200 hover:bg-gray-600"
          >
            <ShareIcon className="w-4 h-4 mr-2" /> Share
          </button>
          {/* ADDITION: Conditional rendering for Pin/Unpin in context menu */}
          {recentQueries.find((q) => q.id === contextMenu.queryId)?.pinned ? (
            <button
              onClick={() => handleMenuItemClick("unpin", contextMenu.queryId)}
              className="flex items-center w-full px-4 py-2 text-sm text-gray-200 hover:bg-gray-600"
            >
              <MapPinIcon className="w-4 h-4 mr-2" /> Unpin
            </button>
          ) : (
            <button
              onClick={() => handleMenuItemClick("pin", contextMenu.queryId)}
              className="flex items-center w-full px-4 py-2 text-sm text-gray-200 hover:bg-gray-600"
            >
              <MapPinIcon className="w-4 h-4 mr-2" /> Pin
            </button>
          )}

          <button
            onClick={() => handleMenuItemClick("rename", contextMenu.queryId)}
            className="flex items-center w-full px-4 py-2 text-sm text-gray-200 hover:bg-gray-600"
          >
            <PencilIcon className="w-4 h-4 mr-2" /> Rename
          </button>
          <button
            onClick={() => handleMenuItemClick("delete", contextMenu.queryId)}
            className="flex items-center w-full px-4 py-2 text-sm text-red-300 hover:bg-gray-600"
          >
            <TrashIcon className="w-4 h-4 mr-2" /> Delete
          </button>
        </div>
      )}
    </>
  );
};

const Sidebar = ({ isSidebarOpen, onToggleSidebar, user }) => {
  const navigate = useNavigate();
  const location = useLocation();
  const currentPathSegments = location.pathname.split("/");
  const activeQueryIdFromUrl =
    currentPathSegments[currentPathSegments.length - 1];

  const [recentQueries, setRecentQueries] = useState([]);
  const [fetchError, setFetchError] = useState(null); // State to handle errors during Firestore fetches
  const currentUserId = user?.uid;

  // New states for multiple selection
  const [selectionMode, setSelectionMode] = useState(false);
  const [selectedQueryIds, setSelectedQueryIds] = useState([]);

  // Effect to fetch recent queries from Firestore in real-time
  useEffect(() => {
    if (!currentUserId) {
      setRecentQueries([]); // Clear queries if no user is logged in
      setFetchError(null); // Clear any previous error
      return;
    }

    const userQuerySessionsRef = collection(
      db,
      "users",
      currentUserId,
      "querySessions"
    );

    // Query to order by 'pinned' status first (true values come before false), then by 'lastUpdated'
    const q = query(
      userQuerySessionsRef,
      orderBy("pinned", "desc"), // Order by pinned status
      orderBy("lastUpdated", "desc")
    );

    // Set up a real-time listener for changes in the query sessions
    const unsubscribe = onSnapshot(
      q,
      (snapshot) => {
        const queriesData = snapshot.docs.map((doc) => ({
          id: doc.id,
          ...doc.data(),
        }));
        setRecentQueries(queriesData);
        setFetchError(null); // Clear error on successful fetch
      },
      (error) => {
        console.error("Error fetching recent queries:", error);
        setFetchError("Failed to load recent queries."); // Set error state if fetch fails
      }
    );

    return () => unsubscribe(); // Cleanup the listener when component unmounts or dependencies change
  }, [currentUserId, db]); // Re-run effect if currentUserId or db changes

  // Function to toggle selection mode
  const toggleSelectionMode = () => {
    setSelectionMode((prevMode) => !prevMode);
    setSelectedQueryIds([]); // Clear selections when entering/exiting selection mode
    // Also close the context menu if it's open when entering selection mode
    // (This is handled within SidebarContent's specific logic, but good to keep in mind)
  };

  // Function to handle individual query selection/deselection
  const handleSelectQuery = (queryId) => {
    setSelectedQueryIds((prevSelected) =>
      prevSelected.includes(queryId)
        ? prevSelected.filter((id) => id !== queryId)
        : [...prevSelected, queryId]
    );
  };

  const handleSignOut = async () => {
    try {
      await signOut(auth);
      navigate(ROUTES.SIGN_IN);
    } catch (error) {
      console.error("Error signing out:", error);
      alert("Failed to sign out. Please try again."); // Provide user feedback
    }
  };

  const handleLinkClick = (path) => {
    navigate(path);
    // Close sidebar on mobile after clicking a link
    if (window.innerWidth < 768) {
      onToggleSidebar();
    }
  };

  const handleNewQueryClick = async () => {
    if (!currentUserId) {
      console.error("No user authenticated to create a new query.");
      navigate(ROUTES.SIGN_IN); // Redirect to login if no user
      return;
    }

    // Exit selection mode if a new query is initiated
    setSelectionMode(false);
    setSelectedQueryIds([]);

    try {
      const newSessionRef = await addDoc(
        collection(db, "users", currentUserId, "querySessions"),
        {
          title: "New Chat", // Initial generic title for the new session
          createdAt: serverTimestamp(),
          lastUpdated: serverTimestamp(),
          pinned: false, // ADDITION: Initialize new queries as not pinned
        }
      );
      navigate(`/dashboard/${newSessionRef.id}`); // Navigate to the newly created session's page
    } catch (error) {
      console.error("Error creating new query session:", error);
      alert("Failed to start a new chat. Please try again.");
    }

    if (window.innerWidth < 768) {
      onToggleSidebar();
    }
  };

  const handleRecentQueryClick = (queryId) => {
    if (!selectionMode) {
      // Only navigate if not in selection mode
      navigate(`/dashboard/${queryId}`); // Navigate to the specific query
      if (window.innerWidth < 768) {
        onToggleSidebar();
      }
    }
  };

  // --- Handlers for Chat History Options (Share, Pin, Rename, Delete) ---

  const handleShareQuery = (queryId) => {
    console.log("Share query:", queryId);
    // TODO: Implement actual sharing logic here (e.g., copy link to clipboard, open share dialog)
    alert(`Share functionality for query ${queryId} is not yet implemented.`);
  };

  // MODIFICATION: Modified handlePinQuery to accept a 'setPinnedTo' argument
  const handlePinQuery = async (queryId, setPinnedTo) => {
    if (!currentUserId) {
      alert("Please sign in to pin queries.");
      return;
    }
    try {
      const queryRef = doc(
        db,
        "users",
        currentUserId,
        "querySessions",
        queryId
      );
      // The `setPinnedTo` argument allows explicit pinning/unpinning from the context menu
      // If it's undefined (e.g., from an old call or direct toggle), we'll toggle the existing state.
      const currentQuery = recentQueries.find((q) => q.id === queryId);
      const newPinnedStatus =
        setPinnedTo !== undefined
          ? setPinnedTo
          : currentQuery
          ? !currentQuery.pinned
          : false; // Default to false if query not found

      await updateDoc(queryRef, {
        pinned: newPinnedStatus, // Use the determined new status
        lastUpdated: serverTimestamp(), // Update timestamp to re-sort (pinned items will float to top)
      });
      alert(
        `${currentQuery?.title || "Untitled Query"} ${
          newPinnedStatus ? "pinned" : "unpinned"
        } successfully!`
      );
    } catch (error) {
      console.error("Error pinning/unpinning query:", error);
      alert("Failed to pin/unpin query. Please try again.");
    }
  };

  const handleRenameQuery = async (queryId) => {
    if (!currentUserId) {
      alert("Please sign in to rename queries.");
      return;
    }
    const currentQuery = recentQueries.find((q) => q.id === queryId);
    const oldTitle = currentQuery?.title || "Untitled Query";
    // Using a simple prompt. For a better UX, consider a modal or inline editing.
    const newTitle = prompt("Enter new title for the query:", oldTitle);

    if (
      newTitle !== null &&
      newTitle.trim() !== "" &&
      newTitle.trim() !== oldTitle
    ) {
      try {
        const queryRef = doc(
          db,
          "users",
          currentUserId,
          "querySessions",
          queryId
        );
        await updateDoc(queryRef, {
          title: newTitle.trim(),
          lastUpdated: serverTimestamp(), // Update timestamp to re-sort if necessary
        });
        alert("Query renamed successfully!");
      } catch (error) {
        console.error("Error renaming query:", error);
        alert("Failed to rename query. Please try again.");
      }
    } else if (newTitle !== null && newTitle.trim() === oldTitle) {
      alert("No change, title is the same.");
    } else if (newTitle !== null && newTitle.trim() === "") {
      alert("Query title cannot be empty.");
    }
    // If newTitle is null, the user cancelled the prompt.
  };

  const handleDeleteQuery = async (queryId) => {
    if (!currentUserId) {
      alert("Please sign in to delete queries.");
      return;
    }
    const confirmDelete = window.confirm(
      "Are you sure you want to delete this query session? This action cannot be undone."
    );

    if (confirmDelete) {
      try {
        await deleteDoc(
          doc(db, "users", currentUserId, "querySessions", queryId)
        );
        alert("Query deleted successfully!");
        // If the deleted query was the one currently open, navigate to the dashboard
        if (`/dashboard/${queryId}` === location.pathname) {
          navigate(ROUTES.DASHBOARD);
        }
      } catch (error) {
        console.error("Error deleting query:", error);
        alert("Failed to delete query. Please try again.");
      }
    }
  };

  // New function to handle deleting multiple selected queries
  const handleDeleteSelectedQueries = async () => {
    if (!currentUserId) {
      alert("Please sign in to delete queries.");
      return;
    }
    if (selectedQueryIds.length === 0) {
      alert("No queries selected for deletion.");
      return;
    }

    const confirmDelete = window.confirm(
      `Are you sure you want to delete ${selectedQueryIds.length} selected query session(s)? This action cannot be undone.`
    );

    if (confirmDelete) {
      try {
        const batch = writeBatch(db); // Initialize a new batch

        selectedQueryIds.forEach((queryId) => {
          const queryRef = doc(
            db,
            "users",
            currentUserId,
            "querySessions",
            queryId
          );
          batch.delete(queryRef); // Add delete operation to the batch
        });

        await batch.commit(); // Commit the batch deletion

        alert(
          `${selectedQueryIds.length} query session(s) deleted successfully!`
        );
        setSelectedQueryIds([]); // Clear selection
        setSelectionMode(false); // Exit selection mode

        // If the currently active query was deleted, navigate to the dashboard
        if (selectedQueryIds.includes(activeQueryIdFromUrl)) {
          navigate(ROUTES.DASHBOARD);
        }
      } catch (error) {
        console.error("Error deleting selected queries:", error);
        setFetchError("Failed to delete selected queries."); // Update fetchError for UI display
        alert("Failed to delete selected queries. Please try again.");
      }
    }
  };

  return (
    <>
      {/* Mobile Sidebar (Fixed, slides in/out, hidden on md+) */}
      <div
        className={`
          fixed inset-y-0 left-0 w-64 bg-gray-900 text-white p-4 flex-col z-40 h-screen
          transition-transform duration-300 ease-in-out
          ${isSidebarOpen ? "translate-x-0" : "-translate-x-full"}
          md:hidden
          ${
            !isSidebarOpen && "pointer-events-none"
          } {/* Prevents interaction when off-screen */}
        `}
      >
        <div className="flex items-center justify-between mb-6">
          <h1 className="text-2xl font-bold">VirLaw</h1>
          <button
            onClick={onToggleSidebar}
            className="p-2 rounded-md hover:bg-gray-700 focus:outline-none focus:ring-2 focus:ring-blue-500"
            aria-label="Close Sidebar"
          >
            <XMarkIcon className="w-6 h-6" />
          </button>
        </div>
        <SidebarContent
          onNewQueryClick={handleNewQueryClick}
          recentQueries={recentQueries}
          activeQueryIdFromUrl={activeQueryIdFromUrl}
          onRecentQueryClick={handleRecentQueryClick}
          user={user}
          handleLinkClick={handleLinkClick}
          handleSignOut={handleSignOut}
          fetchError={fetchError}
          handleShareQuery={handleShareQuery}
          handlePinQuery={handlePinQuery}
          handleRenameQuery={handleRenameQuery}
          handleDeleteQuery={handleDeleteQuery}
          // Pass new multi-selection props
          selectionMode={selectionMode}
          toggleSelectionMode={toggleSelectionMode}
          selectedQueryIds={selectedQueryIds}
          handleSelectQuery={handleSelectQuery}
          handleDeleteSelectedQueries={handleDeleteSelectedQueries}
        />
      </div>

      {/* Desktop Sidebar (Relative, pushes content, hidden on small screens) */}
      <div
        className={`
          hidden md:flex flex-shrink-0 bg-gray-900 text-white p-4 flex-col z-10 h-screen
          transition-all duration-300 ease-in-out
          ${isSidebarOpen ? "w-64" : "w-0 overflow-hidden"}
        `}
      >
        {isSidebarOpen && (
          <>
            <div className="flex items-center justify-between mb-6">
              <h1 className="text-2xl font-bold">VirLaw</h1>
            </div>
            <SidebarContent
              onNewQueryClick={handleNewQueryClick}
              recentQueries={recentQueries}
              activeQueryIdFromUrl={activeQueryIdFromUrl}
              onRecentQueryClick={handleRecentQueryClick}
              user={user}
              handleLinkClick={handleLinkClick}
              handleSignOut={handleSignOut}
              fetchError={fetchError}
              handleShareQuery={handleShareQuery}
              handlePinQuery={handlePinQuery}
              handleRenameQuery={handleRenameQuery}
              handleDeleteQuery={handleDeleteQuery}
              // Pass new multi-selection props
              selectionMode={selectionMode}
              toggleSelectionMode={toggleSelectionMode}
              selectedQueryIds={selectedQueryIds}
              handleSelectQuery={handleSelectQuery}
              handleDeleteSelectedQueries={handleDeleteSelectedQueries}
            />
          </>
        )}
      </div>
    </>
  );
};

export default Sidebar;