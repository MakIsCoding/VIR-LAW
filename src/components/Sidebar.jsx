// src/components/Sidebar.jsx

import React, { useState, useEffect } from "react";

import { useNavigate, useLocation } from "react-router-dom";

import { signOut } from "firebase/auth";

import { auth, db } from "../firebase"; // Import db from firebase

import {
  collection,
  query,
  orderBy,
  onSnapshot,
  addDoc,
  serverTimestamp,
} from "firebase/firestore"; // Import Firestore methods

import {
  ChatBubbleLeftRightIcon,
  HomeIcon,
  Cog6ToothIcon,
  XMarkIcon,
} from "@heroicons/react/24/solid";

// Helper component to avoid duplicating the sidebar's internal content

const SidebarContent = ({
  onNewQueryClick,

  recentQueries,

  activeQueryIdFromUrl, // This will be queryId from URL params

  onRecentQueryClick,

  user,

  handleLinkClick,

  handleSignOut,
}) => {
  const isUserAuthenticated = !!user; // Define the temporary ID used for new queries in App.jsx and QueryPage.jsx

  const NEW_QUERY_ROUTE_ID = "new"; // This refers to the URL segment, not a Firestore ID

  return (
    <>
      {" "}
      {isUserAuthenticated ? (
        <>
          {" "}
          <nav className="flex-grow overflow-y-auto pr-2 custom-scrollbar">
            {" "}
            <ul>
              {" "}
              <li className="mb-2">
                {" "}
                <button
                  onClick={onNewQueryClick} // This will now trigger the Firestore creation
                  className={`flex items-center w-full p-3 rounded-md transition-colors duration-200 text-blue-300

          ${
            activeQueryIdFromUrl === NEW_QUERY_ROUTE_ID
              ? "bg-blue-700" // Active style for "New Query" when on /dashboard/new
              : "hover:bg-blue-600 bg-blue-700" // Default style
          }

         `}
                >
                  {" "}
                  <ChatBubbleLeftRightIcon className="w-5 h-5 mr-3" />
                  New Query{" "}
                </button>{" "}
              </li>{" "}
              <li className="mb-2">
                {" "}
                <button
                  onClick={() => handleLinkClick("/dashboard")}
                  className={`flex items-center w-full p-3 rounded-md hover:bg-gray-700 transition-colors duration-200

          ${
            activeQueryIdFromUrl === "" ||
            activeQueryIdFromUrl === "dashboard" ||
            activeQueryIdFromUrl === "welcome" // Assuming /dashboard and /dashboard/welcome are part of the initial dashboard view
              ? "bg-gray-700"
              : ""
          }

         `}
                >
                  <HomeIcon className="w-5 h-5 mr-3" />
                  Dashboard{" "}
                </button>{" "}
              </li>{" "}
              <li className="mb-2">
                {" "}
                <button
                  onClick={() => handleLinkClick("/dashboard/welcome")}
                  className={`flex items-center w-full p-3 rounded-md hover:bg-gray-700 transition-colors duration-200

          ${activeQueryIdFromUrl === "welcome" ? "bg-gray-700" : ""}

         `}
                >
                  <HomeIcon className="w-5 h-5 mr-3" />
                  Welcome{" "}
                </button>{" "}
              </li>{" "}
            </ul>{" "}
            <div className="border-t border-gray-600 pt-4 mt-4">
              {" "}
              <h3 className="text-sm font-semibold text-gray-400 mb-3">
                Recent Cases{" "}
              </h3>{" "}
              {recentQueries.length > 0 ? (
                <ul>
                  {" "}
                  {recentQueries.map(
                    (
                      queryItem // Changed 'query' to 'queryItem' to avoid conflict with the 'query' import from firestore
                    ) => (
                      <li key={queryItem.id} className="mb-2">
                        {" "}
                        <button
                          onClick={() => onRecentQueryClick(queryItem.id)}
                          className={`

             text-sm w-full text-left p-2 rounded-md truncate

             ${
               activeQueryIdFromUrl === queryItem.id
                 ? "bg-gray-700"
                 : "hover:bg-gray-700"
             }

            `}
                        >
                          {" "}
                          {queryItem.title || "Untitled Query"}{" "}
                        </button>{" "}
                      </li>
                    )
                  )}{" "}
                </ul>
              ) : (
                <p className="text-gray-500 text-sm">No recent queries.</p>
              )}{" "}
            </div>{" "}
          </nav>
          {/* This div now holds Settings & Help and Sign Out */}{" "}
          <div className="mt-auto pt-4 border-t border-gray-700">
            {" "}
            <ul className="mb-2">
              {" "}
              <li className="mb-2">
                {" "}
                <button
                  onClick={() => handleLinkClick("/dashboard/settings-help")}
                  className={`flex items-center w-full p-3 rounded-md hover:bg-gray-700 transition-colors duration-200

          ${activeQueryIdFromUrl === "settings-help" ? "bg-gray-700" : ""}

         `}
                >
                  <Cog6ToothIcon className="w-5 h-5 mr-3" />
                  Settings & Help{" "}
                </button>{" "}
              </li>{" "}
            </ul>{" "}
            <button
              onClick={handleSignOut}
              className="w-full bg-red-600 hover:bg-red-700 text-white font-bold py-2 px-4 rounded transition duration-200 flex items-center justify-center"
            >
              Sign Out{" "}
            </button>{" "}
            <p className="text-gray-500 text-xs text-center mt-2">
              Version 1.0{" "}
            </p>{" "}
          </div>{" "}
        </>
      ) : (
        <div className="flex-grow flex flex-col justify-between p-4 text-gray-400 text-center text-sm">
          {" "}
          <p>
            Sign in to access features like new queries and recent cases.
          </p>{" "}
          <div className="mt-auto pt-4 border-t border-gray-700">
            {" "}
            <button
              onClick={() => handleLinkClick("/signin")}
              className="w-full bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded transition duration-200 flex items-center justify-center mb-2"
            >
              Sign In{" "}
            </button>{" "}
            <button
              onClick={() => handleLinkClick("/dashboard/settings-help")}
              className="flex items-center w-full p-2 rounded hover:bg-gray-700 text-sm mb-2"
            >
              <Cog6ToothIcon className="w-5 h-5 mr-3" /> Settings & Help{" "}
            </button>{" "}
          </div>{" "}
        </div>
      )}{" "}
    </>
  );
};

const Sidebar = ({
  isSidebarOpen,

  onToggleSidebar,

  user, // User prop is essential for fetching user-specific data
}) => {
  const navigate = useNavigate();

  const location = useLocation(); // Hook to get current URL

  const currentPathSegments = location.pathname.split("/"); // activeQueryIdFromUrl will be "new" or a Firestore ID, or "" for /dashboard, or "welcome" for /dashboard/welcome etc.

  const activeQueryIdFromUrl =
    currentPathSegments[currentPathSegments.length - 1];

  const [recentQueries, setRecentQueries] = useState([]);

  const currentUserId = user?.uid; // Get UID safely from user prop // Fetch recent queries for the sidebar

  useEffect(() => {
    if (!currentUserId) {
      setRecentQueries([]); // Clear recent queries if no user is logged in

      return;
    } // Reference to the user's querySessions subcollection

    const userQuerySessionsRef = collection(
      db,

      "users",

      currentUserId,

      "querySessions"
    ); // Create a query to order by lastUpdated (most recent first)

    const q = query(
      userQuerySessionsRef,

      orderBy("lastUpdated", "desc") // Sort by the 'lastUpdated' field descending
    ); // Set up a real-time listener

    const unsubscribe = onSnapshot(
      q,

      (snapshot) => {
        const queriesData = snapshot.docs.map((doc) => ({
          id: doc.id,

          ...doc.data(),
        }));

        setRecentQueries(queriesData);
      },

      (error) => {
        console.error("Error fetching recent queries:", error); // Handle error, e.g., display a message to the user
      }
    ); // Cleanup function for the listener

    return () => unsubscribe();
  }, [currentUserId, db]); // Re-run effect if currentUserId or db changes

  const handleSignOut = async () => {
    try {
      await signOut(auth);

      navigate("/signin");
    } catch (error) {
      console.error("Error signing out:", error);
    }
  };

  const handleLinkClick = (path) => {
    navigate(path); // Close sidebar on mobile after clicking a link

    if (window.innerWidth < 768) {
      onToggleSidebar();
    }
  }; // --- MODIFICATION START (Reverting to proactive session creation) ---

  const handleNewQueryClick = async () => {
    if (!currentUserId) {
      console.error("No user authenticated to create a new query.");

      navigate("/signin"); // Redirect to login if no user

      return;
    }

    try {
      const newSessionRef = await addDoc(
        collection(db, "users", currentUserId, "querySessions"),

        {
          title: "New Chat", // Initial generic title for the sidebar

          createdAt: serverTimestamp(),

          lastUpdated: serverTimestamp(),
        }
      );

      navigate(`/dashboard/${newSessionRef.id}`); // Navigate to the new session's page
    } catch (error) {
      console.error("Error creating new query session:", error);

      alert("Failed to start a new chat. Please try again.");
    } // Close sidebar on mobile after clicking "New Query"

    if (window.innerWidth < 768) {
      onToggleSidebar();
    }
  }; // --- MODIFICATION END ---

  const handleRecentQueryClick = (queryId) => {
    navigate(`/dashboard/${queryId}`); // Navigate to the specific query // Close sidebar on mobile after clicking a recent query

    if (window.innerWidth < 768) {
      onToggleSidebar();
    }
  };

  return (
    <>
      {/* Mobile Sidebar (Fixed, slides in/out, hidden on md+) */}{" "}
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
        {" "}
        <div className="flex items-center justify-between mb-6">
          <h1 className="text-2xl font-bold">VirLaw</h1>{" "}
          <button
            onClick={onToggleSidebar}
            className="p-2 rounded-md hover:bg-gray-700 focus:outline-none focus:ring-2 focus:ring-blue-500"
            aria-label="Close Sidebar"
          >
            <XMarkIcon className="w-6 h-6" />{" "}
          </button>{" "}
        </div>{" "}
        <SidebarContent
          onNewQueryClick={handleNewQueryClick} // Pass the corrected handler
          recentQueries={recentQueries} // Pass the fetched queries
          activeQueryIdFromUrl={activeQueryIdFromUrl}
          onRecentQueryClick={handleRecentQueryClick} // Pass the internal handler
          user={user}
          handleLinkClick={handleLinkClick}
          handleSignOut={handleSignOut}
        />{" "}
      </div>{" "}
      {/* Desktop Sidebar (Relative, pushes content, hidden on small screens) */}{" "}
      <div
        className={`

     hidden md:flex flex-shrink-0 bg-gray-900 text-white p-4 flex-col z-10 h-screen

     transition-all duration-300 ease-in-out

     ${isSidebarOpen ? "w-64" : "w-0 overflow-hidden"}

    `}
      >
        {" "}
        {isSidebarOpen && (
          <>
            {" "}
            <div className="flex items-center justify-between mb-6">
              <h1 className="text-2xl font-bold">VirLaw</h1>{" "}
            </div>{" "}
            <SidebarContent
              onNewQueryClick={handleNewQueryClick} // Pass the corrected handler
              recentQueries={recentQueries} // Pass the fetched queries
              activeQueryIdFromUrl={activeQueryIdFromUrl}
              onRecentQueryClick={handleRecentQueryClick} // Pass the internal handler
              user={user}
              handleLinkClick={handleLinkClick}
              handleSignOut={handleSignOut}
            />{" "}
          </>
        )}{" "}
      </div>{" "}
    </>
  );
};

export default Sidebar;
