// src/components/Sidebar.jsx

import React, { useState, useEffect } from "react";
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
} from "firebase/firestore";

import {
  ChatBubbleLeftRightIcon,
  HomeIcon,
  Cog6ToothIcon,
  XMarkIcon,
} from "@heroicons/react/24/solid";

const SidebarContent = ({
  onNewQueryClick,
  recentQueries,
  activeQueryIdFromUrl,
  onRecentQueryClick,
  user,
  handleLinkClick,
  handleSignOut,
}) => {
  const isUserAuthenticated = !!user;
  const NEW_QUERY_ROUTE_ID = "new";

  return (
    <>
      <aside className="bg-gray-800 text-white p-4 w-64 h-screen overflow-y-auto"> {/*Tailwind Styling*/}
        {isUserAuthenticated ? (
          <>
            <ul className="space-y-2"> {/*Tailwind Styling*/}
              <li>
                <button
                  onClick={onNewQueryClick}
                  className={`flex items-center w-full p-3 rounded-md transition-colors duration-200 text-blue-300
                    ${
                      activeQueryIdFromUrl === NEW_QUERY_ROUTE_ID
                        ? "bg-blue-700"
                        : "hover:bg-blue-600"
                    }`}
                >
                  <ChatBubbleLeftRightIcon className="w-5 h-5 mr-3" />
                  New Query
                </button>
              </li>

              <li>
                <button
                  onClick={() => handleLinkClick("/dashboard")}
                  className={`flex items-center w-full p-3 rounded-md hover:bg-gray-700 transition-colors duration-200 ${
                    [ "", "dashboard", "welcome"].includes(activeQueryIdFromUrl)
                      ? "bg-gray-700"
                      : ""
                  }`}
                >
                  <HomeIcon className="w-5 h-5 mr-3" />
                  Dashboard
                </button>
              </li>

              <li>
                <button
                  onClick={() => handleLinkClick("/dashboard/welcome")}
                  className={`flex items-center w-full p-3 rounded-md hover:bg-gray-700 transition-colors duration-200 ${
                    activeQueryIdFromUrl === "welcome" ? "bg-gray-700" : ""
                  }`}
                >
                  <HomeIcon className="w-5 h-5 mr-3" />
                  Welcome
                </button>
              </li>
            </ul>

            <div className="border-t border-gray-600 pt-4 mt-4">
              <h3 className="text-sm font-semibold text-gray-400 mb-3">
                Recent Cases
              </h3>
              {recentQueries.length > 0 ? (
                <ul className="space-y-2"> {/*Tailwind Styling*/}
                  {recentQueries.map((queryItem) => (
                    <li key={queryItem.id}>
                      <button
                        onClick={() => onRecentQueryClick(queryItem.id)}
                        className={`text-sm w-full text-left p-2 rounded-md truncate ${
                          activeQueryIdFromUrl === queryItem.id
                            ? "bg-gray-700"
                            : "hover:bg-gray-700"
                        }`}
                      >
                        {queryItem.title || "Untitled Query"}
                      </button>
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="text-gray-500 text-sm">No recent queries.</p>
              )}
            </div>
          </>
        ) : (
          <div className="flex flex-col justify-between p-4 text-gray-400 text-center text-sm">
            <p>Sign in to access features like new queries and recent cases.</p>
            <div className="mt-auto pt-4 border-t border-gray-700">
              <button
                onClick={() => handleLinkClick("/signin")}
                className="w-full bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded transition duration-200 flex items-center justify-center mb-2"
              >
                Sign In
              </button>

              <button
                onClick={() => handleLinkClick("/dashboard/settings-help")}
                className="flex items-center w-full p-2 rounded hover:bg-gray-700 text-sm mb-2"
              >
                <Cog6ToothIcon className="w-5 h-5 mr-3" />
                Settings & Help
              </button>
            </div>
          </div>
        )}
      </aside> {/*Tailwind Styling*/}
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
  const [fetchError, setFetchError] = useState(null); // New state for fetch errors
  const currentUserId = user?.uid;

  useEffect(() => {
    if (!currentUserId) {
      setRecentQueries([]);
      return;
    }

    const userQuerySessionsRef = collection(
      db,
      "users",
      currentUserId,
      "querySessions"
    );

    const q = query(userQuerySessionsRef, orderBy("lastUpdated", "desc"));

    const unsubscribe = onSnapshot(
      q,
      (snapshot) => {
        const queriesData = snapshot.docs.map((doc) => ({
          id: doc.id,
          ...doc.data(),
        }));

        setRecentQueries(queriesData);
        setFetchError(null); // Clear error if fetch was successful
      },
      (error) => {
        console.error("Error fetching recent queries:", error);
        setFetchError("Failed to load recent queries."); // Set error state
      }
    );

    return () => unsubscribe();
  }, [currentUserId]); // no need to include `db`

  const handleSignOut = async () => {
    try {
      await signOut(auth);
      navigate("/signin");
    } catch (error) {
      console.error("Error signing out:", error);
      // Optionally, display an error message to the user
    }
  };

  const handleLinkClick = (path) => {
    navigate(path);
    if (window.innerWidth < 768) {
      onToggleSidebar();
    }
  };

  const handleNewQueryClick = async () => {
    if (!currentUserId) {
      console.error("No user authenticated.");
      navigate("/signin");
      return;
    }

    try {
      const newSessionRef = await addDoc(
        collection(db, "users", currentUserId, "querySessions"),
        {
          title: "New Chat",
          createdAt: serverTimestamp(),
          lastUpdated: serverTimestamp(),
        }
      );
      navigate(`/dashboard/${newSessionRef.id}`);
    } catch (error) {
      console.error("Error creating new query session:", error);
      alert("Failed to create new query. Try again.");
    }

    if (window.innerWidth < 768) {
      onToggleSidebar();
    }
  };

  const handleRecentQueryClick = (queryId) => {
    navigate(`/dashboard/${queryId}`);
    if (window.innerWidth < 768) {
      onToggleSidebar();
    }
  };

  return (
    <>
      {/* Mobile sidebar */}
      <div
        className={`fixed inset-y-0 left-0 w-64 bg-gray-900 text-white p-4 z-40 h-screen transition-transform duration-300 ease-in-out ${
          isSidebarOpen ? "translate-x-0" : "-translate-x-full"
        } md:hidden`}
      >
        <div className="flex items-center justify-between mb-6">
          <h1 className="text-2xl font-bold">VirLaw</h1>
          <button
            onClick={onToggleSidebar}
            className="p-2 rounded-md hover:bg-gray-700"
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
        />
      </div>

      {/* Desktop sidebar */}
      <div
        className={`hidden md:flex flex-shrink-0 bg-gray-900 text-white p-4 flex-col z-10 h-screen transition-all duration-300 ease-in-out ${
          isSidebarOpen ? "w-64" : "w-0 overflow-hidden"
        }`}
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
            />
          </>
        )}
      </div>
    </>
  );
};

export default Sidebar;
