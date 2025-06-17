// src/App.jsx

import React, { useState, useEffect } from "react";

import {
  Routes,
  Route,
  Navigate,
  useNavigate, // <--- Import useNavigate
  useLocation, // <--- IMPORTANT: Import useLocation for sidebar highlighting
} from "react-router-dom";

import { onAuthStateChanged } from "firebase/auth";

import { auth, db } from "./firebase"; // Ensure 'db' is imported

import { collection, query, orderBy, onSnapshot } from "firebase/firestore"; // For recent queries

import MainLayout from "./components/MainLayout";

import QueryPage from "./components/QueryPage"; // This will now fetch its own data

import SignInPage from "./components/SignInPage";

import SignUpPage from "./components/SignUpPage"; // Make sure to uncomment if you use it

import WelcomePage from "./components/WelcomePage";

import SettingsHelpPage from "./components/SettingsHelpPage";

import ProfilePage from "./components/ProfilePage";

// Main AppContent component that uses hooks like useNavigate, useLocation

function AppContent() {
  const [user, setUser] = useState(null);

  const [loadingAuth, setLoadingAuth] = useState(true); // Separate loading state for auth

  const [isSidebarOpen, setIsSidebarOpen] = useState(false);

  const [recentQueries, setRecentQueries] = useState([]);

  const [loadingRecentQueries, setLoadingRecentQueries] = useState(true);

  const navigate = useNavigate(); // Initialize useNavigate

  const location = useLocation(); // <--- Initialize useLocation // 1. Authentication State Listener

  useEffect(() => {
    const unsubscribeAuth = onAuthStateChanged(auth, (currentUser) => {
      setUser(currentUser);

      setLoadingAuth(false); // Redirect if user logs out or if they are on signin/root after logging in

      if (!currentUser) {
        if (
          location.pathname !== "/signin" &&
          location.pathname !== "/signup"
        ) {
          navigate("/signin");
        }
      } else {
        if (
          location.pathname === "/signin" ||
          location.pathname === "/signup" ||
          location.pathname === "/"
        ) {
          navigate("/dashboard"); // Redirect to dashboard on successful login/signup
        }
      }
    });

    return () => unsubscribeAuth(); // Clean up auth listener
  }, [navigate, location.pathname]); // location.pathname dependency for conditional redirect // 2. Fetch Recent Queries (for Sidebar) for the authenticated user

  useEffect(() => {
    if (user) {
      setLoadingRecentQueries(true);

      const q = query(
        collection(db, "users", user.uid, "querySessions"),

        orderBy("createdAt", "desc") // Order by creation time, newest first
      );

      const unsubscribeRecentQueries = onSnapshot(
        q,

        (snapshot) => {
          const queries = snapshot.docs.map((doc) => ({
            id: doc.id,

            ...doc.data(),
          }));

          setRecentQueries(queries);

          setLoadingRecentQueries(false);
        },

        (error) => {
          console.error("Error fetching recent queries for sidebar:", error);

          setLoadingRecentQueries(false);
        }
      );

      return () => unsubscribeRecentQueries(); // Clean up listener
    } else {
      setRecentQueries([]); // Clear recent queries if no user

      setLoadingRecentQueries(false);
    }
  }, [user, db]); // Re-run when user or db instance changes // Handler for "New Query" button click

  const handleNewQueryClick = () => {
    if (user) {
      // Only allow new query if user is logged in

      navigate("/dashboard/new"); // Navigate to the dedicated "new query" route

      setIsSidebarOpen(false); // Close sidebar on mobile
    } else {
      console.warn("User not authenticated for new query."); // Optionally, show a message to the user that they need to sign in

      navigate("/signin");
    }
  }; // Handler for clicking a recent query in the sidebar

  const handleRecentQueryClick = (queryId) => {
    if (user) {
      navigate(`/dashboard/${queryId}`); // Navigate to the specific query's URL

      setIsSidebarOpen(false); // Close sidebar on mobile
    } else {
      console.warn("User not authenticated to view recent query.");

      navigate("/signin");
    }
  }; // Determine activeQueryId for sidebar highlighting

  const pathSegments = location.pathname.split("/");

  const activeQueryIdFromUrl = pathSegments[pathSegments.length - 1]; // Show loading screen while checking authentication

  if (loadingAuth) {
    return (
      <div className="flex justify-center items-center h-screen bg-gray-900 text-white text-lg">
        Loading application...{" "}
      </div>
    );
  }

  return (
    <MainLayout
      user={user}
      isSidebarOpen={isSidebarOpen}
      setIsSidebarOpen={setIsSidebarOpen}
      onNewQueryClick={handleNewQueryClick}
      recentQueries={recentQueries} // Pass recent queries to sidebar
      onRecentQueryClick={handleRecentQueryClick}
      loadingRecentQueries={loadingRecentQueries} // Pass loading state for sidebar
      activeQueryIdFromUrl={activeQueryIdFromUrl} // <--- Pass this to MainLayout, then Sidebar
    >
      {" "}
      <Routes>
        {/* Sign-in and Sign-up routes (outside dashboard layout) */}
        <Route path="/signin" element={<SignInPage />} />{" "}
        {/* <Route path="/signup" element={<SignUpPage />} /> uncomment if you use it */}{" "}
        {/* Routes within the Dashboard layout for authenticated users */}
        {/* The order matters: specific paths first */}{" "}
        <Route
          path="/dashboard/new"
          element={user ? <QueryPage /> : <Navigate to="/signin" />}
        />{" "}
        <Route
          path="/dashboard/:queryId"
          element={user ? <QueryPage /> : <Navigate to="/signin" />}
        />{" "}
        <Route
          path="/dashboard/profile"
          element={user ? <ProfilePage /> : <Navigate to="/signin" />}
        />{" "}
        <Route
          path="/dashboard/welcome"
          element={user ? <WelcomePage /> : <Navigate to="/signin" />}
        />{" "}
        <Route
          path="/dashboard/settings-help"
          element={user ? <SettingsHelpPage /> : <Navigate to="/signin" />}
        />{" "}
        {/* Default /dashboard path for authenticated users: navigate to welcome */}{" "}
        <Route
          path="/dashboard"
          element={
            user ? (
              <Navigate to="/dashboard/welcome" />
            ) : (
              <Navigate to="/signin" />
            )
          }
        />
        {/* Catch-all for any other unauthenticated routes */}{" "}
        <Route
          path="*"
          element={<Navigate to={user ? "/dashboard" : "/signin"} />}
        />{" "}
      </Routes>{" "}
    </MainLayout>
  );
}

// Top-level App component: only renders AppContent. BrowserRouter is in main.jsx

function App() {
  return <AppContent />;
}

export default App;
