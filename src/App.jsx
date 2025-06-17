// src/App.jsx
import React, { useState, useEffect } from "react";
import {
  Routes,
  Route,
  Navigate,
  useNavigate,
  useLocation,
} from "react-router-dom";

import { onAuthStateChanged } from "firebase/auth";
import { auth } from "./firebase";

import MainLayout from "./components/MainLayout";
import QueryPage from "./components/QueryPage";
import SignInPage from "./components/SignInPage";
import SignUpPage from "./components/SignUpPage"; // Uncomment if used
import WelcomePage from "./components/WelcomePage";
import SettingsHelpPage from "./components/SettingsHelpPage";
import ProfilePage from "./components/ProfilePage";

function AppContent() {
  const [user, setUser] = useState(null);
  const [loadingAuth, setLoadingAuth] = useState(true);
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);

  const navigate = useNavigate();
  const location = useLocation();

  // 1. Auth State Listener
  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, (currentUser) => {
      setUser(currentUser);
      setLoadingAuth(false);

      if (!currentUser) {
        if (!["/signin", "/signup"].includes(location.pathname)) {
          navigate("/signin");
        }
      } else {
        if (
          location.pathname === "/" ||
          location.pathname === "/signin" ||
          location.pathname === "/signup"
        ) {
          navigate("/dashboard");
        }
      }
    });

    return () => unsubscribe();
  }, [navigate, location.pathname]);

  // Navigation helpers
  const handleNewQueryClick = () => {
    if (user) {
      navigate("/dashboard/new");
      setIsSidebarOpen(false);
    } else {
      navigate("/signin");
    }
  };

  const handleRecentQueryClick = (queryId) => {
    if (user) {
      navigate(`/dashboard/${queryId}`);
      setIsSidebarOpen(false);
    } else {
      navigate("/signin");
    }
  };

  // Sidebar active ID
  const pathSegments = location.pathname.split("/");
  const activeQueryIdFromUrl = pathSegments[pathSegments.length - 1];

  if (loadingAuth) {
    return (
      <div className="flex justify-center items-center h-screen bg-gray-900 text-white text-lg">
        Loading application...
      </div>
    );
  }

  return (
    <MainLayout
      user={user}
      isSidebarOpen={isSidebarOpen}
      setIsSidebarOpen={setIsSidebarOpen}
      onNewQueryClick={handleNewQueryClick}
      onRecentQueryClick={handleRecentQueryClick}
      activeQueryIdFromUrl={activeQueryIdFromUrl}
    >
      <Routes>
        <Route path="/signin" element={<SignInPage />} />
        {/* <Route path="/signup" element={<SignUpPage />} /> */}
        <Route
          path="/dashboard/new"
          element={user ? <QueryPage /> : <Navigate to="/signin" />}
        />
        <Route
          path="/dashboard/:queryId"
          element={user ? <QueryPage /> : <Navigate to="/signin" />}
        />
        <Route
          path="/dashboard/profile"
          element={user ? <ProfilePage /> : <Navigate to="/signin" />}
        />
        <Route
          path="/dashboard/welcome"
          element={user ? <WelcomePage /> : <Navigate to="/signin" />}
        />
        <Route
          path="/dashboard/settings-help"
          element={user ? <SettingsHelpPage /> : <Navigate to="/signin" />}
        />
        <Route
          path="/dashboard"
          element={
            user ? <Navigate to="/dashboard/welcome" /> : <Navigate to="/signin" />
          }
        />
        <Route
          path="*"
          element={<Navigate to={user ? "/dashboard" : "/signin"} />}
        />
      </Routes>
    </MainLayout>
  );
}

function App() {
  return <AppContent />;
}

export default App;
