// src/components/MainLayout.jsx
import React from "react";
import Header from "./Header";
import Sidebar from "./Sidebar";

const MainLayout = ({
  user,
  isSidebarOpen,
  setIsSidebarOpen,
  onNewQueryClick,
  recentQueries,
  onRecentQueryClick,
  loadingRecentQueries,
  activeQueryIdFromUrl, // <--- NEW PROP: Receive from App.jsx
  children,
}) => {
  const toggleSidebar = () => {
    setIsSidebarOpen(!isSidebarOpen);
  };

  return (
    <div className="flex h-screen overflow-hidden bg-gray-800">
      {/* Sidebar Component (handles its own responsive positioning internally) */}
      <Sidebar
        isSidebarOpen={isSidebarOpen}
        onToggleSidebar={toggleSidebar}
        onNewQueryClick={onNewQueryClick}
        recentQueries={recentQueries}
        onRecentQueryClick={onRecentQueryClick}
        user={user}
        activeQueryIdFromUrl={activeQueryIdFromUrl} // <--- Pass it down to Sidebar
      />

      {/* Main Content Area Container (Header + Main Content) */}
      <div
        className={`flex flex-col flex-1 min-w-0 h-full
          transition-all duration-300 ease-in-out`}
      >
        {/* Header - Stays at the top of the main content area */}
        <Header onToggleSidebar={toggleSidebar} user={user} />

        {/* Main Content Area - Takes up remaining vertical space */}
        <main className="flex-1 overflow-y-auto p-6 custom-scrollbar">
          {children}
        </main>
      </div>

      {/* Overlay for small screens when sidebar is open */}
      {isSidebarOpen && (
        <div
          className="fixed inset-0 bg-black bg-opacity-50 z-30 md:hidden"
          onClick={toggleSidebar} // Click outside to close sidebar on mobile
        ></div>
      )}
    </div>
  );
};

export default MainLayout;
