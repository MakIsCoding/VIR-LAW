// src/components/MainLayout.jsx
import React from "react";
import Header from "./Header";
import Sidebar from "./Sidebar";

const MainLayout = ({
  user,
  isSidebarOpen,
  setIsSidebarOpen,
  activeQueryIdFromUrl, // Needed for highlighting in Sidebar
  children,
}) => {
  const toggleSidebar = () => {
    setIsSidebarOpen(!isSidebarOpen);
  };

  return (
    <div className="flex h-screen overflow-hidden bg-gray-800">
      {/* Sidebar Component */}
      <Sidebar
        isSidebarOpen={isSidebarOpen}
        onToggleSidebar={toggleSidebar}
        user={user}
        activeQueryIdFromUrl={activeQueryIdFromUrl}
      />

      {/* Main content area */}
      <div className="flex flex-col flex-1 min-w-0 h-full transition-all duration-300 ease-in-out">
        <Header onToggleSidebar={toggleSidebar} user={user} />
        <main className="flex-1 overflow-y-auto p-6 custom-scrollbar">
          {children}
        </main>
      </div>

      {/* Mobile overlay when sidebar is open */}
      {isSidebarOpen && (
        <div
          className="fixed inset-0 bg-black bg-opacity-50 z-30 md:hidden"
          onClick={toggleSidebar}
        />
      )}
    </div>
  );
};

export default MainLayout;
