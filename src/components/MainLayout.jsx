// FIXED MainLayout.jsx - Complete implementation
import React from "react";
import Header from "./Header";
import UltimateSidebar from "./UltimateSidebar";

const MainLayout = ({
  user,
  isSidebarOpen,
  setIsSidebarOpen,
  onNewQueryClick,
  recentQueries,
  onRecentQueryClick,
  loadingRecentQueries,
  activeQueryIdFromUrl,
  children,
}) => {
  const toggleSidebar = () => {
    setIsSidebarOpen(!isSidebarOpen);
  };

  return (
    <div className="flex h-screen bg-gray-50">
      {/* Enhanced Sidebar */}
      <UltimateSidebar
        user={user}
        isSidebarOpen={isSidebarOpen}
        setIsSidebarOpen={setIsSidebarOpen}
        onNewQueryClick={onNewQueryClick}
        recentQueries={recentQueries}
        onRecentQueryClick={onRecentQueryClick}
        loadingRecentQueries={loadingRecentQueries}
        activeQueryIdFromUrl={activeQueryIdFromUrl}
      />

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Header */}
        <Header onToggleSidebar={toggleSidebar} user={user} />
        
        {/* Main Content */}
        <main className="flex-1 overflow-x-hidden overflow-y-auto">
          {children}
        </main>
      </div>
    </div>
  );
};

export default MainLayout;