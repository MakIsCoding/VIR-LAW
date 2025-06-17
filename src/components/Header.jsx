// src/components/Header.jsx
import React from "react";
import { useNavigate } from "react-router-dom";
import { UserCircleIcon, Bars3Icon } from "@heroicons/react/24/solid";

const Header = ({ onToggleSidebar, user }) => {
  const navigate = useNavigate();

  return (
    <header className="flex-shrink-0 bg-gray-900 text-white p-4 flex items-center justify-between shadow-md z-20 w-full">
      <div className="flex items-center">
        {/* Sidebar Toggle Button */}
        <button
          onClick={onToggleSidebar}
          title="Toggle Sidebar"
          className="p-2 rounded-md hover:bg-gray-700 focus:outline-none focus:ring-2 focus:ring-blue-500 mr-4"
          aria-label="Toggle Sidebar"
        >
          <Bars3Icon className="w-6 h-6" />
        </button>

        {/* Main application title */}
        <span className="text-xl font-semibold">VirLaw Assistant</span>
      </div>

      <div className="flex items-center">
        {user ? (
          <button
            onClick={() => navigate("/dashboard/profile")}
            title="User Profile"
            className="p-2 rounded-md hover:bg-gray-700 focus:outline-none focus:ring-2 focus:ring-blue-500"
            aria-label="User Profile"
          >
            <UserCircleIcon className="w-6 h-6" />
          </button>
        ) : (
          <button
            onClick={() => navigate("/signin")}
            className="bg-blue-600 hover:bg-blue-700 text-white py-1 px-3 rounded text-sm"
          >
            Sign In
          </button>
        )}
      </div>
    </header>
  );
};

export default Header;
