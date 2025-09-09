// FIXED Header.jsx - Complete implementation preserving your design intent
import React from "react";
import { useNavigate } from "react-router-dom";
import { auth } from "../firebase";
import { UserCircleIcon, Bars3Icon } from "@heroicons/react/24/solid";

const Header = ({ onToggleSidebar, user }) => {
  const navigate = useNavigate();

  return (
    <header className="bg-white border-b border-gray-200 px-4 py-3 flex items-center justify-between">
      {/* Left side - Mobile menu button */}
      <div className="flex items-center space-x-4">
        <button
          onClick={onToggleSidebar}
          className="lg:hidden p-2 rounded-md text-gray-600 hover:text-gray-900 hover:bg-gray-100"
        >
          <Bars3Icon className="h-6 w-6" />
        </button>
        
        <div className="hidden lg:block">
          <h1 className="text-xl font-semibold text-gray-900">VirLaw AI</h1>
          <p className="text-xs text-gray-500">Ultimate Legal Assistant</p>
        </div>
      </div>

      {/* Right side - User info */}
      <div className="flex items-center space-x-4">
        {user ? (
          <div className="flex items-center space-x-3">
            <div className="flex items-center space-x-2">
              <UserCircleIcon className="h-8 w-8 text-gray-400" />
              <div className="hidden md:block">
                <p className="text-sm font-medium text-gray-700">
                  {user.displayName || user.email}
                </p>
                <p className="text-xs text-gray-500">Online</p>
              </div>
            </div>
          </div>
        ) : (
          <button
            onClick={() => navigate("/signin")}
            className="px-4 py-2 text-sm font-medium text-blue-600 bg-blue-50 rounded-md hover:bg-blue-100"
          >
            Sign In
          </button>
        )}
      </div>
    </header>
  );
};

export default Header;