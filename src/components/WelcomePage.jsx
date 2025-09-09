// src/components/WelcomePage.jsx
import React from "react";
import { useNavigate } from "react-router-dom";

const WelcomePage = () => {
  const navigate = useNavigate();

  const handleNewQueryClick = () => {
    navigate("/dashboard/new");
  };

  const handleSettingsHelpClick = () => {
    navigate("/dashboard/settings-help");
  };

  // Function to handle the "Get Started" button click
  const handleGetStartedClick = () => {
    navigate("/dashboard/new"); // Directs to the New Query page
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-white text-gray-100 p-6">
      <div className="max-w-3xl text-center">
        {" "}
        {/* This div ensures its content is centered */}
        <h1 className="text-4xl font-bold text-black mb-4">
          Welcome to VirLaw!
        </h1>
        <p className="text-lg text-gray-300 mb-8 leading-relaxed">
          Your intelligent legal assistant is here to help you navigate the
          complexities of law. Start a new query or review your past cases from
          the sidebar.
        </p>
        {/* Moved "Get Started" button here to be directly under the main text,
            inheriting the text-center from its parent div. */}
        <button
          onClick={handleGetStartedClick}
          className="bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 px-8 rounded-full text-xl transition duration-300 ease-in-out mb-10 shadow-lg transform hover:scale-105"
        >
          Get Started
        </button>
        <div className="space-y-4 mb-10 text-left w-full sm:w-3/4 mx-auto">
          <p className="text-gray-200">
            <span className="font-semibold text-blue-300">New Query:</span>{" "}
            Click "New Query" in the sidebar to start a fresh discussion.
          </p>
          <p className="text-gray-200">
            <span className="font-semibold text-blue-300">Recent Cases:</span>{" "}
            Your previous interactions will appear under "Recent Cases" for
            quick access.
          </p>
          <p className="text-gray-200">
            <span className="font-semibold text-blue-300">
              Settings & Help:
            </span>{" "}
            Find options and support here.
            <button
              onClick={handleSettingsHelpClick}
              className="ml-2 text-blue-400 hover:underline focus:outline-none"
            >
              (Go to Settings & Help)
            </button>
          </p>
        </div>
        <p className="text-gray-400 text-sm mt-8">
          VirLaw is currently in beta. Your feedback helps us improve!
        </p>
      </div>
    </div>
  );
};

export default WelcomePage;
