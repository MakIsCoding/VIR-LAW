// src/components/ProfilePage.jsx

import React, { useState, useEffect } from "react";

import { auth, db } from "../firebase"; // Import auth and db

import { doc, getDoc, updateDoc } from "firebase/firestore"; // Import Firestore functions

import {
  updateProfile,
  EmailAuthProvider,
  reauthenticateWithCredential,
  updateEmail,
  updatePassword,
} from "firebase/auth"; // Import Firebase Auth functions

const ProfilePage = () => {
  const user = auth.currentUser; // Get the current authenticated user

  const [displayName, setDisplayName] = useState("");

  const [email, setEmail] = useState("");

  const [newPassword, setNewPassword] = useState(""); // For password update

  const [currentPassword, setCurrentPassword] = useState(""); // For reauthentication

  const [message, setMessage] = useState(""); // For success/error messages

  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (user) {
      setDisplayName(user.displayName || "");

      setEmail(user.email || ""); // Removed: Functionality to fetch custom user data

      setLoading(false); // Set loading to false once basic user data is available
    } else {
      setLoading(false);

      setMessage("Please sign in to view your profile.");
    }
  }, [user]); // Removed 'db' from dependency array as it's no longer used in this useEffect block

  const handleUpdateProfile = async (e) => {
    e.preventDefault();

    if (!user) {
      setMessage("No user signed in.");

      return;
    }

    setMessage(""); // Clear previous messages

    let successMessage = "Profile updated successfully!";

    let hasError = false; // 1. Update Display Name (if changed)

    if (displayName !== user.displayName) {
      try {
        await updateProfile(user, { displayName });

        successMessage = "Display name updated.";
      } catch (error) {
        console.error("Error updating display name:", error);

        setMessage(`Error updating display name: ${error.message}`);

        hasError = true;
      }
    } // 2. Reauthenticate for Email/Password changes

    if ((email !== user.email && email) || newPassword) {
      if (!currentPassword) {
        setMessage(
          "Please enter your current password to change email or password."
        );

        return;
      }

      try {
        const credential = EmailAuthProvider.credential(
          user.email,

          currentPassword
        );

        await reauthenticateWithCredential(user, credential);

        successMessage = "Profile updated and reauthenticated.";
      } catch (error) {
        console.error("Error reauthenticating:", error);

        setMessage(
          `Error reauthenticating: ${error.message}. Please check your current password.`
        );

        return; // Stop if reauthentication fails
      }
    } // 3. Update Email (if changed and reauthenticated)

    if (email !== user.email && email) {
      try {
        await updateEmail(user, email);

        successMessage =
          "Email updated successfully (verification email sent if required).";
      } catch (error) {
        console.error("Error updating email:", error);

        setMessage(`Error updating email: ${error.message}`);

        hasError = true;
      }
    } // 4. Update Password (if newPassword is provided and reauthenticated)

    if (newPassword) {
      try {
        await updatePassword(user, newPassword);

        successMessage = "Password updated successfully.";

        setNewPassword(""); // Clear password field

        setCurrentPassword(""); // Clear current password field
      } catch (error) {
        console.error("Error updating password:", error);

        setMessage(`Error updating password: ${error.message}`);

        hasError = true;
      }
    } // Custom data update logic previously here is now entirely removed.

    if (!hasError) {
      setMessage(successMessage);
    }
  };

  if (loading) {
    return <div className="p-6 text-gray-100">Loading profile...</div>;
  }

  if (!user) {
    return <div className="p-6 text-red-400">{message}</div>;
  }

  return (
    <div className="p-6 bg-gray-800 min-h-screen text-gray-100">
      {" "}
      <h1 className="text-3xl font-bold text-blue-300 mb-6">User Profile</h1>   {" "}
      {message && (
        <div
          className={`p-3 mb-4 rounded-md ${
            message.includes("Error") || message.includes("Failed")
              ? "bg-red-600"
              : "bg-green-600"
          } text-white`}
        >
          {message}{" "}
        </div>
      )}{" "}
      <form
        onSubmit={handleUpdateProfile}
        className="bg-gray-900 p-8 rounded-lg shadow-lg max-w-lg mx-auto"
      >
        {" "}
        <div className="mb-6">
          {" "}
          <label
            htmlFor="displayName"
            className="block text-sm font-medium text-gray-300 mb-2"
          >
            Display Name{" "}
          </label>{" "}
          <input
            type="text"
            id="displayName"
            value={displayName}
            onChange={(e) => setDisplayName(e.target.value)}
            className="w-full px-4 py-2 bg-gray-700 border border-gray-600 rounded-md text-gray-100 placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500"
            placeholder="Your Display Name"
          />{" "}
        </div>{" "}
        <div className="mb-6">
          {" "}
          <label
            htmlFor="email"
            className="block text-sm font-medium text-gray-300 mb-2"
          >
            Email (Cannot be changed here){" "}
          </label>{" "}
          <input
            type="email"
            id="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            className="w-full px-4 py-2 bg-gray-700 border border-gray-600 rounded-md text-gray-100 placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500"
            placeholder="Your Email"
            disabled // Email field is disabled for direct editing, requiring reauthentication
          />{" "}
        </div>{" "}
        <div className="mb-6">
          {" "}
          <label
            htmlFor="newPassword"
            className="block text-sm font-medium text-gray-300 mb-2"
          >
            New Password (Leave blank if not changing){" "}
          </label>{" "}
          <input
            type="password"
            id="newPassword"
            value={newPassword}
            onChange={(e) => setNewPassword(e.target.value)}
            className="w-full px-4 py-2 bg-gray-700 border border-gray-600 rounded-md text-gray-100 focus:outline-none focus:ring-2 focus:ring-blue-500"
            placeholder="Enter new password"
          />{" "}
        </div>{" "}
        {(email !== user.email || newPassword) && ( // Show current password field only if email or password are being changed
          <div className="mb-6">
            {" "}
            <label
              htmlFor="currentPassword"
              className="block text-sm font-medium text-gray-300 mb-2"
            >
              Current Password (Required for Email/Password changes){" "}
            </label>{" "}
            <input
              type="password"
              id="currentPassword"
              value={currentPassword}
              onChange={(e) => setCurrentPassword(e.target.value)}
              className="w-full px-4 py-2 bg-gray-700 border border-gray-600 rounded-md text-gray-100 focus:outline-none focus:ring-2 focus:ring-blue-500"
              placeholder="Enter your current password"
            />{" "}
          </div>
        )}{" "}
        <button
          type="submit"
          className="w-full bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 px-4 rounded-md transition duration-200 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 focus:ring-offset-gray-900"
        >
          Update Profile{" "}
        </button>{" "}
      </form>{" "}
    </div>
  );
};

export default ProfilePage;
