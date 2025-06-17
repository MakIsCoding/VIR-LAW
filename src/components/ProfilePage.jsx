// src/components/ProfilePage.jsx
import React, { useState, useEffect } from "react";
import { auth, db } from "../firebase";
import {
  doc,
  getDoc,
  updateDoc
} from "firebase/firestore";
import {
  onAuthStateChanged,
  updateProfile,
  EmailAuthProvider,
  reauthenticateWithCredential,
  updateEmail,
  updatePassword
} from "firebase/auth";

const ProfilePage = () => {
  const [user, setUser] = useState(null); // use state instead of auth.currentUser directly
  const [displayName, setDisplayName] = useState("");
  const [email, setEmail] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [currentPassword, setCurrentPassword] = useState("");
  const [message, setMessage] = useState("");
  const [loading, setLoading] = useState(true);

  // Listen for auth state changes
  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, async (firebaseUser) => {
      if (firebaseUser) {
        setUser(firebaseUser);
        setDisplayName(firebaseUser.displayName || "");
        setEmail(firebaseUser.email || "");
      } else {
        setUser(null);
        setMessage("Please sign in to view your profile.");
      }
      setLoading(false);
    });

    return () => unsubscribe();
  }, []);

  const handleUpdateProfile = async (e) => {
    e.preventDefault();
    if (!user) {
      setMessage("No user signed in.");
      return;
    }

    setMessage(""); // clear old messages
    let successMessage = "Profile updated successfully!";
    let hasError = false;

    // 1. Update display name
    if (displayName !== user.displayName) {
      try {
        await updateProfile(user, { displayName });
        successMessage = "Display name updated.";
      } catch (error) {
        setMessage(`Error updating display name: ${error.message}`);
        hasError = true;
      }
    }

    // 2. Re-authenticate before sensitive changes
    if (newPassword || (email !== user.email && email)) {
      if (!currentPassword) {
        setMessage("Enter current password to change email or password.");
        return;
      }

      try {
        const credential = EmailAuthProvider.credential(
          user.email,
          currentPassword
        );
        await reauthenticateWithCredential(user, credential);
      } catch (error) {
        setMessage(`Error reauthenticating: ${error.message}`);
        return;
      }
    }

    // 3. Update email
    if (email !== user.email && email) {
      try {
        await updateEmail(user, email);
        successMessage = "Email updated.";
      } catch (error) {
        setMessage(`Error updating email: ${error.message}`);
        hasError = true;
      }
    }

    // 4. Update password
    if (newPassword) {
      try {
        await updatePassword(user, newPassword);
        successMessage = "Password updated.";
        setNewPassword("");
        setCurrentPassword("");
      } catch (error) {
        setMessage(`Error updating password: ${error.message}`);
        hasError = true;
      }
    }

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
      <h1 className="text-3xl font-bold text-blue-300 mb-6">User Profile</h1>

      {message && (
        <div
          className={`p-3 mb-4 rounded-md ${
            message.toLowerCase().includes("error")
              ? "bg-red-600"
              : "bg-green-600"
          } text-white`}
        >
          {message}
        </div>
      )}

      <form
        onSubmit={handleUpdateProfile}
        className="bg-gray-900 p-8 rounded-lg shadow-lg max-w-lg mx-auto"
      >
        <div className="mb-6">
          <label
            htmlFor="displayName"
            className="block text-sm font-medium text-gray-300 mb-2"
          >
            Display Name
          </label>
          <input
            type="text"
            id="displayName"
            value={displayName}
            onChange={(e) => setDisplayName(e.target.value)}
            className="w-full px-4 py-2 bg-gray-700 border border-gray-600 rounded-md text-gray-100"
            placeholder="Your display name"
          />
        </div>

        <div className="mb-6">
          <label
            htmlFor="email"
            className="block text-sm font-medium text-gray-300 mb-2"
          >
            Email
          </label>
          <input
            type="email"
            id="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            className="w-full px-4 py-2 bg-gray-700 border border-gray-600 rounded-md text-gray-100"
            placeholder="Your email"
          />
        </div>

        <div className="mb-6">
          <label
            htmlFor="newPassword"
            className="block text-sm font-medium text-gray-300 mb-2"
          >
            New Password (leave blank if not changing)
          </label>
          <input
            type="password"
            id="newPassword"
            value={newPassword}
            onChange={(e) => setNewPassword(e.target.value)}
            className="w-full px-4 py-2 bg-gray-700 border border-gray-600 rounded-md text-gray-100"
            placeholder="New password"
          />
        </div>

        {(newPassword || (email !== user.email && email)) && (
          <div className="mb-6">
            <label
              htmlFor="currentPassword"
              className="block text-sm font-medium text-gray-300 mb-2"
            >
              Current Password (required for email/password changes)
            </label>
            <input
              type="password"
              id="currentPassword"
              value={currentPassword}
              onChange={(e) => setCurrentPassword(e.target.value)}
              className="w-full px-4 py-2 bg-gray-700 border border-gray-600 rounded-md text-gray-100"
              placeholder="Enter your current password"
              required
            />
          </div>
        )}

        <button
          type="submit"
          className="w-full bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 px-4 rounded-md transition duration-200"
        >
          Update Profile
        </button>
      </form>
    </div>
  );
};

export default ProfilePage;
