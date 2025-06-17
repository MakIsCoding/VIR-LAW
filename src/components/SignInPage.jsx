import React, { useState } from "react";
import {
  signInWithEmailAndPassword,
  createUserWithEmailAndPassword,
  signInWithPopup,
} from "firebase/auth";
import { auth, googleProvider, db } from "../firebase";
import { doc, setDoc, getDoc } from "firebase/firestore";  // <-- FULL import here
import { useNavigate } from "react-router-dom";

const SignInPage = () => {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [isRegistering, setIsRegistering] = useState(false);
  const [error, setError] = useState("");
  const navigate = useNavigate();

  const createUserProfileInFirestore = async (user) => {
    const userDocRef = doc(db, "users", user.uid);
    try {
      const docSnap = await getDoc(userDocRef);
      if (!docSnap.exists()) {
        await setDoc(userDocRef, {
          displayName: user.displayName || "",
          email: user.email || "",
          createdAt: new Date(),
          customField: "",
        }, { merge: true });
        console.log("User profile created in Firestore.");
      } else {
        console.log("User profile already exists.");
      }
    } catch (firestoreError) {
      console.error("Error creating/updating user profile:", firestoreError);
    }
  };

  const handleAuth = async (e) => {
    e.preventDefault();
    setError("");
    try {
      let userCredential;
      if (isRegistering) {
        userCredential = await createUserWithEmailAndPassword(auth, email, password);
        alert("Registration successful!");
      } else {
        userCredential = await signInWithEmailAndPassword(auth, email, password);
        alert("Login successful!");
      }
      await createUserProfileInFirestore(userCredential.user);
      navigate("/dashboard");  // <-- redirect after login
    } catch (err) {
      console.error("Authentication error:", err);
      setError(err.message);
    }
  };

  const handleGoogleSignIn = async () => {
    setError("");
    try {
      const result = await signInWithPopup(auth, googleProvider);
      alert("Signed in with Google!");
      await createUserProfileInFirestore(result.user);
      navigate("/dashboard");
    } catch (err) {
      console.error("Google Sign-In error:", err);
      setError(err.message);
    }
  };

  return (
    <div className="flex items-center justify-center h-full w-full">
      <div className="bg-gray-700 p-8 rounded-lg shadow-xl w-full max-w-md text-gray-100">
        <h2 className="text-3xl font-bold text-center mb-6 text-blue-300">
          {isRegistering ? "Register" : "Login"}
        </h2>
        {error && <p className="text-red-400 text-center mb-4">{error}</p>}
        <form onSubmit={handleAuth}>
          <div className="mb-4">
            <label className="block text-gray-300 text-sm font-bold mb-2" htmlFor="email">
              Email
            </label>
            <input
              type="email"
              id="email"
              className="shadow appearance-none border rounded w-full py-2 px-3 text-gray-900 bg-gray-200"
              placeholder="Enter your email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
            />
          </div>
          <div className="mb-6">
            <label className="block text-gray-300 text-sm font-bold mb-2" htmlFor="password">
              Password
            </label>
            <input
              type="password"
              id="password"
              className="shadow appearance-none border rounded w-full py-2 px-3 text-gray-900 bg-gray-200"
              placeholder="********"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
            />
          </div>
          <div className="flex items-center justify-between">
            <button type="submit" className="bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded transition duration-200">
              {isRegistering ? "Register" : "Login"}
            </button>
            <button type="button" onClick={() => setIsRegistering(!isRegistering)} className="text-blue-400 hover:text-blue-200 font-bold text-sm">
              {isRegistering ? "Already have an account? Login" : "Need an account? Register"}
            </button>
          </div>
        </form>

        <div className="mt-6 border-t border-gray-600 pt-6">
          <p className="text-center text-gray-300 mb-4">Or sign in with:</p>
          <button onClick={handleGoogleSignIn} className="w-full bg-red-600 hover:bg-red-700 text-white font-bold py-2 px-4 rounded flex items-center justify-center">
            <svg className="w-5 h-5 mr-2" viewBox="0 0 24 24" fill="currentColor">
              <path d="M12 12c-2.761 0-5 2.239-5 5s2.239 5 5 5 5-2.239 5-5-2.239-5-5-5zm0-10c-3.866 0-7 3.134-7 7s3.134 7 7 7 7-3.134 7-7-3.134-7-7-7zm0 2c2.761 0 5 2.239 5 5s-2.239 5-5 5-5-2.239-5-5 2.239-5 5-5z" fill="white" />
            </svg>
            Sign in with Google
          </button>
        </div>
      </div>
    </div>
  );
};

export default SignInPage;
