import React, { useState, useRef } from "react";
import { Link, useNavigate } from "react-router-dom";
import { auth, db } from "/src/firebase.js"; // Assuming firebase.js is in the src directory
import { createUserWithEmailAndPassword } from "firebase/auth";
import { doc, setDoc } from "firebase/firestore";
import ReCAPTCHA from "react-google-recaptcha"; // Make sure you have installed react-google-recaptcha

const SignUpPage = () => {
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [error, setError] = useState("");
  const [recaptchaToken, setRecaptchaToken] = useState(null);
  const recaptchaRef = useRef(null);
  const navigate = useNavigate();

  const handleSignUp = async (e) => {
    e.preventDefault();
    setError(""); // Clear previous errors

    if (password !== confirmPassword) {
      setError("Passwords do not match.");
      return;
    }

    // Basic email validation
    if (!/\S+@\S+\.\S+/.test(email)) {
      setError("Please enter a valid email address.");
      return;
    }

    // Basic password validation (at least 6 characters)
    if (password.length < 6) {
      setError("Password must be at least 6 characters long.");
      return;
    }

    if (!recaptchaToken) {
      setError("Please complete the reCAPTCHA verification.");
      return;
    }

    try {
      const userCredential = await createUserWithEmailAndPassword(auth, email, password);
      const user = userCredential.user;

      // Store user's name in Firestore
      await setDoc(doc(db, "users", user.uid), {
        name: name,
        email: email, // Optionally store email
      });

      // Clear form fields
      setName("");
      setEmail("");
      setPassword("");
      setConfirmPassword("");
      setRecaptchaToken(null);
      recaptchaRef.current.reset();

      // Navigate to another page after successful signup
      navigate("/profile"); // Example: Navigate to a profile page

    } catch (err) {
      setError(err.message);
    }
  };

  const handleRecaptchaChange = (token) => {
    setRecaptchaToken(token);
  };

  return (
    <div className="flex items-center justify-center min-h-screen bg-gray-100">
      <div className="px-8 py-6 mt-4 text-left bg-white shadow-lg">
        <h3 className="text-2xl font-bold text-center">Sign Up</h3>
        <form onSubmit={handleSignUp}>
          <div className="mt-4">
            <div>
              <label className="block" htmlFor="name">
                Name
              </label>
              <input
                type="text"
                placeholder="Name"
                className="w-full px-4 py-2 mt-2 border rounded-md focus:outline-none focus:ring-1 focus:ring-blue-600"
                id="name"
                value={name}
                onChange={(e) => setName(e.target.value)}
                required
              />
            </div>
            <div className="mt-4">
              <label className="block" htmlFor="email">
                Email
              </label>
              <input
                type="email"
                placeholder="Email"
                className="w-full px-4 py-2 mt-2 border rounded-md focus:outline-none focus:ring-1 focus:ring-blue-600"
                id="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
              />
            </div>
            <div className="mt-4">
              <label className="block" htmlFor="password">
                Password
              </label>
              <input
                type="password"
                placeholder="Password"
                className="w-full px-4 py-2 mt-2 border rounded-md focus:outline-none focus:ring-1 focus:ring-blue-600"
                id="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
              />
            </div>
            <div className="mt-4">
              <label className="block" htmlFor="confirm-password">
                Confirm Password
              </label>
              <input
                type="password"
                placeholder="Confirm Password"
                className="w-full px-4 py-2 mt-2 border rounded-md focus:outline-none focus:ring-1 focus:ring-blue-600"
                id="confirm-password"
                value={confirmPassword}
                onChange={(e) => setConfirmPassword(e.target.value)}
                required
              />
            </div>
            <div className="mt-4 flex justify-center">
              <ReCAPTCHA
                ref={recaptchaRef}
                sitekey="YOUR_RECAPTCHA_SITE_KEY" // Replace with your reCAPTCHA site key
                onChange={handleRecaptchaChange}
              />
            </div>
            {error && <p className="text-red-500 text-xs italic mt-4">{error}</p>}
            <div className="flex items-baseline justify-between">
              <button
                type="submit"
                className="px-6 py-2 mt-4 text-white bg-blue-600 rounded-lg hover:bg-blue-900"
              >
                Sign Up
              </button>
              <Link to="/signin" className="text-sm text-blue-600 hover:underline">
                Already have an account? Sign In
              </Link>
            </div>
          </div>
        </form>
      </div>
    </div>
  );
};

export default SignUpPage;
