// API Configuration for Hugging Face Space Backend
const API_BASE_URL = import.meta.env.VITE_API_URL || "https://makiscoding-virlaw.hf.space";
const HF_TOKEN = import.meta.env.VITE_HF_TOKEN;

import axios from 'axios';

// Create axios instance with authentication
export const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
    ...(HF_TOKEN && { 'Authorization': `Bearer ${HF_TOKEN}` })
  },
  timeout: 90000  // 90 seconds for RAG queries
});

export { API_BASE_URL, HF_TOKEN };
