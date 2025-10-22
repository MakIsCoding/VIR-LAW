// API Configuration for Hugging Face Space Backend
const API_BASE_URL = process.env.REACT_APP_API_URL || "https://MakIsCoding-VirLaw.hf.space";
const HF_TOKEN = process.env.REACT_APP_HF_TOKEN;

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
