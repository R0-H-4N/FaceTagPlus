import axios from 'axios';

// Use environment variable or fallback to localhost for development
const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:3000/api';

// Create axios instance
const api = axios.create({
  baseURL: API_URL,
});

// Add token to requests if available
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  
  // Set Content-Type for non-FormData requests
  if (!(config.data instanceof FormData)) {
    config.headers['Content-Type'] = 'application/json';
  }
  // For FormData, browser will set Content-Type automatically with boundary
  
  return config;
});

// Handle response errors (e.g., expired tokens)
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      // Token is invalid or expired
      localStorage.removeItem('token');
      localStorage.removeItem('user');
      // Only redirect if not already on login/signup page
      if (!window.location.pathname.includes('/login') && !window.location.pathname.includes('/signup')) {
        window.location.href = '/login';
      }
    }
    return Promise.reject(error);
  }
);

// Auth APIs
export const authAPI = {
  signup: async (userData) => {
    // userData is FormData, so Content-Type will be set automatically
    const response = await api.post('/auth/signup', userData);
    return response.data;
  },
  login: async (credentials) => {
    const response = await api.post('/auth/login', credentials);
    return response.data;
  },
  logout: () => {
    localStorage.removeItem('token');
    localStorage.removeItem('user');
  },
};

// User APIs
export const userAPI = {
  getProfile: async () => {
    const response = await api.get('/users/profile');
    return response.data;
  },
  searchUsers: async (query) => {
    const response = await api.get(`/users/search?query=${query}`);
    return response.data;
  },
};

// Group APIs
export const groupAPI = {
  getGroups: async () => {
    const response = await api.get('/groups');
    return response.data;
  },
  getGroup: async (groupId) => {
    const response = await api.get(`/groups/${groupId}`);
    return response.data;
  },
  createGroup: async (groupData) => {
    const response = await api.post('/groups', groupData);
    return response.data;
  },
  addMember: async (groupId, userId) => {
    const response = await api.post(`/groups/${groupId}/members`, { userId });
    return response.data;
  },
  removeMember: async (groupId, userId) => {
    const response = await api.delete(`/groups/${groupId}/members/${userId}`);
    return response.data;
  },
};

// Photo APIs
export const photoAPI = {
  uploadPhoto: async (groupId, files, caption) => {
    const formData = new FormData();
    
    // Support both single file and array of files
    if (Array.isArray(files)) {
      files.forEach(file => {
        formData.append('photos', file);
      });
    } else {
      formData.append('photos', files);
    }
    
    if (caption) formData.append('description', caption);
    
    const response = await api.post(`/photos/upload/${groupId}`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  },
  getGroupPhotos: async (groupId) => {
    const response = await api.get(`/photos/group/${groupId}`);
    return response.data;
  },
  searchPhotos: async (groupId, query, useNaturalLanguage = false) => {
    const response = await api.post(`/photos/search/${groupId}`, {
      query,
      naturalLanguage: useNaturalLanguage
    });
    return response.data;
  },
  downloadPhoto: async (photoId) => {
    const response = await api.get(`/photos/download/${photoId}`, {
      responseType: 'blob',
    });
    return response.data;
  },
};

export default api;
