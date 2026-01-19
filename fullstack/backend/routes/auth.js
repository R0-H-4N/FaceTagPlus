const express = require('express');
const jwt = require('jsonwebtoken');
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const User = require('../models/User');
const auth = require('../middleware/auth');
const axios = require('axios');
const FormData = require('form-data');

const router = express.Router();
const JWT_SECRET = process.env.JWT_SECRET || 'your-secret-key-change-this';
const FASTAPI_URL = process.env.FASTAPI_URL || 'http://localhost:8000';

// Configure multer for handling file uploads
const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    const uploadDir = path.join(__dirname, '../uploads/profiles');
    // Create directory if it doesn't exist
    if (!fs.existsSync(uploadDir)) {
      fs.mkdirSync(uploadDir, { recursive: true });
    }
    cb(null, uploadDir);
  },
  filename: function (req, file, cb) {
    const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1E9);
    cb(null, 'profile-' + uniqueSuffix + path.extname(file.originalname));
  }
});

const upload = multer({ 
  storage: storage,
  limits: { fileSize: 10 * 1024 * 1024 } // 10MB limit
});

// Generate JWT token
const generateToken = (userId) => {
  return jwt.sign({ userId }, JWT_SECRET, { expiresIn: '7d' });
};

// Signup with face image
router.post('/signup', upload.single('faceImage'), async (req, res) => {
  try {
    console.log('=== Signup Request Received ===');
    console.log('Body:', req.body);
    console.log('File:', req.file ? {
      fieldname: req.file.fieldname,
      originalname: req.file.originalname,
      mimetype: req.file.mimetype,
      size: req.file.size
    } : 'No file received');
    console.log('==============================');
    
    const { name, username, password } = req.body;

    // Validate input
    if (!name || !username || !password) {
      return res.status(400).json({ error: 'All fields are required' });
    }

    if (password.length < 6) {
      return res.status(400).json({ error: 'Password must be at least 6 characters' });
    }

    if (!req.file) {
      return res.status(400).json({ error: 'Face image is required' });
    }

    // Check if user already exists
    const existingUser = await User.findOne({ username });
    if (existingUser) {
      return res.status(400).json({ error: 'User with this username already exists' });
    }

    // Save profile picture path
    const profilePicturePath = `/uploads/profiles/${req.file.filename}`;

    // Extract face embedding from the uploaded image
    let faceEmbedding = null;
    try {
      const formData = new FormData();
      const fileBuffer = fs.readFileSync(req.file.path);
      formData.append('files', fileBuffer, {
        filename: req.file.originalname,
        contentType: req.file.mimetype
      });
      formData.append('labels', username); // Use username as the label

      console.log('Calling face embedding service...');
      const embeddingResponse = await axios.post(
        `${FASTAPI_URL}/getembeddings`, 
        formData,
        {
          headers: {
            ...formData.getHeaders()
          }
        }
      );

      console.log('Embedding response:', JSON.stringify(embeddingResponse.data, null, 2));

      // Check for errors in the response
      if (embeddingResponse.data.error !== 'No error') {
        console.error('Face embedding error:', embeddingResponse.data.error);
        return res.status(400).json({ 
          error: embeddingResponse.data.error[username] || 'Failed to extract face embedding. Please ensure the image contains a clear face.' 
        });
      }

      // Get the embedding for this user
      const embeddings = embeddingResponse.data.embeddings;
      if (!embeddings[username]) {
        return res.status(400).json({ 
          error: 'Failed to extract face embedding. Please ensure the image contains a clear face.' 
        });
      }

      faceEmbedding = embeddings[username];
      console.log(faceEmbedding);
      console.log(`Successfully extracted face embedding (${faceEmbedding.length} dimensions)`);
      
    } catch (embeddingError) {
      console.error('Face embedding error:', embeddingError.response?.data || embeddingError.message);
      
      // If FastAPI service is not available, provide a helpful error message
      if (embeddingError.code === 'ECONNREFUSED') {
        return res.status(503).json({ 
          error: 'Face recognition service is currently unavailable. Please ensure the FastAPI service is running on port 8000.' 
        });
      }
      
      return res.status(400).json({ 
        error: 'Failed to extract face embedding. Please ensure the image contains a clear face.' 
      });
    }

    // Create user with face embedding and profile picture
    const user = new User({
      name,
      username,
      password,
      faceEmbedding,
      profilePicture: profilePicturePath
    });

    await user.save();

    // Generate token
    const token = generateToken(user._id);

    res.status(201).json({
      message: 'User created successfully',
      token,
      user: {
        id: user._id,
        name: user.name,
        username: user.username,
        profilePicture: user.profilePicture
      }
    });
  } catch (error) {
    console.error('Signup error:', error);
    res.status(500).json({ error: 'Error creating user' });
  }
});

// Login
router.post('/login', async (req, res) => {
  try {
    const { username, password } = req.body;

    // Validate input
    if (!username || !password) {
      return res.status(400).json({ error: 'Username and password are required' });
    }

    // Find user
    const user = await User.findOne({ username });
    if (!user) {
      return res.status(401).json({ error: 'Invalid credentials' });
    }

    // Check password
    const isMatch = await user.comparePassword(password);
    if (!isMatch) {
      return res.status(401).json({ error: 'Invalid credentials' });
    }

    // Generate token
    const token = generateToken(user._id);

    res.json({
      message: 'Login successful',
      token,
      user: {
        id: user._id,
        name: user.name,
        username: user.username,
        profilePicture: user.profilePicture
      }
    });
  } catch (error) {
    console.error('Login error:', error);
    res.status(500).json({ error: 'Error logging in' });
  }
});

// Get current user profile
router.get('/me', auth, async (req, res) => {
  try {
    const user = await User.findById(req.userId).populate('groups', 'name description');
    res.json({ user });
  } catch (error) {
    console.error('Get profile error:', error);
    res.status(500).json({ error: 'Error fetching profile' });
  }
});

// Upload face embedding for user (during signup or profile update)
router.post('/upload-face', auth, async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No face picture provided' });
    }

    // Send to FastAPI to extract face embedding
    const formData = new FormData();
    formData.append('face_picture', req.file.buffer, {
      filename: req.file.originalname,
      contentType: req.file.mimetype
    });

    const response = await axios.post(`${FASTAPI_URL}/face-embedding`, formData, {
      headers: {
        ...formData.getHeaders()
      }
    });

    const faceEmbedding = response.data.face_embedding;

    if (!faceEmbedding || faceEmbedding.length === 0) {
      return res.status(400).json({ error: 'No face detected in the image' });
    }

    // Use first face embedding
    const embedding = faceEmbedding[0];

    // Update user with face embedding
    const user = await User.findById(req.userId);
    user.faceEmbedding = embedding;
    await user.save();

    res.json({
      message: 'Face embedding uploaded successfully',
      user: {
        id: user._id,
        name: user.name,
        username: user.username,
        faceEmbedding: user.faceEmbedding
      }
    });
  } catch (error) {
    console.error('Upload face error:', error);
    res.status(500).json({ error: 'Error uploading face embedding' });
  }
});

module.exports = router;
