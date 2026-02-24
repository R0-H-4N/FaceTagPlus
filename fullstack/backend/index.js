const express = require('express');
const mongoose = require('mongoose');
const cors = require('cors');
const multer = require('multer');
const axios = require('axios');
const FormData = require('form-data');
require('dotenv').config();

// Import routes
const authRoutes = require('./routes/auth');
const groupRoutes = require('./routes/groups');
const photoRoutes = require('./routes/photos');
const userRoutes = require('./routes/users');

const app = express();

// Configure multer for handling file uploads
const storage = multer.memoryStorage();
const upload = multer({ 
  storage: storage,
  limits: { fileSize: 10 * 1024 * 1024 } // 10MB limit
});

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Serve static files (profile pictures and photos)
app.use('/uploads', express.static('uploads'));

// FastAPI service URL
const FASTAPI_URL = process.env.FASTAPI_URL || 'http://localhost:8000';

// Database connection
const db_user = process.env.DB_USER;
const db_user_pass = process.env.DB_USER_PASS;
const url =`mongodb+srv://${db_user}:${db_user_pass}@cluster0.y5kqkbm.mongodb.net/?appName=Cluster0`;
mongoose.connect(url)
  .then(() => {
    console.log('✅ Successfully connected to MongoDB');
  })
  .catch((err) => {
    console.error('❌ Error connecting to MongoDB:', err.message);
  });

// Interface 1: Get face embedding from a single face picture
app.post('/api/face-embedding', upload.single('face_picture'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No face picture provided' });
    }

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

    res.json({ face_embedding: response.data.face_embedding });
  } catch (error) {
    console.error('Error processing face embedding:', error);
    res.status(500).json({ error: 'Error processing face embedding' });
  }
});

// Interface 2: Get person IDs from a photo using face embeddings dictionary
app.post('/api/identify-faces', upload.single('picture'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No picture provided' });
    }

    const embeddings_dict = req.body.embeddings;
    if (!embeddings_dict) {
      return res.status(400).json({ error: 'No embeddings dictionary provided' });
    }

    const formData = new FormData();
    formData.append('picture', req.file.buffer, {
      filename: req.file.originalname,
      contentType: req.file.mimetype
    });
    formData.append('embeddings_dict', JSON.stringify(embeddings_dict));

    const response = await axios.post(`${FASTAPI_URL}/identify-faces`, formData, {
      headers: {
        ...formData.getHeaders()
      }
    });

    res.json({ person_ids: response.data.person_ids });
  } catch (error) {
    console.error('Error identifying faces:', error);
    res.status(500).json({ error: 'Error identifying faces' });
  }
});

// Use routes
app.use('/api/auth', authRoutes);
app.use('/api/groups', groupRoutes);
app.use('/api/photos', photoRoutes);
app.use('/api/users', userRoutes);

// Health check route
app.get('/health', (req, res) => {
  res.json({ status: 'ok', message: 'Server is running' });
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`🚀 Server is running on port ${PORT}`);
  console.log(`📊 Health check: http://localhost:${PORT}/health`);
});