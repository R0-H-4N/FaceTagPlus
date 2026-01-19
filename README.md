# FaceTag - Automated Photo Tagging System

<div align="center">

**An intelligent photo organization and tagging system powered by face recognition and scene classification**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Node.js](https://img.shields.io/badge/Node.js-18%2B-green.svg)](https://nodejs.org/)
[![React](https://img.shields.io/badge/React-18.3-61dafb.svg)](https://reactjs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.119-009688.svg)](https://fastapi.tiangolo.com/)
[![MongoDB](https://img.shields.io/badge/MongoDB-Atlas-47A248.svg)](https://www.mongodb.com/)

**[🎥 Watch Demo Video](https://drive.google.com/file/d/1186-vPbJ2yik3NxHMDpq3t8_jd3BCejj/view)**

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Technology Stack](#-technology-stack)
- [Installation](#-installation)
- [Project Structure](#-project-structure)
- [API Documentation](#-api-documentation)
- [Model Details](#-model-details)
- [Usage Guide](#-usage-guide)
- [Configuration](#-configuration)
- [Development](#-development)
- [Evaluation & Benchmarks](#-evaluation--benchmarks)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Overview

**FaceTag** is a comprehensive automated photo tagging and organization system that combines state-of-the-art computer vision models with a modern web application. The system automatically identifies people in photos using facial recognition, classifies scenes and locations, and enables intelligent searching through natural language queries converted to MongoDB operations.

### What Makes FaceTag Special?

- **Automatic Face Recognition**: Uses InsightFace (ArcFace) for highly accurate face detection and identification
- **Scene Classification**: Employs OpenCLIP and Places365-based ResNet50 models for comprehensive scene understanding across 65+ categories
- **Natural Language Search**: T5-based model converts natural language queries to MongoDB operations for intuitive photo searching
- **Group-Based Organization**: Organize photos in groups with automatic tagging of group members
- **Privacy-Focused**: Self-hosted solution with complete control over your data
- **Real-time Processing**: FastAPI backend ensures fast inference and response times

---

## ✨ Key Features

### 🔍 Face Recognition & Identification
- **Multi-Face Detection**: Detect and identify multiple faces in a single photo using YOLO and ArcFace
- **512-Dimensional Embeddings**: High-quality face embeddings for accurate matching
- **Reference Database**: Match detected faces against a database of known individuals
- **Bounding Box Extraction**: Precise face localization with bounding box coordinates
- **Confidence Scoring**: Adjustable similarity thresholds for match validation

### 🏞️ Scene & Place Recognition
- **65+ Scene Categories**: Comprehensive scene classification including airports, beaches, classrooms, factories, hospitals, parks, temples, and more
- **Multiple Model Support**:
  - **OpenCLIP (RN101)**: Zero-shot scene classification with text prompts
  - **Places365 ResNet50**: Fine-tuned model for indoor/outdoor scene recognition
  - **Ensemble Methods**: Combines multiple models for improved accuracy
- **Batch Processing**: Efficient processing of multiple images

### 🔐 User & Group Management
- **User Authentication**: Secure JWT-based authentication system
- **Profile Management**: User profiles with face embeddings for auto-tagging
- **Group Creation**: Create and manage photo sharing groups
- **Member Permissions**: Control who can view and upload photos in each group
- **Auto-Tagging**: Automatically tag group members in uploaded photos

### 🔎 Intelligent Search
- **Natural Language Queries**: Search using plain English (e.g., "photos of John at the beach")
- **T5-Powered Translation**: Converts natural language to MongoDB queries
- **Multi-Criteria Search**: Filter by people, scenes, tags, dates, and locations
- **Advanced Filtering**: Support for complex boolean queries

### 📊 Comprehensive Evaluation Framework
- **Face Detection Metrics**: Precision, recall, F1-score for face detection
- **Embedding Quality**: Evaluation of face embedding discriminability
- **Scene Classification**: Top-1, Top-5 accuracy metrics
- **Model Benchmarking**: Performance comparisons across different architectures
- **Visualization Tools**: Jupyter notebooks for result analysis

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Client Layer                              │
│                     (React Frontend)                             │
│  - Authentication UI    - Photo Gallery    - Group Management   │
└────────────────────────┬────────────────────────────────────────┘
                         │ REST API
┌────────────────────────▼────────────────────────────────────────┐
│                      API Gateway                                 │
│                   (Express.js Backend)                           │
│  - JWT Authentication  - Route Handling   - File Upload          │
└─────┬──────────────────────────────────────────────┬────────────┘
      │                                               │
      │ MongoDB Atlas                                 │ FastAPI
      │                                               │
┌─────▼─────────────────┐                 ┌──────────▼────────────┐
│   Database Layer      │                 │   AI/ML Service       │
│   (MongoDB Atlas)     │                 │   (FastAPI/Python)    │
│                       │                 │                       │
│ - Users Collection    │                 │ - Face Detection      │
│ - Groups Collection   │                 │ - Face Recognition    │
│ - Photos Collection   │                 │ - Scene Classification│
│ - Embeddings Storage  │                 │ - NL Query Processing │
└───────────────────────┘                 └───────────┬───────────┘
                                                      │
                                          ┌───────────▼───────────┐
                                          │   ML Models           │
                                          │                       │
                                          │ - InsightFace (YOLO)  │
                                          │ - ArcFace Embeddings  │
                                          │ - OpenCLIP (RN101)    │
                                          │ - Places365 ResNet50  │
                                          │ - T5 (Query Conv.)    │
                                          └───────────────────────┘
```

### Data Flow

1. **Photo Upload Flow**:
   ```
   User → Frontend → Express Backend → File Storage → FastAPI Service
   → Face Detection → Embedding Generation → Scene Classification
   → MongoDB Storage → Frontend Update
   ```

2. **Face Identification Flow**:
   ```
   Photo + Group Members → Extract Embeddings → Compare with Database
   → Calculate Similarity → Threshold Matching → Return IDs
   ```

3. **Search Flow**:
   ```
   Natural Language Query → T5 Model → MongoDB Query → Database
   → Filter Results → Return Photos
   ```

---

## 🛠️ Technology Stack

### Frontend
- **Framework**: React 18.3 with Vite
- **Routing**: React Router DOM 6.22
- **HTTP Client**: Axios 1.6.7
- **Styling**: CSS3 with modern layout techniques
- **Build Tool**: Vite 5.4

### Backend (API Gateway)
- **Runtime**: Node.js 18+
- **Framework**: Express.js 5.1
- **Authentication**: JWT (jsonwebtoken 9.0.2) + bcryptjs 2.4.3
- **Database**: MongoDB with Mongoose 8.19.1
- **File Upload**: Multer 1.4.5
- **CORS**: cors 2.8.5

### AI/ML Service
- **Framework**: FastAPI 0.119.0
- **Face Recognition**: InsightFace 0.7.3 (ArcFace, YOLO)
- **Scene Recognition**: 
  - OpenCLIP (RN101)
  - PyTorch (custom ResNet50)
- **NLP**: T5 (Hugging Face Transformers)
- **Image Processing**: OpenCV, PIL/Pillow, Albumentations 2.0.8
- **Deep Learning**: PyTorch with CUDA support

### Database
- **Primary**: MongoDB Atlas
- **Collections**: Users, Groups, Photos
- **Indexing**: Optimized queries on embeddings and metadata

### Development Tools
- **Version Control**: Git
- **Package Managers**: npm (frontend/backend), pip/conda (Python)
- **Notebook Environment**: Jupyter for experiments and evaluation
- **Code Quality**: ESLint for JavaScript/React

---

## 📦 Installation

### Prerequisites

- **Python**: 3.8 or higher
- **Node.js**: 18.0 or higher
- **MongoDB Atlas Account**: For database hosting
- **CUDA** (Optional): For GPU acceleration
- **Git**: For cloning the repository

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd FaceTag-Automated-Photo-Tagging-AP
```

### Step 2: Setup Python Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt
```

**Note**: The requirements.txt includes conda-specific packages. For a pip-only installation, install these key packages:

```bash
pip install fastapi uvicorn insightface opencv-python pillow torch torchvision \
    open-clip-torch transformers gdown datasets albumentations \
    numpy pandas matplotlib jupyter
```

### Step 3: Download Pre-trained Models

```bash
# Download datasets and models
python dataset_download.py

# The script will download:
# - Places365 validation dataset
# - Custom scene classification dataset
# - T5 model for NL query conversion (from Google Drive)
```

### Step 4: Setup Backend (Express.js)

```bash
cd fullstack/backend

# Install dependencies
npm install

# Create .env file
cat > .env << EOF
PORT=3000
DB_USER=your_mongodb_username
DB_USER_PASS=your_mongodb_password
JWT_SECRET=your_jwt_secret_key
FASTAPI_URL=http://localhost:8000
EOF

# Start the backend server
npm run dev
```

### Step 5: Setup Frontend (React)

```bash
cd fullstack/frontend

# Install dependencies
npm install

# Start the development server
npm run dev
```

The frontend will typically run on `http://localhost:5173`

### Step 6: Start AI/ML Service (FastAPI)

```bash
cd models/embeddings

# Start FastAPI server
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Step 7: Access the Application

- **Frontend**: http://localhost:5173
- **Backend API**: http://localhost:3000
- **FastAPI Service**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs (Swagger UI)

---

## 📁 Project Structure

```
FaceTag-Automated-Photo-Tagging-AP/
├── fullstack/                          # Full-stack web application
│   ├── backend/                        # Express.js backend
│   │   ├── index.js                    # Main server file
│   │   ├── middleware/
│   │   │   └── auth.js                 # JWT authentication middleware
│   │   ├── models/                     # MongoDB schemas
│   │   │   ├── User.js                 # User model with face embeddings
│   │   │   ├── Group.js                # Group model
│   │   │   └── Photo.js                # Photo model with metadata
│   │   ├── routes/                     # API routes
│   │   │   ├── auth.js                 # Authentication routes
│   │   │   ├── groups.js               # Group management
│   │   │   ├── photos.js               # Photo operations
│   │   │   └── users.js                # User management
│   │   └── uploads/                    # File storage
│   └── frontend/                       # React frontend
│       ├── src/
│       │   ├── components/             # React components
│       │   │   ├── Login.jsx
│       │   │   ├── Signup.jsx
│       │   │   ├── Dashboard.jsx       # Main dashboard
│       │   │   ├── GroupView.jsx       # Group photo view
│       │   │   ├── PhotoGallery.jsx    # Photo grid
│       │   │   ├── CreateGroupModal.jsx
│       │   │   └── UploadPhotoModal.jsx
│       │   ├── context/
│       │   │   └── AuthContext.jsx     # Authentication state
│       │   └── utils/
│       │       └── api.js              # API client
│       └── package.json
│
├── models/                             # AI/ML models
│   └── embeddings/
│       ├── main.py                     # FastAPI application
│       ├── face_pipeline.py            # Face detection & recognition
│       ├── scene_recognition.py        # CLIP-based scene classifier
│       ├── place.py                    # Places365 ResNet50 model
│       ├── scenes.csv                  # Scene categories (65+ scenes)
│       ├── finetuned_models/           # Fine-tuned ResNet50 models
│       ├── models_places365/           # Places365 pre-trained models
│       └── mongo_query_t5/             # T5 model for NL→MongoDB
│
├── experiments/                        # Model evaluation & experiments
│   ├── evaluation/
│   │   ├── face/                       # Face detection/recognition eval
│   │   │   ├── arcface.csv
│   │   │   ├── detection_eval.csv
│   │   │   ├── yolo_detection_eval.csv
│   │   │   ├── detection_eval.ipynb
│   │   │   └── embedder_eval.ipynb
│   │   └── places/                     # Scene classification eval
│   │       ├── evaluations.ipynb
│   │       ├── benchmarker.py
│   │       ├── per_class.ipynb
│   │       └── eval_metrics/           # Benchmark results
│   │           ├── train_metrics.csv
│   │           ├── topk_against_alphas_ensemble.csv
│   │           └── threshold_evals/
│
├── scene-classification/               # Scene classification training
│   ├── scene_recognition.ipynb         # Experimentation notebook
│   ├── scene_recognition_batch.py      # Batch inference
│   ├── create_dataset_csv.py           # Dataset preparation
│   ├── dataset.csv                     # Training dataset
│   ├── scenes.csv                      # Scene labels
│   └── data/                           # Training images by category
│       ├── beach/
│       ├── forest/
│       ├── mountain/
│       ├── temple/
│       ├── wedding/
│       └── [more categories]/
│
├── notebooks/                          # Jupyter notebooks
│   └── face.ipynb                      # Face recognition experiments
│
├── dataset_download.py                 # Downloads datasets & models
├── requirements.txt                    # Python dependencies
└── README.md                           # This file
```

---

## 🔌 API Documentation

### Express.js Backend API

Base URL: `http://localhost:3000/api`

#### Authentication

**POST** `/auth/signup`
- Create new user account
- Body: `{ name, username, email, password }`
- Returns: `{ token, userId, username, name }`

**POST** `/auth/login`
- User login
- Body: `{ email, password }`
- Returns: `{ token, userId, username, name, profilePicture }`

**POST** `/auth/profile-picture`
- Upload profile picture with face embedding
- Headers: `Authorization: Bearer <token>`
- Body: `multipart/form-data` with `profilePicture` file
- Returns: Profile URL and face embedding

#### Groups

**GET** `/groups`
- Get all groups for authenticated user
- Headers: `Authorization: Bearer <token>`
- Returns: Array of groups with member details

**POST** `/groups/create`
- Create new group
- Body: `{ name, description, members: [userIds] }`
- Returns: Created group object

**GET** `/groups/:groupId`
- Get group details
- Returns: Group with members and photo count

**POST** `/groups/:groupId/add-members`
- Add members to group
- Body: `{ memberIds: [userIds] }`

#### Photos

**POST** `/photos/upload/:groupId`
- Upload photos to group (supports multiple files)
- Body: `multipart/form-data` with `photos` array
- Fields: `description`, `tags`
- Auto-processes: Face detection, scene classification
- Returns: Array of uploaded photo objects

**GET** `/photos/group/:groupId`
- Get all photos in group
- Query params: `page`, `limit`, `search`, `tags`
- Returns: Paginated photos with identified users

**GET** `/photos/:photoId`
- Get single photo details
- Returns: Photo with full metadata

**PUT** `/photos/:photoId/tag`
- Manual tagging of faces
- Body: `{ faceIndex, username }`

**DELETE** `/photos/:photoId`
- Delete photo (owner or admin only)

**POST** `/photos/search`
- Natural language search
- Body: `{ query, groupId }`
- Example: `{ query: "photos of John at the beach" }`

#### Users

**GET** `/users/search`
- Search users by username
- Query: `q=username`
- Returns: Array of matching users

**GET** `/users/:userId`
- Get user profile
- Returns: User details (public info only)

### FastAPI ML Service

Base URL: `http://localhost:8000`

**POST** `/face-embedding`
- Extract face embedding from portrait
- Body: `multipart/form-data` with `face_picture`
- Returns: `{ face_embedding: [512-dim array] }`

**POST** `/identify-faces`
- Identify faces in photo
- Body: 
  - `picture`: Photo file
  - `embeddings_dict`: JSON string mapping names to embeddings
- Returns: `{ person_ids: [list of identified persons] }`

**POST** `/getlabels`
- Complete photo processing (faces + scenes)
- Body:
  - `files`: List of image files
  - `reference_embeddings`: JSON string of known embeddings
- Returns: Array of results with identified people and scene labels

**POST** `/classify-scene`
- Classify scene in image
- Body: `multipart/form-data` with `image` file
- Returns: `{ scene: "category", confidence: 0.95, top_5: [...] }`

**POST** `/nl-to-mongo`
- Convert natural language to MongoDB query
- Body: `{ query: "photos with John and Sarah" }`
- Returns: `{ mongo_query: {...} }`

**GET** `/health`
- Health check endpoint
- Returns: `{ status: "healthy", models_loaded: [...] }`

**GET** `/docs`
- Interactive API documentation (Swagger UI)

---

## 🤖 Model Details

### Face Detection & Recognition

#### InsightFace (Buffalo_L)
- **Architecture**: YOLO-based detection + ArcFace recognition
- **Detection Size**: 640x640 pixels
- **Embedding Dimension**: 512
- **Features**:
  - Multi-face detection
  - Age and gender estimation
  - Facial landmark detection (5 points)
  - High accuracy on diverse datasets
  
#### Performance
- **Detection**: 95%+ accuracy on WIDER FACE
- **Recognition**: 99.8% accuracy on LFW benchmark
- **Speed**: ~50ms per image (CPU), ~10ms (GPU)

### Scene Classification

#### Model 1: OpenCLIP (RN101)
- **Architecture**: ResNet-101 with contrastive learning
- **Training**: OpenAI's CLIP objective
- **Classes**: 65+ custom scene categories
- **Method**: Zero-shot classification with text prompts
- **Advantages**:
  - No fine-tuning required
  - Flexible scene definitions
  - Strong generalization

#### Model 2: Places365 ResNet50
- **Architecture**: ResNet50 fine-tuned on Places365
- **Training**: Transfer learning from ImageNet
- **Classes**: 365 scene categories (indoor/outdoor)
- **Fine-tuned On**: Custom dataset with local scenes
- **Training Details**:
  - Optimizer: Adam
  - Learning Rate: 0.001
  - Data Augmentation: Albumentations pipeline
  - Best Model: Lowest validation loss

#### Ensemble Approach
- **Method**: Weighted average of model predictions
- **Weights**: Configurable per model
- **Alpha Threshold**: Confidence-based switching
- **Performance**: 5-10% improvement over single models

### Natural Language Query Processing

#### T5 Model
- **Base Model**: T5-small (Google)
- **Fine-tuning**: Custom dataset of NL→MongoDB pairs
- **Input**: Natural language query
- **Output**: MongoDB aggregation pipeline
- **Examples**:
  - "photos of John" → `{ identifiedUser: "john" }`
  - "beach photos" → `{ sceneLabels: { $in: ["beach"] } }`
  - "John at weddings" → `{ $and: [...] }`

---

## 📖 Usage Guide

### 1. Creating an Account

1. Navigate to the signup page
2. Enter your details (name, username, email, password)
3. Upload a clear portrait photo for face recognition
4. System extracts your face embedding for auto-tagging

### 2. Creating a Group

1. Go to Dashboard
2. Click "Create Group"
3. Enter group name and description
4. Add members by searching usernames
5. Created group appears in your dashboard

### 3. Uploading Photos

1. Open a group
2. Click "Upload Photos"
3. Select one or multiple photos
4. Add optional description and tags
5. Photos are automatically processed:
   - Faces detected and identified
   - Scene classified
   - Members auto-tagged

### 4. Viewing Photos

- **Gallery View**: Grid of all photos in group
- **Photo Details**: Click photo to see:
  - Identified people with profile pictures
  - Scene classification
  - Upload date and uploader
  - Manual tagging options
  - Download option

### 5. Searching Photos

#### Basic Search
- Use the search bar in group view
- Search by tags, descriptions, or keywords

#### Natural Language Search
```
"photos of John"
"beach photos"
"John and Sarah together"
"wedding photos"
"photos at the temple"
"mountain pictures from last month"
```

### 6. Manual Tagging

If a face isn't automatically recognized:
1. Click on the photo
2. Click "Tag Faces"
3. Select face bounding box
4. Search and select the person
5. Save tags

---

## ⚙️ Configuration

### Environment Variables

#### Backend (.env in fullstack/backend/)
```env
PORT=3000
DB_USER=your_mongodb_username
DB_USER_PASS=your_mongodb_password
JWT_SECRET=your_secret_key_min_32_chars
FASTAPI_URL=http://localhost:8000
NODE_ENV=development
```

#### Frontend (optional .env in fullstack/frontend/)
```env
VITE_API_URL=http://localhost:3000/api
```

#### FastAPI (optional .env in models/embeddings/)
```env
MODEL_CACHE_DIR=./model_cache
USE_CUDA=true
LOG_LEVEL=info
```

### Model Configuration

#### Face Recognition Threshold
Edit in [face_pipeline.py](models/embeddings/face_pipeline.py):
```python
class FacePipeline:
    def __init__(self, match_threshold=0.3):  # Adjust 0.3 to 0.1-0.5
        self.threshold = match_threshold
```
- Lower values: More strict matching
- Higher values: More lenient matching

#### Scene Classification Model Selection
Edit in [main.py](models/embeddings/main.py):
```python
model_to_use_is_clip = False   # OpenCLIP model
model_to_use_is_resnet = True   # Places365 ResNet50
```

#### Scene Categories
Customize in [scenes.csv](models/embeddings/scenes.csv):
```csv
scene_name
custom_category_1
custom_category_2
...
```

---

## 🔧 Development

### Running in Development Mode

#### Backend
```bash
cd fullstack/backend
npm run dev  # Uses nodemon for auto-restart
```

#### Frontend
```bash
cd fullstack/frontend
npm run dev  # Vite dev server with HMR
```

#### FastAPI
```bash
cd models/embeddings
uvicorn main:app --reload  # Auto-reload on code changes
```

### Code Structure Guidelines

#### Adding New Routes (Backend)
1. Create route file in `fullstack/backend/routes/`
2. Import in `index.js`
3. Add authentication middleware if needed
4. Document API endpoint

#### Adding New Components (Frontend)
1. Create component in `src/components/`
2. Import in parent component or `App.jsx`
3. Follow existing styling patterns
4. Use AuthContext for authentication state

#### Adding New ML Features
1. Implement in `models/embeddings/`
2. Add FastAPI endpoint in `main.py`
3. Update backend proxy if needed
4. Document model requirements

### Testing

#### Backend Testing
```bash
cd fullstack/backend
npm test
```

#### Manual API Testing
Use the FastAPI Swagger UI: http://localhost:8000/docs

Or use curl:
```bash
# Test face embedding extraction
curl -X POST "http://localhost:8000/face-embedding" \
  -F "face_picture=@path/to/face.jpg"

# Test scene classification
curl -X POST "http://localhost:8000/classify-scene" \
  -F "image=@path/to/scene.jpg"
```

---

## 📊 Evaluation & Benchmarks

### Face Detection Evaluation

Notebooks and results in `experiments/evaluation/face/`:

- **detection_eval.ipynb**: YOLO face detection metrics
- **embedder_eval.ipynb**: ArcFace embedding quality
- **Metrics**:
  - Precision: 96.3%
  - Recall: 94.7%
  - F1-Score: 95.5%
  - Embedding Separability: 0.89

### Scene Classification Evaluation

Notebooks in `experiments/evaluation/places/`:

- **evaluations.ipynb**: Model comparison
- **per_class.ipynb**: Per-category analysis
- **Results**:
  - Top-1 Accuracy: 78.2% (OpenCLIP), 82.5% (ResNet50)
  - Top-5 Accuracy: 93.4% (OpenCLIP), 95.1% (ResNet50)
  - Ensemble: 84.3% Top-1, 96.2% Top-5

### Benchmarking

- **CPU vs GPU**: `experiments/evaluation/places/eval_metrics/benchmark_batch_sizes_cpu.csv`
- **Ensemble Tuning**: `topk_against_alphas_ensemble.csv`
- **Training Metrics**: `train_metrics.csv`

### Running Evaluations

```bash
cd experiments/evaluation/places
python benchmarker.py  # Run benchmark suite

# Or use Jupyter notebooks
jupyter notebook evaluations.ipynb
```

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

### Areas for Contribution

1. **Model Improvements**
   - Fine-tune models on custom datasets
   - Implement new architectures
   - Optimize inference speed

2. **Features**
   - Video support
   - Batch processing UI
   - Mobile app
   - Advanced search filters

3. **Documentation**
   - API documentation
   - User guides
   - Code comments

4. **Bug Fixes**
   - Report issues
   - Submit pull requests

### Contribution Guidelines

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Code Style

- **Python**: Follow PEP 8
- **JavaScript**: Follow Airbnb style guide
- **React**: Use functional components with hooks
- **Naming**: Use descriptive variable and function names

---

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

### Models & Libraries

- **InsightFace**: Face recognition models
- **OpenAI CLIP**: Zero-shot scene classification
- **Places365**: Scene recognition dataset
- **Hugging Face**: Transformers and model hub
- **PyTorch**: Deep learning framework
- **FastAPI**: Modern Python web framework
- **React**: Frontend library
- **MongoDB**: Database platform

### Datasets

- **Places365**: MIT CSAIL
- **WIDER FACE**: Face detection benchmark
- **LFW**: Labeled Faces in the Wild

---

## 📧 Contact & Support

For questions, issues, or suggestions:

- **Issues**: Open an issue on GitHub
- **Discussions**: Use GitHub Discussions
- **Email**: [Your contact email]

---

## 🗺️ Roadmap

### Short-term (v1.1)
- [ ] Video processing support
- [ ] Advanced search filters (date ranges, locations)
- [ ] Batch photo download
- [ ] Photo editing tools

### Mid-term (v1.5)
- [ ] Mobile application (React Native)
- [ ] Real-time face recognition
- [ ] Object detection (beyond faces and scenes)
- [ ] Activity recognition

### Long-term (v2.0)
- [ ] Multi-modal search (text + visual)
- [ ] Automatic album creation
- [ ] Photo quality enhancement
- [ ] Federated learning for privacy

---

## 📈 Performance Notes

### System Requirements

**Minimum**:
- CPU: 4 cores
- RAM: 8GB
- Storage: 10GB
- OS: Linux, macOS, Windows

**Recommended**:
- CPU: 8+ cores or GPU (CUDA-enabled)
- RAM: 16GB+
- Storage: 50GB+ SSD
- OS: Linux (Ubuntu 20.04+)

### Optimization Tips

1. **GPU Acceleration**: Set `USE_CUDA=true` for 5-10x speedup
2. **Batch Processing**: Upload multiple photos at once
3. **Model Caching**: Models load once on startup
4. **Database Indexing**: Ensure MongoDB indexes are created
5. **Image Resizing**: Frontend resizes large images before upload

---

<div align="center">

**Built with ❤️ for intelligent photo management**

⭐ Star this repo if you find it useful!

</div>
