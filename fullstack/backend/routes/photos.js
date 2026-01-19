const express = require('express');
const Photo = require('../models/Photo');
const Group = require('../models/Group');
const User = require('../models/User');
const auth = require('../middleware/auth');
const axios = require('axios');
const FormData = require('form-data');
const path = require('path');
const fs = require('fs').promises;
const multer = require('multer');

const router = express.Router();
const FASTAPI_URL = process.env.FASTAPI_URL || 'http://localhost:8000';
const UPLOAD_DIR = path.join(__dirname, '../uploads/photos');

// Configure multer for handling file uploads
const storage = multer.memoryStorage();
const upload = multer({ 
  storage: storage,
  limits: { fileSize: 10 * 1024 * 1024 } // 10MB limit per file
});

// Ensure upload directory exists
fs.mkdir(UPLOAD_DIR, { recursive: true }).catch(console.error);

// Helper function to populate identified users manually
async function populateIdentifiedUsers(photos) {
  // Collect all unique usernames from face embeddings
  const allUsernames = new Set();
  const photoArray = Array.isArray(photos) ? photos : [photos];
  
  photoArray.forEach(photo => {
    if (photo.faceEmbeddings) {
      photo.faceEmbeddings.forEach(face => {
        if (face.identifiedUser) {
          allUsernames.add(face.identifiedUser);
        }
      });
    }
  });

  // Fetch user details for all identified usernames
  const usersMap = {};
  if (allUsernames.size > 0) {
    const users = await User.find({ 
      username: { $in: Array.from(allUsernames) } 
    }, 'name username profilePicture');
    
    users.forEach(user => {
      usersMap[user.username] = {
        _id: user._id,
        name: user.name,
        username: user.username,
        profilePicture: user.profilePicture
      };
    });
  }

  // Transform photos to include user details
  return photoArray.map(photo => {
    const photoObj = photo.toObject ? photo.toObject() : photo;
    if (photoObj.faceEmbeddings) {
      photoObj.faceEmbeddings = photoObj.faceEmbeddings.map(face => ({
        ...face,
        identifiedUserDetails: face.identifiedUser ? usersMap[face.identifiedUser] : null
      }));
    }
    return photoObj;
  });
}

// Upload photo(s) to group - supports single and multiple uploads
router.post('/upload/:groupId', auth, upload.array('photos', 20), async (req, res) => {
  try {
    const files = req.files || (req.file ? [req.file] : []);
    
    if (!files || files.length === 0) {
      return res.status(400).json({ error: 'No photos provided' });
    }

    const { groupId } = req.params;
    const { description, tags } = req.body;

    // Check if group exists and user is a member
    const group = await Group.findById(groupId);
    if (!group) {
      return res.status(404).json({ error: 'Group not found' });
    }

    if (!group.members.includes(req.userId)) {
      return res.status(403).json({ error: 'You are not a member of this group' });
    }

    // Parse tags once (same for all photos)
    const parsedTags = tags ? tags.split(',').map(tag => tag.trim()) : [];

    // Get all group members with face embeddings
    const groupMembers = await User.find({ 
      _id: { $in: group.members },
      faceEmbedding: { $ne: null }
    });

    // Save all photos to disk first
    const photoMetadata = [];
    for (const file of files) {
      const filename = `${Date.now()}-${Math.random().toString(36).substring(7)}-${file.originalname}`;
      const filepath = path.join(UPLOAD_DIR, filename);
      await fs.writeFile(filepath, file.buffer);
      
      photoMetadata.push({
        file,
        filename,
        filepath,
        originalName: file.originalname
      });
    }

    // Call FastAPI /getlabels to identify faces in all photos at once
    let identificationResults = [];
    if (groupMembers.length > 0) {
      try {
        // Create reference embeddings dictionary with username as key
        const referenceEmbeddings = {};
        groupMembers.forEach(member => {
          referenceEmbeddings[member.username] = member.faceEmbedding;
        });

        // Prepare FormData for /getlabels
        const formData = new FormData();
        
        // Add all photo files
        files.forEach(file => {
          formData.append('files', file.buffer, {
            filename: file.originalname,
            contentType: file.mimetype
          });
        });
        
        // Add reference embeddings
        formData.append('reference_embeddings', JSON.stringify(referenceEmbeddings));

        // Call /getlabels endpoint
        const labelsResponse = await axios.post(
          `${FASTAPI_URL}/getlabels`,
          formData,
          {
            headers: {
              ...formData.getHeaders()
            }
          }
        );

        identificationResults = labelsResponse.data.results || [];
        console.log('Face identification results:', identificationResults);
      } catch (identifyError) {
        console.error('Face identification error:', identifyError.message);
        // Continue without identification
      }
    }

    // Call FastAPI /classifyscene to classify scenes in all photos
    let sceneResults = [];
    try {
      // Prepare FormData for /classifyscene
      const sceneFormData = new FormData();
      
      // Add all photo files
      files.forEach(file => {
        sceneFormData.append('files', file.buffer, {
          filename: file.originalname,
          contentType: file.mimetype
        });
      });

      // Call /classifyscene endpoint
      const sceneResponse = await axios.post(
        `${FASTAPI_URL}/classifyscene`,
        sceneFormData,
        {
          headers: {
            ...sceneFormData.getHeaders()
          }
        }
      );

      sceneResults = sceneResponse.data.results || [];
      console.log('Scene classification results:', sceneResults);
    } catch (sceneError) {
      console.error('Scene classification error:', sceneError.message);
      // Continue without scene classification
    }

    // Create photo documents with identified faces
    const photoIds = [];
    const uploadResults = [];
    
    for (let i = 0; i < photoMetadata.length; i++) {
      const metadata = photoMetadata[i];
      try {
        // Get identification results for this photo
        const photoLabels = identificationResults.find(
          r => r.filename === metadata.originalName
        );
        
        // Build faceEmbeddings array with identified users
        // labels now contain usernames instead of userIds
        const faceEmbeddings = [];
        let unknownFacesCount = 0;
        if (photoLabels && photoLabels.labels && Array.isArray(photoLabels.labels)) {
          photoLabels.labels.forEach(label => {
            faceEmbeddings.push({
              identifiedUser: label !== 'unknown' ? label : null, // username
              embedding: [] // We don't store the embedding in the photo, only the reference
            });
          });
          // Extract unknown faces count from FastAPI response
          unknownFacesCount = photoLabels.unknown_faces || 0;
        }

        // Get scene classification result for this photo
        const sceneData = sceneResults.find(
          r => r.filename === metadata.originalName
        );
        const detectedScene = sceneData && sceneData.top_scene ? sceneData.top_scene : '';

        // Create photo document
        const photo = new Photo({
          filename: metadata.filename,
          originalName: metadata.originalName,
          url: `/uploads/photos/${metadata.filename}`,
          uploadedBy: req.userId,
          group: groupId,
          faceEmbeddings,
          unknownFacesCount,
          scene: detectedScene,
          tags: parsedTags,
          description: description || ''
        });

        await photo.save();
        photoIds.push(photo._id);
        uploadResults.push({ 
          success: true, 
          filename: metadata.originalName,
          facesDetected: faceEmbeddings.length
        });
      } catch (error) {
        console.error(`Error processing photo ${metadata.originalName}:`, error);
        uploadResults.push({ 
          success: false, 
          filename: metadata.originalName, 
          error: error.message 
        });
      }
    }

    // Add successfully uploaded photos to group
    if (photoIds.length > 0) {
      group.photos.push(...photoIds);
      await group.save();
    }

    // Fetch photos with uploadedBy populated
    const populatedPhotos = await Photo.find({ _id: { $in: photoIds } })
      .populate('uploadedBy', 'name username profilePicture');

    // Manually populate identified users using helper function
    const photosWithUserDetails = await populateIdentifiedUsers(populatedPhotos);

    const successCount = uploadResults.filter(r => r.success).length;
    const failCount = uploadResults.filter(r => !r.success).length;

    res.status(201).json({
      message: `Successfully uploaded ${successCount} photo(s)${failCount > 0 ? `, ${failCount} failed` : ''}`,
      photos: photosWithUserDetails,
      uploadResults
    });
  } catch (error) {
    console.error('Upload photo error:', error);
    res.status(500).json({ error: 'Error uploading photos' });
  }
});

// Get all photos in a group
router.get('/group/:groupId', auth, async (req, res) => {
  try {
    const { groupId } = req.params;

    // Check if group exists and user is a member
    const group = await Group.findById(groupId);
    if (!group) {
      return res.status(404).json({ error: 'Group not found' });
    }

    if (!group.members.includes(req.userId)) {
      return res.status(403).json({ error: 'You are not a member of this group' });
    }

    const photos = await Photo.find({ group: groupId })
      .populate('uploadedBy', 'name username profilePicture')
      .sort({ uploadedAt: -1 });

    // Manually populate identified users using helper function
    const photosWithUserDetails = await populateIdentifiedUsers(photos);

    res.json({ photos: photosWithUserDetails });
  } catch (error) {
    console.error('Get photos error:', error);
    res.status(500).json({ error: 'Error fetching photos' });
  }
});

// Search photos by query (user name, tags, etc.)
router.post('/search/:groupId', auth, async (req, res) => {
  try {
    const { groupId } = req.params;
    const { query, searchType, naturalLanguage } = req.body;

    // Check if group exists and user is a member
    const group = await Group.findById(groupId);
    if (!group) {
      return res.status(404).json({ error: 'Group not found' });
    }

    if (!group.members.includes(req.userId)) {
      return res.status(403).json({ error: 'You are not a member of this group' });
    }

    let photos = [];
    let parsedQuery = null;
    let generatedQuery = null;

    // Natural language search using T5 model
    if (naturalLanguage) {
      try {
        console.log('Natural language query:', query);
        
        // Call FastAPI to convert natural language to MongoDB query
        const formData = new FormData();
        formData.append('natural_language', query);
        
        const fastApiResponse = await axios.post(`${FASTAPI_URL}/getquery`, formData, {
          headers: formData.getHeaders()
        });

        if (!fastApiResponse.data.success) {
          return res.status(400).json({ 
            error: 'Failed to parse natural language query',
            details: fastApiResponse.data.error 
          });
        }

        parsedQuery = fastApiResponse.data.parsed;
        generatedQuery = fastApiResponse.data.query;
        
        console.log('Generated query:', generatedQuery);
        console.log('Parsed MongoDB query:', parsedQuery);
        console.log('Final MongoDB query:', { group: groupId, ...parsedQuery });

        // Execute the parsed MongoDB query
        photos = await Photo.find({
          group: groupId,
          ...parsedQuery
        })
          .populate('uploadedBy', 'name username profilePicture')
          .sort({ uploadedAt: -1 });

      } catch (error) {
        console.error('Natural language search error:', error);
        return res.status(500).json({ 
          error: 'Failed to process natural language query',
          details: error.message 
        });
      }
    } 
    // Traditional search
    else if (searchType === 'person') {
      // Search by person username (identifiedUser is now a username string)
      photos = await Photo.find({
        group: groupId,
        'faceEmbeddings.identifiedUser': { $regex: query, $options: 'i' }
      })
        .populate('uploadedBy', 'name username profilePicture')
        .sort({ uploadedAt: -1 });

    } else if (searchType === 'tag') {
      // Search by tags
      photos = await Photo.find({
        group: groupId,
        tags: { $regex: query, $options: 'i' }
      })
        .populate('uploadedBy', 'name username profilePicture')
        .sort({ uploadedAt: -1 });

    } else {
      // General search (description, tags, etc.)
      photos = await Photo.find({
        group: groupId,
        $or: [
          { description: { $regex: query, $options: 'i' } },
          { tags: { $regex: query, $options: 'i' } },
          { originalName: { $regex: query, $options: 'i' } }
        ]
      })
        .populate('uploadedBy', 'name username profilePicture')
        .sort({ uploadedAt: -1 });
    }

    // Manually populate identified users using helper function
    const photosWithUserDetails = await populateIdentifiedUsers(photos);

    res.json({ 
      photos: photosWithUserDetails, 
      count: photosWithUserDetails.length,
      ...(naturalLanguage && { 
        generatedQuery, 
        parsedQuery 
      })
    });
  } catch (error) {
    console.error('Search photos error:', error);
    res.status(500).json({ error: 'Error searching photos' });
  }
});

// Download photo - MUST come before /:groupId/:photoId to avoid route conflict
router.get('/download/:photoId', auth, async (req, res) => {
  try {
    const { photoId } = req.params;

    const photo = await Photo.findById(photoId);
    if (!photo) {
      return res.status(404).json({ error: 'Photo not found' });
    }

    // Check if user is a member of the group
    const group = await Group.findById(photo.group);
    if (!group) {
      return res.status(404).json({ error: 'Group not found' });
    }

    if (!group.members.includes(req.userId)) {
      return res.status(403).json({ error: 'You are not a member of this group' });
    }

    // Send the file
    const filepath = path.join(UPLOAD_DIR, photo.filename);
    res.download(filepath, photo.originalName || photo.filename, (err) => {
      if (err) {
        console.error('Download error:', err);
        if (!res.headersSent) {
          res.status(500).json({ error: 'Error downloading photo' });
        }
      }
    });
  } catch (error) {
    console.error('Download photo error:', error);
    if (!res.headersSent) {
      res.status(500).json({ error: 'Error downloading photo' });
    }
  }
});

// Get single photo details
router.get('/:groupId/:photoId', auth, async (req, res) => {
  try {
    const { groupId, photoId } = req.params;

    // Check if group exists and user is a member
    const group = await Group.findById(groupId);
    if (!group) {
      return res.status(404).json({ error: 'Group not found' });
    }

    if (!group.members.includes(req.userId)) {
      return res.status(403).json({ error: 'You are not a member of this group' });
    }

    const photo = await Photo.findById(photoId)
      .populate('uploadedBy', 'name username profilePicture');

    if (!photo) {
      return res.status(404).json({ error: 'Photo not found' });
    }

    // Manually populate identified users using helper function
    const [photoWithUserDetails] = await populateIdentifiedUsers([photo]);

    res.json({ photo: photoWithUserDetails });
  } catch (error) {
    console.error('Get photo error:', error);
    res.status(500).json({ error: 'Error fetching photo' });
  }
});

// Delete photo
router.delete('/:groupId/:photoId', auth, async (req, res) => {
  try {
    const { groupId, photoId } = req.params;

    const photo = await Photo.findById(photoId);
    if (!photo) {
      return res.status(404).json({ error: 'Photo not found' });
    }

    // Check if user is the uploader or group creator
    const group = await Group.findById(groupId);
    if (!group.creator.equals(req.userId) && !photo.uploadedBy.equals(req.userId)) {
      return res.status(403).json({ error: 'Permission denied' });
    }

    // Delete file from disk
    const filepath = path.join(UPLOAD_DIR, photo.filename);
    try {
      await fs.unlink(filepath);
    } catch (err) {
      console.error('Error deleting file:', err);
    }

    // Remove from group
    await Group.findByIdAndUpdate(groupId, {
      $pull: { photos: photoId }
    });

    await Photo.findByIdAndDelete(photoId);

    res.json({ message: 'Photo deleted successfully' });
  } catch (error) {
    console.error('Delete photo error:', error);
    res.status(500).json({ error: 'Error deleting photo' });
  }
});

module.exports = router;
