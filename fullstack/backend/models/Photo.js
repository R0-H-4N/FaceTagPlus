const mongoose = require('mongoose');

const photoSchema = new mongoose.Schema({
  filename: {
    type: String,
    required: true
  },
  originalName: {
    type: String,
    required: true
  },
  url: {
    type: String,
    required: true
  },
  uploadedBy: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User',
    required: true
  },
  group: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'Group',
    required: true
  },
  // Store face embeddings detected in this photo
  faceEmbeddings: [{
    embedding: {
      type: [Number],
      required: false,
      validate: {
        validator: function(v) {
          // If embedding exists, it must be 512-dimensional
          return !v || v.length === 0 || v.length === 512;
        },
        message: 'Face embedding must be empty or 512-dimensional'
      }
    },
    // Identified person username (if matched with a user)
    identifiedUser: {
      type: String,
      default: null
    },
    // Bounding box coordinates (if available)
    boundingBox: {
      x: Number,
      y: Number,
      width: Number,
      height: Number
    }
  }],
  tags: [{
    type: String,
    trim: true
  }],
  description: {
    type: String,
    default: ''
  },
  scene: {
    type: String,
    default: ''
  },
  unknownFacesCount: {
    type: Number,
    default: 0
  },
  uploadedAt: {
    type: Date,
    default: Date.now
  }
});

// Index for faster embedding searches
photoSchema.index({ 'faceEmbeddings.identifiedUser': 1 });
photoSchema.index({ group: 1 });
photoSchema.index({ uploadedAt: -1 });

module.exports = mongoose.model('Photo', photoSchema);
