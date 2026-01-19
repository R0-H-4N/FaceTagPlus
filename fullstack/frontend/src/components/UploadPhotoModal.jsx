import { useState } from 'react';
import { photoAPI } from '../utils/api';
import './Modal.css';

const UploadPhotoModal = ({ groupId, onClose, onPhotoUploaded }) => {
  const [files, setFiles] = useState([]);
  const [caption, setCaption] = useState('');
  const [previewUrls, setPreviewUrls] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [uploadProgress, setUploadProgress] = useState(0);

  const handleFileChange = (e) => {
    const selectedFiles = Array.from(e.target.files);
    if (selectedFiles.length === 0) return;

    // Validate all files
    const validFiles = [];
    const newPreviewUrls = [];
    let hasError = false;

    for (const file of selectedFiles) {
      // Validate file type
      if (!file.type.startsWith('image/')) {
        setError('Please select only valid image files');
        hasError = true;
        break;
      }

      // Validate file size (max 10MB)
      if (file.size > 10 * 1024 * 1024) {
        setError('Each file must be less than 10MB');
        hasError = true;
        break;
      }

      validFiles.push(file);
      
      // Create preview
      const reader = new FileReader();
      reader.onloadend = () => {
        newPreviewUrls.push(reader.result);
        if (newPreviewUrls.length === validFiles.length) {
          setPreviewUrls(newPreviewUrls);
        }
      };
      reader.readAsDataURL(file);
    }

    if (!hasError) {
      setFiles(validFiles);
      setError('');
    }
  };

  const removeFile = (index) => {
    setFiles(files.filter((_, i) => i !== index));
    setPreviewUrls(previewUrls.filter((_, i) => i !== index));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (files.length === 0) {
      setError('Please select at least one photo to upload');
      return;
    }

    setLoading(true);
    setError('');

    try {
      const data = await photoAPI.uploadPhoto(groupId, files, caption);
      
      // Call onPhotoUploaded for each successfully uploaded photo
      if (data.photos && Array.isArray(data.photos)) {
        data.photos.forEach(photo => onPhotoUploaded(photo));
      }
      
      // Show success message if there were any failures
      if (data.uploadResults) {
        const failed = data.uploadResults.filter(r => !r.success);
        if (failed.length > 0) {
          setError(`${failed.length} photo(s) failed to upload`);
          setTimeout(() => {
            onClose();
          }, 2000);
        } else {
          onClose();
        }
      } else {
        onClose();
      }
    } catch (err) {
      setError(err.response?.data?.error || 'Failed to upload photos');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h2>Upload Photo</h2>
          <button className="close-btn" onClick={onClose}>&times;</button>
        </div>

        <form onSubmit={handleSubmit}>
          {error && <div className="error-message">{error}</div>}

          <div className="form-group">
            <label htmlFor="photo">Select Photos *</label>
            <input
              type="file"
              id="photo"
              onChange={handleFileChange}
              accept="image/*"
              multiple
              required
            />
            <p className="help-text">You can select multiple photos (max 20)</p>
            {previewUrls.length > 0 && (
              <div className="image-previews">
                {previewUrls.map((url, index) => (
                  <div key={index} className="image-preview-item">
                    <img src={url} alt={`Preview ${index + 1}`} />
                    <button 
                      type="button" 
                      className="remove-image-btn"
                      onClick={() => removeFile(index)}
                      disabled={loading}
                    >
                      &times;
                    </button>
                  </div>
                ))}
              </div>
            )}
          </div>

          <div className="form-group">
            <label htmlFor="caption">Caption (optional)</label>
            <textarea
              id="caption"
              value={caption}
              onChange={(e) => setCaption(e.target.value)}
              placeholder="Add a caption for this photo..."
              rows="3"
            />
          </div>

          {loading && (
            <div className="upload-progress">
              <p>Uploading {files.length} photo{files.length > 1 ? 's' : ''} and extracting face embeddings...</p>
              <div className="progress-bar">
                <div className="progress-fill"></div>
              </div>
            </div>
          )}

          <div className="modal-actions">
            <button 
              type="button" 
              onClick={onClose} 
              className="btn-secondary"
              disabled={loading}
            >
              Cancel
            </button>
            <button 
              type="submit" 
              className="btn-primary"
              disabled={loading}
            >
              {loading ? 'Uploading...' : `Upload Photo${files.length > 1 ? 's' : ''}`}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default UploadPhotoModal;
