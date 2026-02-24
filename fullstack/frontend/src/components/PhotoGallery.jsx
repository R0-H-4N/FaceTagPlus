import { useState, useEffect } from 'react';
import './PhotoGallery.css';

const API_BASE_URL = import.meta.env.VITE_BASE_URL || 'http://localhost:3000';

const PhotoGallery = ({ photos, onDownload }) => {
  const [selectedPhoto, setSelectedPhoto] = useState(null);

  // Keyboard navigation
  useEffect(() => {
    if (!selectedPhoto) return;

    const handleKeyDown = (e) => {
      if (e.key === 'Escape') {
        setSelectedPhoto(null);
      } else if (e.key === 'ArrowLeft') {
        const currentIndex = photos.findIndex(p => p._id === selectedPhoto._id);
        if (currentIndex > 0) {
          setSelectedPhoto(photos[currentIndex - 1]);
        }
      } else if (e.key === 'ArrowRight') {
        const currentIndex = photos.findIndex(p => p._id === selectedPhoto._id);
        if (currentIndex < photos.length - 1) {
          setSelectedPhoto(photos[currentIndex + 1]);
        }
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [selectedPhoto, photos]);

  if (!photos || photos.length === 0) {
    return (
      <div className="empty-gallery">
        <p>No photos to display</p>
        <p>Upload your first photo to get started!</p>
      </div>
    );
  }

  const formatDate = (dateString) => {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
    });
  };

  const getPhotoUrl = (url) => {
    // If URL is already absolute, return as is
    if (url.startsWith('http')) {
      return url;
    }
    // Otherwise, prepend the backend URL
    return `${API_BASE_URL}${url}`;
  };

  const handlePhotoClick = (photo) => {
    setSelectedPhoto(photo);
  };

  const handleClosePreview = () => {
    setSelectedPhoto(null);
  };

  const handlePrevPhoto = (e) => {
    e.stopPropagation();
    const currentIndex = photos.findIndex(p => p._id === selectedPhoto._id);
    if (currentIndex > 0) {
      setSelectedPhoto(photos[currentIndex - 1]);
    }
  };

  const handleNextPhoto = (e) => {
    e.stopPropagation();
    const currentIndex = photos.findIndex(p => p._id === selectedPhoto._id);
    if (currentIndex < photos.length - 1) {
      setSelectedPhoto(photos[currentIndex + 1]);
    }
  };

  return (
    <>
      <div className="photo-gallery">
        {photos.map((photo) => (
          <div key={photo._id} className="photo-card">
            <div className="photo-image" onClick={() => handlePhotoClick(photo)}>
              <img 
                src={getPhotoUrl(photo.url)} 
                alt={photo.caption || 'Photo'} 
                loading="lazy"
              />
              <div className="photo-overlay">
                <span className="preview-icon">🔍</span>
              </div>
            </div>
            <div className="photo-info">
              {photo.caption && <p className="photo-caption">{photo.caption}</p>}
              <div className="photo-meta">
                <span className="photo-uploader">
                  👤 {photo.uploadedBy?.name || 'Unknown'}
                </span>
                <span className="photo-date">
                  📅 {formatDate(photo.uploadedAt)}
                </span>
              </div>
              {photo.faceEmbeddings && photo.faceEmbeddings.length > 0 && (
                <div className="photo-faces">
                  <div className="faces-count">
                    👥 {photo.faceEmbeddings.length} face(s) recognized
                  </div>
                  <div className="faces-list">
                    {photo.faceEmbeddings.map((face, idx) => (
                      face.identifiedUserDetails ? (
                        <span key={idx} className="face-tag">
                          {face.identifiedUserDetails.name || face.identifiedUserDetails.username}
                        </span>
                      ) : face.identifiedUser ? (
                        <span key={idx} className="face-tag">
                          {face.identifiedUser}
                        </span>
                      ) : null
                    ))}
                  </div>
                </div>
              )}
              <button 
                onClick={(e) => {
                  e.stopPropagation();
                  onDownload(photo._id, photo.filename);
                }}
                className="btn-download"
              >
                ⬇️ Download
              </button>
            </div>
          </div>
        ))}
      </div>

      {/* Photo Preview Modal */}
      {selectedPhoto && (
        <div className="photo-preview-modal" onClick={handleClosePreview}>
          <div className="preview-header">
            <button className="close-preview" onClick={handleClosePreview}>
              &times;
            </button>
          </div>
          
          <div className="preview-content" onClick={(e) => e.stopPropagation()}>
            <img 
              src={getPhotoUrl(selectedPhoto.url)} 
              alt={selectedPhoto.caption || 'Photo'} 
            />
          </div>

          {/* Navigation arrows */}
          {photos.length > 1 && (
            <>
              <button 
                className="preview-nav prev" 
                onClick={handlePrevPhoto}
                disabled={photos.findIndex(p => p._id === selectedPhoto._id) === 0}
              >
                ‹
              </button>
              <button 
                className="preview-nav next" 
                onClick={handleNextPhoto}
                disabled={photos.findIndex(p => p._id === selectedPhoto._id) === photos.length - 1}
              >
                ›
              </button>
            </>
          )}

          {/* Photo info overlay */}
          <div className="preview-info">
            {selectedPhoto.caption && (
              <h3 className="preview-caption">{selectedPhoto.caption}</h3>
            )}
            <div className="preview-meta">
              <span>👤 {selectedPhoto.uploadedBy?.name || 'Unknown'}</span>
              <span>📅 {formatDate(selectedPhoto.uploadedAt)}</span>
              {selectedPhoto.faceEmbeddings && selectedPhoto.faceEmbeddings.length > 0 && (
                <>
                  <span>👥 {selectedPhoto.faceEmbeddings.length} face(s)</span>
                  <div className="preview-faces">
                    {selectedPhoto.faceEmbeddings.map((face, idx) => (
                      face.identifiedUserDetails ? (
                        <span key={idx} className="preview-face-tag">
                          {face.identifiedUserDetails.name || face.identifiedUserDetails.username}
                        </span>
                      ) : face.identifiedUser ? (
                        <span key={idx} className="preview-face-tag">
                          {face.identifiedUser}
                        </span>
                      ) : null
                    ))}
                  </div>
                </>
              )}
            </div>
            <button 
              className="preview-download-btn"
              onClick={(e) => {
                e.stopPropagation();
                onDownload(selectedPhoto._id, selectedPhoto.filename);
              }}
            >
              ⬇️ Download Photo
            </button>
          </div>
        </div>
      )}
    </>
  );
};

export default PhotoGallery;
