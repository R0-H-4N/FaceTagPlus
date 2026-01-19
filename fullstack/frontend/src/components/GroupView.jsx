import { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { groupAPI, photoAPI } from '../utils/api';
import { useAuth } from '../context/AuthContext';
import UploadPhotoModal from './UploadPhotoModal';
import PhotoGallery from './PhotoGallery';
import './GroupView.css';

const GroupView = () => {
  const { groupId } = useParams();
  const navigate = useNavigate();
  const { user } = useAuth();
  
  const [group, setGroup] = useState(null);
  const [photos, setPhotos] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [showUploadModal, setShowUploadModal] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [searching, setSearching] = useState(false);
  const [searchResults, setSearchResults] = useState(null);
  const [queryInfo, setQueryInfo] = useState(null);

  useEffect(() => {
    fetchGroupData();
  }, [groupId]);

  const fetchGroupData = async () => {
    try {
      setLoading(true);
      const [groupData, photosData] = await Promise.all([
        groupAPI.getGroup(groupId),
        photoAPI.getGroupPhotos(groupId),
      ]);
      setGroup(groupData.group);
      setPhotos(photosData.photos);
    } catch (err) {
      setError('Failed to load group data');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handlePhotoUploaded = (newPhoto) => {
    setPhotos([newPhoto, ...photos]);
    setShowUploadModal(false);
  };

  const handleSearch = async (e) => {
    e.preventDefault();
    if (!searchQuery.trim()) {
      setSearchResults(null);
      setQueryInfo(null);
      return;
    }

    try {
      setSearching(true);
      setError('');
      const data = await photoAPI.searchPhotos(groupId, searchQuery, true); // Always use AI search
      setSearchResults(data.photos);
      
      // Store query info if available
      if (data.generatedQuery) {
        setQueryInfo({
          generated: data.generatedQuery,
          parsed: data.parsedQuery
        });
      } else {
        setQueryInfo(null);
      }
    } catch (err) {
      setError(err.response?.data?.error || 'Search failed. Please try again.');
      console.error('Search error:', err);
    } finally {
      setSearching(false);
    }
  };

  const handleClearSearch = () => {
    setSearchQuery('');
    setSearchResults(null);
    setQueryInfo(null);
  };

  const handleDownloadPhoto = async (photoId, filename) => {
    try {
      const blob = await photoAPI.downloadPhoto(photoId);
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = filename || 'photo.jpg';
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (err) {
      console.error('Download failed:', err);
      alert('Failed to download photo');
    }
  };

  if (loading) {
    return <div className="loading-page">Loading group...</div>;
  }

  if (error && !group) {
    return (
      <div className="error-page">
        <p>{error}</p>
        <button onClick={() => navigate('/dashboard')} className="btn-primary">
          Back to Dashboard
        </button>
      </div>
    );
  }

  const displayPhotos = searchResults !== null ? searchResults : photos;

  return (
    <div className="group-view">
      <header className="group-header">
        <div className="header-left">
          <button onClick={() => navigate('/dashboard')} className="back-btn">
            ← Back to Dashboard
          </button>
          <div className="group-info">
            <h1>{group?.name}</h1>
            <p>{group?.description}</p>
            <div className="group-meta">
              <span>{group?.members?.length || 0} members</span>
              <span>•</span>
              <span>{photos.length} photos</span>
            </div>
          </div>
        </div>
        {user?.profilePicture && (
          <div className="user-profile-pic">
            <img 
              src={`http://localhost:3000${user.profilePicture}`} 
              alt={user.name}
              title={user.name}
            />
          </div>
        )}
      </header>

      <div className="group-actions">
        <button 
          onClick={() => setShowUploadModal(true)} 
          className="btn-primary"
        >
          📤 Upload Photo
        </button>
        
        <div className="search-container">
          <form onSubmit={handleSearch} className="search-form">
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Ask naturally: 'Show me photos of Alice' or 'Find beach photos'"
              className="search-input"
            />
            <button type="submit" className="btn-run" disabled={searching}>
              {searching ? 'Running...' : '▶ Run'}
            </button>
            {searchResults !== null && (
              <button 
                type="button" 
                onClick={handleClearSearch}
                className="btn-secondary"
              >
                Clear
              </button>
            )}
          </form>
          
          {!searchResults && (
            <div className="search-examples">
              <span className="examples-label">💡 Try asking:</span>
              <button 
                type="button" 
                onClick={() => setSearchQuery("I need photo of " + (user?.username || "username"))}
                className="example-query"
              >
                I need photo of {user?.username || "username"}
              </button>
              <button 
                type="button" 
                onClick={() => setSearchQuery("Show me beach photos")}
                className="example-query"
              >
                Show me beach photos
              </button>
              <button 
                type="button" 
                onClick={() => setSearchQuery("Find nature photos")}
                className="example-query"
              >
                Find nature photos
              </button>
            </div>
          )}
        </div>
      </div>

      {error && <div className="error-message">{error}</div>}

      {searchResults !== null && (
        <div className="search-info">
          <div className="search-results-header">
            Found {searchResults.length} photo(s) matching "{searchQuery}"
          </div>
          {queryInfo && (
            <div className="query-info">
              <details>
                <summary>🔍 Query Details</summary>
                <div className="query-details">
                  <div className="query-item">
                    <strong>Generated:</strong> <code>{queryInfo.generated}</code>
                  </div>
                  <div className="query-item">
                    <strong>MongoDB Query:</strong> <code>{JSON.stringify(queryInfo.parsed, null, 2)}</code>
                  </div>
                </div>
              </details>
            </div>
          )}
        </div>
      )}

      <PhotoGallery 
        photos={displayPhotos}
        onDownload={handleDownloadPhoto}
      />

      {showUploadModal && (
        <UploadPhotoModal
          groupId={groupId}
          onClose={() => setShowUploadModal(false)}
          onPhotoUploaded={handlePhotoUploaded}
        />
      )}
    </div>
  );
};

export default GroupView;
