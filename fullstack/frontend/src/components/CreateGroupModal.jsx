import { useState } from 'react';
import { groupAPI, userAPI } from '../utils/api';
import './Modal.css';

const CreateGroupModal = ({ onClose, onGroupCreated }) => {
  const [formData, setFormData] = useState({
    name: '',
    description: '',
  });
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [selectedMembers, setSelectedMembers] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [searching, setSearching] = useState(false);

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value,
    });
  };

  const handleSearch = async (e) => {
    const query = e.target.value;
    setSearchQuery(query);
    
    if (query.length < 2) {
      setSearchResults([]);
      return;
    }

    try {
      setSearching(true);
      const data = await userAPI.searchUsers(query);
      setSearchResults(data.users);
    } catch (err) {
      console.error('Search error:', err);
    } finally {
      setSearching(false);
    }
  };

  const toggleMember = (user) => {
    const isSelected = selectedMembers.find(m => m._id === user._id);
    if (isSelected) {
      setSelectedMembers(selectedMembers.filter(m => m._id !== user._id));
    } else {
      setSelectedMembers([...selectedMembers, user]);
    }
    setSearchQuery('');
    setSearchResults([]);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    
    if (!formData.name.trim()) {
      setError('Group name is required');
      return;
    }

    setLoading(true);

    try {
      const groupData = {
        name: formData.name,
        description: formData.description,
        memberIds: selectedMembers.map(m => m._id),
      };
      
      const data = await groupAPI.createGroup(groupData);
      onGroupCreated(data.group);
    } catch (err) {
      setError(err.response?.data?.error || 'Failed to create group');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h2>Create New Group</h2>
          <button className="close-btn" onClick={onClose}>&times;</button>
        </div>

        <form onSubmit={handleSubmit}>
          {error && <div className="error-message">{error}</div>}

          <div className="form-group">
            <label htmlFor="name">Group Name *</label>
            <input
              type="text"
              id="name"
              name="name"
              value={formData.name}
              onChange={handleChange}
              required
              placeholder="Enter group name"
            />
          </div>

          <div className="form-group">
            <label htmlFor="description">Description</label>
            <textarea
              id="description"
              name="description"
              value={formData.description}
              onChange={handleChange}
              placeholder="Enter group description (optional)"
              rows="3"
            />
          </div>

          <div className="form-group">
            <label htmlFor="search">Add Members</label>
            <input
              type="text"
              id="search"
              value={searchQuery}
              onChange={handleSearch}
              placeholder="Search users by name or username"
            />
            {searching && <div className="search-loading">Searching...</div>}
            {searchResults.length > 0 && (
              <div className="search-results">
                {searchResults.map(user => (
                  <div 
                    key={user._id} 
                    className="search-result-item"
                    onClick={() => toggleMember(user)}
                  >
                    <span>{user.name} (@{user.username})</span>
                    <button type="button" className="btn-small">Add</button>
                  </div>
                ))}
              </div>
            )}
          </div>

          {selectedMembers.length > 0 && (
            <div className="selected-members">
              <label>Selected Members:</label>
              <div className="member-tags">
                {selectedMembers.map(member => (
                  <span key={member._id} className="member-tag">
                    {member.name}
                    <button 
                      type="button"
                      onClick={() => toggleMember(member)}
                      className="remove-tag"
                    >
                      &times;
                    </button>
                  </span>
                ))}
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
              {loading ? 'Creating...' : 'Create Group'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default CreateGroupModal;
