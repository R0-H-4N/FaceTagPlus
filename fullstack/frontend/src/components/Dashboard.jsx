import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { groupAPI } from '../utils/api';
import CreateGroupModal from './CreateGroupModal';
import './Dashboard.css';

const Dashboard = () => {
  const [groups, setGroups] = useState([]);
  const [loading, setLoading] = useState(true);
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [error, setError] = useState('');
  
  const { user, logout } = useAuth();
  const navigate = useNavigate();

  useEffect(() => {
    fetchGroups();
  }, []);

  const fetchGroups = async () => {
    try {
      setLoading(true);
      const data = await groupAPI.getGroups();
      setGroups(data.groups);
    } catch (err) {
      setError('Failed to load groups');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  const handleGroupClick = (groupId) => {
    navigate(`/group/${groupId}`);
  };

  const handleGroupCreated = (newGroup) => {
    setGroups([...groups, newGroup]);
    setShowCreateModal(false);
  };

  return (
    <div className="dashboard">
      <header className="dashboard-header">
        <div className="header-content">
          <h1>FaceTag Dashboard</h1>
          <div className="user-info">
            {user?.profilePicture && (
              <div className="profile-picture">
                <img 
                  src={`http://localhost:3000${user.profilePicture}`} 
                  alt={user.name}
                />
              </div>
            )}
            <span>Welcome, {user?.name}</span>
            <button onClick={handleLogout} className="btn-secondary">
              Logout
            </button>
          </div>
        </div>
      </header>

      <main className="dashboard-content">
        <div className="groups-section">
          <div className="section-header">
            <h2>Your Groups</h2>
            <button 
              onClick={() => setShowCreateModal(true)} 
              className="btn-primary"
            >
              + Create New Group
            </button>
          </div>

          {error && <div className="error-message">{error}</div>}

          {loading ? (
            <div className="loading">Loading groups...</div>
          ) : groups.length === 0 ? (
            <div className="empty-state">
              <p>You don't have any groups yet.</p>
              <p>Create a new group to start sharing photos!</p>
            </div>
          ) : (
            <div className="groups-grid">
              {groups.map((group) => (
                <div 
                  key={group._id} 
                  className="group-card"
                  onClick={() => handleGroupClick(group._id)}
                >
                  <div className="group-icon">👥</div>
                  <h3>{group.name}</h3>
                  <p>{group.description}</p>
                  <div className="group-stats">
                    <span>{group.members?.length || 0} members</span>
                    <span>•</span>
                    <span>{group.photoCount || 0} photos</span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </main>

      {showCreateModal && (
        <CreateGroupModal 
          onClose={() => setShowCreateModal(false)}
          onGroupCreated={handleGroupCreated}
        />
      )}
    </div>
  );
};

export default Dashboard;
