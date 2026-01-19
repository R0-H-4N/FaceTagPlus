const express = require('express');
const Group = require('../models/Group');
const User = require('../models/User');
const auth = require('../middleware/auth');

const router = express.Router();

// Create a new group
router.post('/', auth, async (req, res) => {
  try {
    const { name, description, memberIds } = req.body;

    if (!name) {
      return res.status(400).json({ error: 'Group name is required' });
    }

    // Create group with creator as first member
    const group = new Group({
      name,
      description: description || '',
      creator: req.userId,
      members: [req.userId]
    });

    // Add additional members if provided
    if (memberIds && Array.isArray(memberIds)) {
      const validMembers = await User.find({ _id: { $in: memberIds } });
      const validMemberIds = validMembers.map(m => m._id);
      
      // Add valid members (avoid duplicates)
      validMemberIds.forEach(id => {
        if (!group.members.includes(id)) {
          group.members.push(id);
        }
      });
    }

    await group.save();

    // Add group to all members' groups array
    await User.updateMany(
      { _id: { $in: group.members } },
      { $addToSet: { groups: group._id } }
    );

    const populatedGroup = await Group.findById(group._id)
      .populate('creator', 'name username profilePicture')
      .populate('members', 'name username profilePicture');

    res.status(201).json({
      message: 'Group created successfully',
      group: populatedGroup
    });
  } catch (error) {
    console.error('Create group error:', error);
    res.status(500).json({ error: 'Error creating group' });
  }
});

// Get all groups for current user
router.get('/', auth, async (req, res) => {
  try {
    const groups = await Group.find({ members: req.userId })
      .populate('creator', 'name username profilePicture')
      .populate('members', 'name username profilePicture')
      .sort({ updatedAt: -1 });

    res.json({ groups });
  } catch (error) {
    console.error('Get groups error:', error);
    res.status(500).json({ error: 'Error fetching groups' });
  }
});

// Get single group details
router.get('/:groupId', auth, async (req, res) => {
  try {
    const group = await Group.findById(req.params.groupId)
      .populate('creator', 'name username profilePicture')
      .populate('members', 'name username profilePicture')
      .populate({
        path: 'photos',
        populate: { path: 'uploadedBy', select: 'name username profilePicture' }
      });

    if (!group) {
      return res.status(404).json({ error: 'Group not found' });
    }

    // Check if user is a member
    if (!group.members.some(member => member._id.equals(req.userId))) {
      return res.status(403).json({ error: 'You are not a member of this group' });
    }

    res.json({ group });
  } catch (error) {
    console.error('Get group error:', error);
    res.status(500).json({ error: 'Error fetching group' });
  }
});

// Add members to group
router.post('/:groupId/members', auth, async (req, res) => {
  try {
    const { memberIds } = req.body;

    if (!memberIds || !Array.isArray(memberIds)) {
      return res.status(400).json({ error: 'Member IDs array is required' });
    }

    const group = await Group.findById(req.params.groupId);

    if (!group) {
      return res.status(404).json({ error: 'Group not found' });
    }

    // Check if user is a member or creator
    if (!group.members.includes(req.userId)) {
      return res.status(403).json({ error: 'You are not a member of this group' });
    }

    // Validate and add new members
    const validMembers = await User.find({ _id: { $in: memberIds } });
    const validMemberIds = validMembers.map(m => m._id);

    const newMembers = [];
    validMemberIds.forEach(id => {
      if (!group.members.includes(id)) {
        group.members.push(id);
        newMembers.push(id);
      }
    });

    await group.save();

    // Add group to new members' groups array
    if (newMembers.length > 0) {
      await User.updateMany(
        { _id: { $in: newMembers } },
        { $addToSet: { groups: group._id } }
      );
    }

    const populatedGroup = await Group.findById(group._id)
      .populate('creator', 'name username profilePicture')
      .populate('members', 'name username profilePicture');

    res.json({
      message: `${newMembers.length} member(s) added successfully`,
      group: populatedGroup
    });
  } catch (error) {
    console.error('Add members error:', error);
    res.status(500).json({ error: 'Error adding members' });
  }
});

// Remove member from group
router.delete('/:groupId/members/:memberId', auth, async (req, res) => {
  try {
    const { groupId, memberId } = req.params;

    const group = await Group.findById(groupId);

    if (!group) {
      return res.status(404).json({ error: 'Group not found' });
    }

    // Only creator can remove members, or members can remove themselves
    if (!group.creator.equals(req.userId) && !req.userId.equals(memberId)) {
      return res.status(403).json({ error: 'Permission denied' });
    }

    // Don't allow removing the creator
    if (group.creator.equals(memberId)) {
      return res.status(400).json({ error: 'Cannot remove group creator' });
    }

    // Remove member
    group.members = group.members.filter(id => !id.equals(memberId));
    await group.save();

    // Remove group from user's groups array
    await User.findByIdAndUpdate(memberId, {
      $pull: { groups: groupId }
    });

    res.json({ message: 'Member removed successfully' });
  } catch (error) {
    console.error('Remove member error:', error);
    res.status(500).json({ error: 'Error removing member' });
  }
});

// Delete group
router.delete('/:groupId', auth, async (req, res) => {
  try {
    const group = await Group.findById(req.params.groupId);

    if (!group) {
      return res.status(404).json({ error: 'Group not found' });
    }

    // Only creator can delete group
    if (!group.creator.equals(req.userId)) {
      return res.status(403).json({ error: 'Only group creator can delete the group' });
    }

    // Remove group from all members
    await User.updateMany(
      { _id: { $in: group.members } },
      { $pull: { groups: group._id } }
    );

    await Group.findByIdAndDelete(req.params.groupId);

    res.json({ message: 'Group deleted successfully' });
  } catch (error) {
    console.error('Delete group error:', error);
    res.status(500).json({ error: 'Error deleting group' });
  }
});

module.exports = router;
