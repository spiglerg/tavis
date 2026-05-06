import copy
import math

import numpy as np
from scipy.spatial.transform import Rotation as R


# =============================================================================
# Action structure constants for quaternion representation
# 19D action: [pos_left(3), quat_left(4), pos_right(3), quat_right(4), ...]
# =============================================================================
QUAT_ACTION_INDICES = [3, 10]  # Starting indices of quaternion blocks in 19D action
QUAT_SIZE = 4


def _set_identity_normalization(feature_stats, idx):
    """Set identity normalization for a single index in a feature's stats dict.

    After this, ANY normalization mode (MIN_MAX, MEAN_STD, QUANTILES, QUANTILE10)
    will produce an identity transformation for the given index.
    """
    # MIN_MAX identity: 2*(x-(-1))/(1-(-1))-1 = x
    if 'min' in feature_stats:
        feature_stats['min'][idx] = -1.0
    if 'max' in feature_stats:
        feature_stats['max'][idx] = 1.0
    # MEAN_STD identity: (x-0)/1 = x
    if 'mean' in feature_stats:
        feature_stats['mean'][idx] = 0.0
    if 'std' in feature_stats:
        feature_stats['std'][idx] = 1.0
    # QUANTILES identity
    if 'q01' in feature_stats:
        feature_stats['q01'][idx] = -1.0
    if 'q99' in feature_stats:
        feature_stats['q99'][idx] = 1.0
    # QUANTILE10 identity
    if 'q10' in feature_stats:
        feature_stats['q10'][idx] = -1.0
    if 'q90' in feature_stats:
        feature_stats['q90'][idx] = 1.0


def fix_quat_stats(stats, quat_indices=None, quat_size=None):
    """
    Modify action stats so quaternion indices use identity normalization.

    Quaternion components are naturally bounded in [-1, 1] and must maintain
    unit norm. Element-wise normalization with varying min/max/mean/std breaks
    the geometric structure of quaternions. This function sets stats such that
    ANY normalization mode (MIN_MAX, MEAN_STD, QUANTILES, QUANTILE10) produces
    identity transformation for quaternion elements.

    Args:
        stats: Dataset statistics dictionary (will be deep-copied)
        quat_indices: Starting indices of quaternion blocks in action vector.
                      Defaults to QUAT_ACTION_INDICES.
        quat_size: Size of each quaternion block. Defaults to QUAT_SIZE.

    Returns:
        Modified stats dictionary with identity normalization for quaternion elements.
    """
    if quat_indices is None:
        quat_indices = QUAT_ACTION_INDICES
    if quat_size is None:
        quat_size = QUAT_SIZE

    stats = copy.deepcopy(stats)

    if 'action' not in stats:
        return stats

    action_stats = stats['action']

    for idx_start in quat_indices:
        for i in range(quat_size):
            _set_identity_normalization(action_stats, idx_start + i)

    return stats


def fix_constant_dims(stats, std_threshold=1e-2):
    """
    Set identity normalization for near-constant dimensions in ALL features.

    Dimensions with std < threshold are effectively constant in the training data
    (e.g., locked joints, base joints, antennas). Without this fix, the normalizer
    divides by their tiny std, amplifying any small eval-time deviation into an
    enormous OOD signal (e.g., a 0.3 rad difference at std=6e-4 becomes a normalized
    value of ~500, catastrophically OOD for the model).

    This is applied to ALL feature keys (observation.state, action, etc.) so that
    any near-constant dimension is passed through unchanged.

    Args:
        stats: Dataset statistics dictionary (will be modified in-place if already
               deep-copied, otherwise use the return value).
        std_threshold: Dimensions with std below this are treated as constant.
                       Default: 1e-2.

    Returns:
        Modified stats dictionary with identity normalization for constant dimensions.
    """
    n_fixed = 0
    for key, feature_stats in stats.items():
        if not isinstance(feature_stats, dict) or 'std' not in feature_stats:
            continue
        std = feature_stats['std']
        for idx in range(len(std)):
            if abs(float(std[idx])) < std_threshold:
                _set_identity_normalization(feature_stats, idx)
                n_fixed += 1

    if n_fixed > 0:
        print(f"[fix_constant_dims] Set identity normalization for {n_fixed} near-constant "
              f"dimensions (std < {std_threshold})")

    return stats


# =============================================================================
# Unity to Isaac coordinate conversion
# =============================================================================

def _unity_quat_to_isaac_scipy_R(unity_quat):
    """
    Convert quaternion from Unity to MuJoCo coordinate system, and returns it as a scipy Rotation object.

    Args:
        unity_quat: quaternion in Unity format, but already as [w, x, y, z, w]

    Returns:
        mujoco_quat: quaternion in MuJoCo format [w, x, y, z]
    """

    # Define coordinate system transformation matrix
    # Unity: X=right, Y=up, Z=forward
    # MuJoCo: X=forward, Y=left, Z=up
    coord_transform = np.array([
        [0,  0,  1],  # Unity Z -> MuJoCo X
        [-1, 0,  0],  # Unity X -> -MuJoCo Y
        [0,  1,  0]   # Unity Y -> MuJoCo Z
    ])

    # Convert Unity quaternion to rotation object
    # Quat format is [w, x, y, z], so convert to scipy format [x, y, z, w]
    unity_R = R.from_quat([unity_quat[1], unity_quat[2], unity_quat[3], unity_quat[0]])

    # Get rotation matrix in Unity coordinates
    unity_rotmat = unity_R.as_matrix()

    # Apply coordinate transformation: R_mujoco = T * R_unity * T^(-1)
    mujoco_rotmat = coord_transform @ unity_rotmat @ coord_transform.T

    # Convert back to quaternion
    mujoco_R = R.from_matrix(mujoco_rotmat)

    correction_R = R.from_euler('z', -90, degrees=True)
    corrected_mujoco_R = mujoco_R * correction_R

    return corrected_mujoco_R
