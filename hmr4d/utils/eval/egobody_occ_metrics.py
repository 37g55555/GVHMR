"""Local metric methods from mikeqzy/MoRo utils/eval_utils.py.
Pinned commit: 1f18c4f0d6db55d6b00327fb35c3e8d8d2fbfe82.
Only trailing whitespace is removed from the transplanted methods.
"""

import numpy as np
import torch
from typing import Dict, List


class EvalMetrics:
    def compute_jpe(self, S1: torch.Tensor, S2: torch.Tensor) -> np.ndarray:
        """Compute joint position error."""
        return torch.sqrt(((S1 - S2) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()

    def batch_align_by_pelvis(self, data_list: List[torch.Tensor],
                             pelvis_idxs: List[int] = [1, 2]) -> List[torch.Tensor]:
        """
        Align all data to the corresponding pelvis location.

        Args:
            data_list: [pred_j3d, target_j3d, pred_verts, target_verts]
            pelvis_idxs: Indices for pelvis joints

        Returns:
            Aligned data list
        """
        pred_j3d, target_j3d, pred_verts, target_verts = data_list

        pred_pelvis = pred_j3d[..., pelvis_idxs, :].mean(dim=-2, keepdims=True).clone()
        target_pelvis = target_j3d[..., pelvis_idxs, :].mean(dim=-2, keepdims=True).clone()

        # Align to the pelvis
        pred_j3d = pred_j3d - pred_pelvis
        target_j3d = target_j3d - target_pelvis
        pred_verts = pred_verts - pred_pelvis
        target_verts = target_verts - target_pelvis

        return [pred_j3d, target_j3d, pred_verts, target_verts]

    def compute_visibility_mpjpe(self, pred_joints: torch.Tensor, gt_joints: torch.Tensor,
                               visibility_mask: torch.Tensor) -> Dict[str, np.ndarray]:
        """
        Compute MPJPE for visible and occluded joints separately.

        Args:
            pred_joints: (..., J, 3) predicted joints
            gt_joints: (..., J, 3) ground truth joints
            visibility_mask: (..., J) boolean mask where True indicates visible joints

        Returns:
            Dictionary with visible_mpjpe and occluded_mpjpe
        """
        # Align by pelvis first
        pred_joints_aligned, gt_joints_aligned, _, _ = self.batch_align_by_pelvis(
            [pred_joints, gt_joints, pred_joints, gt_joints], pelvis_idxs=[1, 2]
        )

        # Compute joint position errors
        jpe = torch.sqrt(((pred_joints_aligned - gt_joints_aligned) ** 2).sum(dim=-1)) * 1000  # Convert to mm

        # Separate visible and occluded joints
        visible_mask = visibility_mask.bool()
        occluded_mask = ~visible_mask

        # Extract errors for visible and occluded joints
        visible_jpe = jpe[visible_mask]
        occluded_jpe = jpe[occluded_mask]

        return {
            "visible_mpjpe": visible_jpe.cpu().numpy(),
            "occluded_mpjpe": occluded_jpe.cpu().numpy(),
        }
