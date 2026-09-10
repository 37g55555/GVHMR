"""GVHMR callback for the local part of pinned MoRo's EgoBody evaluator."""

import json
from pathlib import Path

import numpy as np
import torch
import pytorch_lightning as pl
from pytorch3d.transforms import axis_angle_to_matrix
from smplx import SMPL, SMPLHLayer, SMPLXLayer
from smplx.utils import Struct

from hmr4d.configs import MainStore, builds
from hmr4d.dataset.egobody.egobody_occ import MORO_COMMIT
from hmr4d.utils.comm.gather import all_gather
from hmr4d.utils.eval.egobody_occ_metrics import EvalMetrics
from hmr4d.utils.pylogger import Log


class MetricEgoBodyOcc(pl.Callback):
    def __init__(self, output_dir, body_model_root="inputs/checkpoints/body_models"):
        self.output_dir = Path(output_dir)
        self.body_model_root = Path(body_model_root)
        self.metrics = EvalMetrics()
        self.results = {}

    def on_test_start(self, trainer, pl_module):
        # MoRo BodyModel's SMPLH NPZ loading, without its unrelated mesh/renderer wrapper.
        self.gt_models = {}
        for gender in ["male", "female"]:
            bm_path = self.body_model_root / "smplh" / f"SMPLH_{gender.upper()}.npz"
            smpl_dict = np.load(bm_path, encoding="latin1", allow_pickle=True)
            data_struct = Struct(**smpl_dict)
            data_struct.hands_componentsl = np.zeros((0))
            data_struct.hands_componentsr = np.zeros((0))
            data_struct.hands_meanl = np.zeros((15 * 3))
            data_struct.hands_meanr = np.zeros((15 * 3))
            V, D, B = data_struct.shapedirs.shape
            data_struct.shapedirs = np.concatenate(
                [data_struct.shapedirs, np.zeros((V, D, SMPL.SHAPE_SPACE_DIM - B))], axis=-1
            )
            self.gt_models[gender] = SMPLHLayer(
                str(bm_path), data_struct=data_struct, num_betas=10, use_pca=False, flat_hand_mean=True
            ).to(pl_module.device)
        self.pred_model = SMPLXLayer(
            str(self.body_model_root / "smplx" / "SMPLX_NEUTRAL.npz"),
            num_betas=10, use_pca=False, flat_hand_mean=True,
        ).to(pl_module.device)
        self.J_regressor = torch.load(
            "hmr4d/utils/body_model/smpl_neutral_J_regressor.pt", map_location="cpu", weights_only=True
        )[:22].to(pl_module.device)
        self.smplx2smpl = torch.load(
            "hmr4d/utils/body_model/smplx2smpl_sparse.pt", map_location="cpu", weights_only=True
        ).to(pl_module.device).to_dense()
        self.results = {}

    @torch.no_grad()
    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if batch["meta"][0]["dataset_id"] != "EgoBody-Occ":
            return
        assert batch["B"] == 1
        meta = batch["meta"][0]
        start, end = meta["eval_range"]
        assert end <= int(batch["length"][0])
        pred_params = outputs["pred_smpl_params_incam"]
        gt_params = batch["gt_params"]
        pred_joints, gt_joints = [], []
        # FK is frame-independent; chunking bounds memory without changing temporal inference.
        for offset in range(0, end - start, 128):
            stop = min(offset + 128, end - start)
            pred = {k: v[start + offset:start + stop] for k, v in pred_params.items()}
            gt = {k: v[0, offset:stop] for k, v in gt_params.items()}
            for params in [pred, gt]:
                params["global_orient"] = axis_angle_to_matrix(params["global_orient"])
                params["body_pose"] = axis_angle_to_matrix(params["body_pose"].reshape(-1, 21, 3))
            pred_verts_cam = self.smplx2smpl @ self.pred_model(**pred).vertices
            gt_verts_master = self.gt_models[batch["gender"][0]](**gt).vertices
            pred_joints_cam = self.J_regressor @ pred_verts_cam
            gt_joints_master = self.J_regressor @ gt_verts_master
            master2cam = batch["master2cam"][0]
            gt_joints_cam = gt_joints_master @ master2cam[:3, :3].T + master2cam[:3, 3]
            pred_joints.append(pred_joints_cam)
            gt_joints.append(gt_joints_cam)
        pred_joints = torch.cat(pred_joints)[None]
        gt_joints = torch.cat(gt_joints)[None]
        visibility = batch["visibility"]
        if pred_joints.shape != gt_joints.shape or visibility.shape != pred_joints.shape[:-1]:
            raise ValueError("EgoBody-Occ prediction/GT/visibility frame or joint mismatch")
        # A common camera->world rigid transform cancels in these pelvis-aligned Euclidean errors.
        pred_aligned, gt_aligned, _, _ = self.metrics.batch_align_by_pelvis(
            [pred_joints, gt_joints, pred_joints, gt_joints], pelvis_idxs=[1, 2]
        )
        metrics = {"mpjpe": self.metrics.compute_jpe(pred_aligned, gt_aligned) * 1000}
        metrics.update(self.metrics.compute_visibility_mpjpe(pred_joints, gt_joints, visibility))
        self.results[meta["vid"]] = metrics

    def on_test_epoch_end(self, trainer, pl_module):
        with torch.inference_mode(False):
            gathered = all_gather(self.results)
        if not trainer.is_global_zero:
            return
        results = {}
        for result in gathered:
            results.update(result)
        if not results:
            raise RuntimeError("No EgoBody-Occ results were evaluated")
        # MoRo pools frames for all-joint MPJPE and individual joints for vis/occ MPJPE.
        pooled = {key: np.concatenate([r[key].flatten() for r in results.values()])
                  for key in ["mpjpe", "visible_mpjpe", "occluded_mpjpe"]}
        summary = {key: float(value.mean()) for key, value in pooled.items() if key == "mpjpe" or value.size}
        self.output_dir.mkdir(parents=True, exist_ok=True)
        report = {
            "moro_commit": MORO_COMMIT, "prediction_stage": "incam_pre_pp_ik", "unit": "mm",
            "recordings": list(results), "counts": {k: int(v.size) for k, v in pooled.items()},
            "metrics": summary,
            "per_recording": {vid: {k: float(v.mean()) for k, v in r.items() if k == "mpjpe" or v.size}
                              for vid, r in results.items()},
        }
        with open(self.output_dir / "egobody_occ_metrics.json", "w") as f:
            json.dump(report, f, indent=2)
        Log.info(f"[Metrics] EgoBody-Occ local (mm): {summary}")
        self.results = {}


MainStore.store(
    name="metric_egobody_occ",
    node=builds(MetricEgoBodyOcc, output_dir="${output_dir}", populate_full_signature=True),
    group="callbacks", package="callbacks.metric_egobody_occ",
)
