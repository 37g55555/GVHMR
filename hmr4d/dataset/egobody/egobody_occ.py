"""EgoBody-Occ protocol from MoRo@1f18c4f, with GVHMR observations.

Sources: eval_egobody.py; models/mask_transformer/data/egobody_dataset.py.
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from hmr4d.configs import MainStore, builds
from hmr4d.utils.geo_transform import compute_cam_angvel

MORO_COMMIT = "1f18c4f0d6db55d6b00327fb35c3e8d8d2fbfe82"


def read_occ_info(root, recording=None):
    df = pd.read_csv(Path(root) / "egobody_occ_info.csv")
    columns = ["recording_name", "view", "target_idx", "target_gender", "body_idx_fpv",
               "scene_name", "target_start_frame", "target_end_frame"]
    if df[columns].isna().any().any() or df["recording_name"].duplicated().any():
        raise ValueError("EgoBody-Occ requires one complete metadata row per recording")
    if recording is not None:
        df = df[df["recording_name"] == recording]
    if len(df) == 0:
        raise ValueError(f"No EgoBody-Occ recordings selected: {recording}")
    return df.to_dict("records")


def recording_paths(root, info):
    body_idx = int(info["target_idx"])
    interactee_idx = int(info["body_idx_fpv"].split(" ")[0])
    role = "smpl_interactee" if body_idx == interactee_idx else "smpl_camera_wearer"
    gt_root = Path(root) / role / info["recording_name"] / f"body_idx_{body_idx}" / "results"
    # Full recording context as in MoRo; only the CSV target interval is scored.
    frame_names = sorted(p.name for p in gt_root.iterdir() if p.is_dir())
    start = int(info["target_start_frame"])
    end = int(info["target_end_frame"])
    start_idx = frame_names.index(f"frame_{start:05d}")
    end_idx = frame_names.index(f"frame_{end:05d}") + 1
    if frame_names[start_idx:end_idx] != [f"frame_{i:05d}" for i in range(start, end + 1)]:
        raise ValueError(f"Non-contiguous target frames: {info['recording_name']}")
    return gt_root, frame_names, (start_idx, end_idx)


def load_camera(root, info):
    root = Path(root)
    calib_trans_dir = root / "calibrations" / info["recording_name"] / "cal_trans"
    with open(calib_trans_dir / "kinect12_to_world" / f"{info['scene_name']}.json") as f:
        master2world = np.asarray(json.load(f)["trans"])
    view = info["view"]
    if view == "master":
        cam2world = master2world
    else:
        camera_ids = {"sub_1": 11, "sub_2": 13, "sub_3": 14, "sub_4": 15}
        with open(calib_trans_dir / f"kinect_{camera_ids[view]}to12_color.json") as f:
            trans_subtomain = np.asarray(json.load(f)["trans"])
        cam2world = np.matmul(master2world, trans_subtomain)
    with open(root / "kinect_cam_params" / f"kinect_{view}" / "Color.json") as f:
        color_cam = json.load(f)
    cam2world = torch.from_numpy(cam2world).float()
    master2world = torch.from_numpy(master2world).float()
    master2cam = torch.linalg.solve(cam2world, master2world)
    return torch.tensor(color_cam["camera_mtx"]).float(), np.array(color_cam["k"]).astype(np.float32), master2cam


class EgoBodyOccDataset(Dataset):
    def __init__(self, root="inputs/EgoBody", recording=None):
        self.root = Path(root)
        self.recordings = read_occ_info(root, recording)

    def __len__(self):
        return len(self.recordings)

    def __getitem__(self, idx):
        info = self.recordings[idx]
        recording, view = info["recording_name"], info["view"]
        gt_root, frame_names, (start, end) = recording_paths(self.root, info)
        length = len(frame_names)
        support_path = self.root / "hmr4d_support" / recording / view / f"body_idx_{int(info['target_idx'])}.pt"
        obs = torch.load(support_path, map_location="cpu", weights_only=True)
        if obs["frame_names"] != frame_names or obs["moro_commit"] != MORO_COMMIT:
            raise ValueError(f"Stale frame/protocol metadata: {support_path}")
        K, _, master2cam = load_camera(self.root, info)
        if not torch.equal(obs["K_fullimg"], K):
            raise ValueError(f"Camera calibration changed: {support_path}")
        for key, shape in {"bbx_xys": (length, 3), "kp2d": (length, 17, 3),
                           "f_imgseq": (length, 1024)}.items():
            if obs[key].shape != shape or not torch.isfinite(obs[key]).all():
                raise ValueError(f"Invalid {key}: {support_path}")
        mask_joint = np.load(self.root / "mask_joint" / recording / view / "mask_joint.npy")
        # Released masks already cover the target interval, not the full recording.
        if mask_joint.ndim != 2 or mask_joint.shape[0] != end - start or mask_joint.shape[1] < 22:
            raise ValueError(f"Visibility mask must match target interval: {recording}")
        if not np.isin(mask_joint, [0, 1]).all():
            raise ValueError(f"Expected official binary visibility mask: {recording}")
        params = []
        for frame in frame_names[start:end]:
            with open(gt_root / frame / "000.pkl", "rb") as f:
                params.append(pickle.load(f))
        gt_params = {key: torch.from_numpy(np.concatenate([p[key] for p in params])).float()
                     for key in ["transl", "global_orient", "body_pose", "betas"]}
        gt_params["body_pose"] = gt_params["body_pose"][..., :63]
        return {
            "meta": {"dataset_id": "EgoBody-Occ", "vid": recording, "view": view,
                     "target_idx": int(info["target_idx"]), "eval_range": (start, end),
                     "frame_names": frame_names, "eval_postproc": False},
            "length": length,
            "bbx_xys": obs["bbx_xys"],
            "kp2d": obs["kp2d"],
            "f_imgseq": obs["f_imgseq"],
            "K_fullimg": K[None].repeat(length, 1, 1),
            "cam_angvel": compute_cam_angvel(torch.eye(3)[None].repeat(length, 1, 1)),
            "gt_params": gt_params,
            "gender": info["target_gender"],
            "master2cam": master2cam,
            "visibility": torch.from_numpy(mask_joint[:, :22]).bool(),
        }


MainStore.store(
    name="all", node=builds(EgoBodyOccDataset, populate_full_signature=True), group="test_datasets/egobody_occ"
)
