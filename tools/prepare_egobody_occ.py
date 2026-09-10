"""MoRo bbox protocol + GVHMR HMR2/ViTPose observation preparation.

process_egobody_bbox is transplanted from MoRo@1f18c4f (trailing whitespace removed).
No model is loaded in --bbox-only or --check-only mode.
"""

import argparse
import json
import os
from glob import glob
from pathlib import Path

import numpy as np

from hmr4d.dataset.egobody.egobody_occ import MORO_COMMIT, read_occ_info, recording_paths, load_camera


def process_egobody_bbox(kps_path):
    scale_factor_bbox = 1.2
    kps_files = sorted(glob(os.path.join(kps_path, "frame*.json")))
    for body_idx in [0, 1]:
        last_bbox = np.array([100., 100., 200., 200.], dtype=np.float32) # dummy bbox
        center_dict = {}
        scale_dict = {}
        bbox_dict = {}
        for kps_file in kps_files:
            frame_name = os.path.basename(kps_file)[:11]

            with open(kps_file, 'r') as f:
                kps_data = json.load(f)

            body_kps = np.array(kps_data["people"][int(body_idx)]["pose_keypoints_2d"]).astype(np.float32).reshape((-1, 3))
            valid = body_kps[:, 2] > 0.2
            valid_kps = body_kps[valid, :-1]

            if valid_kps.shape[0] < 2:
                bbox = last_bbox
            else:
                x0, y0 = np.min(valid_kps, axis=0)
                x1, y1 = np.max(valid_kps, axis=0)
                bbox = np.array([x0, y0, x1, y1])
            last_bbox = bbox
            center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
            scale = max(
                bbox[2] - bbox[0],
                bbox[3] - bbox[1],
            ) * scale_factor_bbox / 200
            center_dict[frame_name] = center
            scale_dict[frame_name] = scale
            bbox_dict[frame_name] = bbox

        # save the center and scale as numpy arrays
        frame_names = list(center_dict.keys())
        centers = np.array([center_dict[name] for name in frame_names])
        scales = np.array([scale_dict[name] for name in frame_names])
        bboxes = np.array([bbox_dict[name] for name in frame_names])

        save_path = os.path.join(kps_path, f"bbox_idx{body_idx}.npz")
        np.savez(save_path, frame_names=frame_names, centers=centers, scales=scales, bboxes=bboxes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--recording", default=None)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--bbox-only", action="store_true")
    group.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    recordings = read_occ_info(args.root, args.recording)
    if args.bbox_only:
        for info in recordings:
            kps_path = args.root / "keypoints_cleaned" / info["recording_name"] / info["view"]
            if not list(kps_path.glob("frame*.json")):
                raise FileNotFoundError(f"No official cleaned keypoints: {kps_path}")
            process_egobody_bbox(str(kps_path))
        return

    prepared = []
    for info in recordings:
        recording, view = info["recording_name"], info["view"]
        gt_root, frame_names, (start, end) = recording_paths(args.root, info)
        K, dist_coeffs, _ = load_camera(args.root, info)
        bbox_path = args.root / "keypoints_cleaned" / recording / view / f"bbox_idx{int(info['target_idx'])}.npz"
        bbox = np.load(bbox_path)
        centers = dict(zip(bbox["frame_names"], bbox["centers"]))
        scales = dict(zip(bbox["frame_names"], bbox["scales"]))
        bbx_xys = np.asarray([[*centers[name], scales[name] * 200] for name in frame_names], dtype=np.float32)
        if not np.isfinite(bbx_xys).all() or not (bbx_xys[:, 2] > 0).all():
            raise ValueError(f"Invalid official bbox: {bbox_path}")
        mask = np.load(args.root / "mask_joint" / recording / view / "mask_joint.npy")
        if mask.ndim != 2 or mask.shape[0] != end - start or mask.shape[1] < 22 or not np.isin(mask, [0, 1]).all():
            raise ValueError(f"Official target-interval visibility mismatch: {recording}")
        images = [args.root / "kinect_color" / recording / view / f"{name}.jpg" for name in frame_names]
        for path in images + [gt_root / name / "000.pkl" for name in frame_names[start:end]]:
            if not path.is_file():
                raise FileNotFoundError(path)
        output = args.root / "hmr4d_support" / recording / view / f"body_idx_{int(info['target_idx'])}.pt"
        print(f"{recording}/{view}: input={len(frame_names)}, evaluated={end-start}, output={output}")
        prepared.append((frame_names, images, bbx_xys, K, dist_coeffs, output))
    if args.check_only:
        return

    import cv2
    import torch
    from hmr4d.utils.preproc.vitfeat_extractor import Extractor, get_batch
    from hmr4d.utils.preproc.vitpose import VitPoseExtractor

    for *_, output in prepared:
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite prepared observations: {output}")
    extractor = Extractor()
    vitpose = VitPoseExtractor()
    for frame_names, images, bbx_xys, K, dist_coeffs, output in prepared:
        bbx_xys = torch.from_numpy(bbx_xys)
        features, keypoints, boxes = [], [], []
        for start in range(0, len(images), 16):
            rgb = []
            for path in images[start:start+16]:
                img = cv2.imread(str(path))
                if img is None:
                    raise ValueError(f"Unreadable RGB image: {path}")
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.undistort(img.copy(), K.numpy(), dist_coeffs)
                rgb.append(img)
            crops, crop_boxes = get_batch(np.stack(rgb), bbx_xys[start:start+16], img_ds=1.0, path_type="np")
            features.append(extractor.extract_video_features(crops, crop_boxes))
            keypoints.append(vitpose.extract(crops, crop_boxes))
            boxes.append(crop_boxes)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "moro_commit": MORO_COMMIT, "frame_names": frame_names, "K_fullimg": K,
            "bbx_xys": torch.cat(boxes).float(), "kp2d": torch.cat(keypoints).float(),
            "f_imgseq": torch.cat(features).float(),
        }, output)
        print(f"Saved {output}")


if __name__ == "__main__":
    main()
