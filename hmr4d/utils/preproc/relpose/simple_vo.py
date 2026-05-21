import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from .utils import focal_length_from_mm
from .matcher_wrapper import Matcher
from .solver_two_view import TwoPairSolver, CameraParams, interpolate_missing_frames
from tqdm import tqdm

from hmr4d.utils.video_io_utils import get_video_lwh, read_video_np


class SimpleVO:
    def __init__(self, video_path, scale=0.5, step=8, method="sift", f_mm=None, num_workers=1):
        self.video_path = video_path
        self.scale = scale
        self.step = step
        self.method = method
        self.f_mm = 24 if f_mm is None else f_mm  # fullframe camera focal length in mm
        self.num_workers = num_workers

    def compute(self):
        # Read video
        frames = read_video_np(self.video_path, scale=self.scale)

        # Downsample frames, and interpolate missing frames
        F_all = frames.shape[0]
        sample_idxs = np.arange(0, F_all, self.step)
        if sample_idxs[-1] != F_all - 1:
            sample_idxs = np.concatenate([sample_idxs, [F_all - 1]])
        frames = frames[sample_idxs]
        F, H, W, C = frames.shape
        print(f"[SimpleVO] Choosen frames shape: {frames.shape}")

        matcher: Matcher = Matcher(self.method)
        camera_params = CameraParams(W, H, focal_length=focal_length_from_mm(W, H, self.f_mm))
        solver: TwoPairSolver = TwoPairSolver(camera_params, solver="pycolmap")

        # TODO:We should use different pipelines for different methods
        T_w2c_list = self.process_video_T_w2c_list_np(frames, matcher, solver)

        # Interpolate missing frames
        T_w2c_list = interpolate_missing_frames(T_w2c_list, sample_idxs)

        return T_w2c_list

    def process_video_T_w2c_list_np(self, frames, matcher: Matcher, solver: TwoPairSolver):
        n_pairs = len(frames) - 1
        if self.num_workers <= 1:
            # Serial path (unchanged behavior)
            T_deltas = [None] * n_pairs
            for i in tqdm(range(n_pairs)):
                pts0, pts1 = matcher.match_np(frames[i], frames[i + 1])
                T_deltas[i] = solver.solve(pts0, pts1)
        else:
            # Parallel path: each thread owns its own Matcher/TwoPairSolver to avoid
            # sharing cv2.SIFT / pycolmap internals across threads.
            tls = threading.local()
            H, W = frames.shape[1:3]
            camera_params = CameraParams(W, H, focal_length=focal_length_from_mm(W, H, self.f_mm))

            def get_worker_state():
                if not hasattr(tls, "matcher"):
                    tls.matcher = Matcher(self.method)
                    tls.solver = TwoPairSolver(camera_params, solver="pycolmap")
                return tls.matcher, tls.solver

            def solve_pair(i):
                m, s = get_worker_state()
                pts0, pts1 = m.match_np(frames[i], frames[i + 1])
                return s.solve(pts0, pts1)

            T_deltas = [None] * n_pairs
            with ThreadPoolExecutor(max_workers=self.num_workers) as ex:
                for i, T in tqdm(
                    zip(range(n_pairs), ex.map(solve_pair, range(n_pairs))),
                    total=n_pairs,
                ):
                    T_deltas[i] = T

        # Serial accumulation (cheap, O(N) 4x4 matmuls)
        T_w2c_list = [np.eye(4)]
        for T_delta in T_deltas:
            T_w2c_list.append(T_delta @ T_w2c_list[-1])
        return T_w2c_list
