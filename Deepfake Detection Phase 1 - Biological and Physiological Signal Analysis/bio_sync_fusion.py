# BioSyncFusion Deepfake Detection System — Fully corrected single-file script
# Minimal edits: KLT tracking + geometry-check; reuse landmarks in explainability; removed polarity flip; HRV guarded.
# Requirements: face-alignment, torch, numpy, opencv-python-headless, pywavelets, scipy, scikit-learn, matplotlib, tqdm

import sys
import subprocess
import os
import math
from typing import List, Tuple, Optional
import glob
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# ---------------------------
# Install dependencies if missing (attempt)
# ---------------------------
def pip_install(pkgs):
    import importlib
    for pkg in pkgs:
        root = pkg.split("==")[0].split(">=")[0]
        try:
            importlib.import_module(root)
        except Exception:
            print(f"Installing {pkg} ...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

pip_install([
    "opencv-python-headless>=4.7.0.72",
    "face-alignment>=1.3.5",
    "torch",
    "pywavelets>=1.5.1",
    "scipy>=1.9.0",
    "numpy>=1.24.0",
    "matplotlib>=3.6.0",
    "tqdm>=4.65.0",
    "scikit-learn>=1.2.0"
])

# ---------------------------
# Imports
# ---------------------------
import cv2
import numpy as np
import pywt
from scipy import signal
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from tqdm import tqdm

# face-alignment import
import torch
import face_alignment

print("Libraries loaded.")
print("Torch device:", "cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# CONFIGURATION & HYPERPARAMETERS
# ==========================================
class BioConfig:
    """
    Research-grade hyperparameters (tune as needed).
    """
    # Sampling / HR band defaults (will be updated per-video)
    FS = 30.0
    MIN_BPM = 45.0
    MAX_BPM = 180.0

    # Wavelet parameters
    WAVELET_NAME = 'cmor6-1.0'
    SCALES = np.arange(1, 128)

    # EVM params
    EVM_ALPHA_COLOR = 50
    EVM_ALPHA_MOTION = 15
    EVM_LEVELS = 4
    EVM_LOW_CUTOFF = 0.8
    EVM_HIGH_CUTOFF = 2.5

    # KLT params
    MAX_FEATURES = 120
    KLT_QUALITY = 0.01
    KLT_MIN_DIST = 7
    KLT_BLOCK_SIZE = 7

    # HR bounds (Hz)
    MIN_HZ = 0.75
    MAX_HZ = 4.0

    # Fusion threshold (tune)
    FUSION_THRESHOLD = 0.25

    # Tracking / detection
    DETECTION_INTERVAL = 15   # run FAN every 15 frames (tune)
    TRACKING_FAILURE_RATIO = 0.5  # if more than this fraction of points lost -> re-detect

    # Geometry drift threshold
    GEOMETRY_THRESHOLD = 0.20  # 20% change triggers re-detection
    # Affine canonical frame size
    AFFINE_W = 256
    AFFINE_H = 256


# ==========================================
# Utility functions
# ==========================================
def reflect_pad_if_odd(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    pad_bottom = 1 if (h % 2 == 1) else 0
    pad_right = 1 if (w % 2 == 1) else 0
    if pad_bottom == 0 and pad_right == 0:
        return img
    return cv2.copyMakeBorder(img, 0, pad_bottom, 0, pad_right, cv2.BORDER_REFLECT)


def crop_to_shape(img: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    return img[:target_h, :target_w].copy()


# ==========================================
# MODULE 1: Robust Laplacian Pyramid EVM
# ==========================================
class LaplacianEVM:
    def build_gaussian_pyramid(self, img: np.ndarray, levels: int) -> List[np.ndarray]:
        cur = img.astype(np.float32)
        gp = [cur]
        for _ in range(levels):
            cur = reflect_pad_if_odd(cur)
            cur = cv2.pyrDown(cur)
            gp.append(cur)
        return gp

    def build_laplacian_pyramid(self, img: np.ndarray, levels: int) -> List[np.ndarray]:
        gaussian_pyr = self.build_gaussian_pyramid(img, levels)
        laplacian_pyr = []
        for i in range(levels):
            current = gaussian_pyr[i]
            nxt = gaussian_pyr[i + 1]
            up = cv2.pyrUp(nxt)
            th, tw = current.shape[:2]
            if up.shape[0] != th or up.shape[1] != tw:
                up = crop_to_shape(up, th, tw)
            laplacian = current - up
            laplacian_pyr.append(laplacian)
        laplacian_pyr.append(gaussian_pyr[-1])
        return laplacian_pyr

    def reconstruct_from_laplacian(self, pyramid: List[np.ndarray]) -> np.ndarray:
        recon = pyramid[-1]
        for i in range(len(pyramid) - 2, -1, -1):
            up = cv2.pyrUp(recon)
            th, tw = pyramid[i].shape[:2]
            if up.shape[0] != th or up.shape[1] != tw:
                up = crop_to_shape(up, th, tw)
            recon = pyramid[i] + up
        return recon

    def magnify_video(self, frames_buffer: List[np.ndarray], fs: float, mode='color') -> List[np.ndarray]:
        n = len(frames_buffer)
        if n == 0:
            return []
        levels = BioConfig.EVM_LEVELS
        pyramids = [self.build_laplacian_pyramid(f.astype(np.float32), levels) for f in frames_buffer]

        low = BioConfig.EVM_LOW_CUTOFF / (0.5 * fs)
        high = BioConfig.EVM_HIGH_CUTOFF / (0.5 * fs)
        low = max(low, 1e-6)
        high = min(high, 0.999)
        b, a = signal.butter(2, [low, high], btype='bandpass')

        magnify_levels = [levels - 1, levels - 2]
        alpha = BioConfig.EVM_ALPHA_COLOR if mode == 'color' else BioConfig.EVM_ALPHA_MOTION

        filtered_layers = []
        for l in range(levels + 1):
            layer_stack = np.stack([p[l] for p in pyramids], axis=0).astype(np.float32)
            if l in magnify_levels:
                try:
                    filtered = signal.filtfilt(b, a, layer_stack, axis=0, padlen=3 * (max(len(b), len(a))))
                except Exception:
                    filtered = np.zeros_like(layer_stack)
                    for h in range(layer_stack.shape[1]):
                        for w in range(layer_stack.shape[2]):
                            for c in range(layer_stack.shape[3]):
                                ts = layer_stack[:, h, w, c]
                                filtered[:, h, w, c] = signal.lfilter(b, a, ts)
                layer_stack = layer_stack + alpha * filtered
            filtered_layers.append(layer_stack)

        magnified = []
        for t in range(n):
            mod_pyr = [filtered_layers[l][t] for l in range(levels + 1)]
            recon = self.reconstruct_from_laplacian(mod_pyr)
            recon = np.clip(recon, 0, 255).astype(np.uint8)
            magnified.append(recon)
        return magnified

# ==========================================
# MODULE 2: Biological Signal Processor (face-alignment 2D + POS + KLT PCA BCG)
# ==========================================
class BioSignalProcessor:
    def __init__(self, device: Optional[str] = None):
        # Device selection
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        # Use 2D landmarks for stable tracking
        self.fa = face_alignment.FaceAlignment(
            face_alignment.LandmarksType.TWO_D,
            device=self.device,
            flip_input=False
        )

        # --- NEW BUFFERS FOR ROI SIGNALS ---
        self.buffer_forehead = []
        self.buffer_cheek_l = []
        self.buffer_cheek_r = []

        # Tracking state
        self.prev_landmarks_pts = None   # np.ndarray shape (N,1,2)
        self.prev_gray = None
        self.tracking_active = False
        self.last_detection_frame_idx = -999

        # reference geometry distances
        self.ref_eye_dist = None
        self.ref_nose_dist = None
        self.lm_history_queue = [] # Simple moving average buffer
        self.SMOOTHING_WINDOW = 5

    # ===============================
    #       LANDMARK DETECTION
    # ===============================
    def detect_landmarks(self, image: np.ndarray):
        """
        Returns (68,2) landmarks or None
        """
        if image is None:
            return None
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        try:
            preds = self.fa.get_landmarks(rgb)
        except Exception as e:
            #print("face_alignment error:", e)
            preds = None
        if preds is None or len(preds) == 0:
            return None
        return preds[0][:, :2]  # (68,2)

    # ===============================
    #       KLT tracking helper
    # ===============================
    def track_landmarks_klt(self, prev_gray, gray, prev_pts):
        """
        prev_pts: Nx1x2 float32
        returns new_pts Nx1x2 or None and status
        """
        if prev_pts is None or len(prev_pts.reshape(-1, 2)) == 0:
            return None, None
        lk_params = dict(winSize=(21, 21), maxLevel=3,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None, **lk_params)
        return next_pts, status

    # ===============================
    #      GEOMETRY CHECK (NEW)
    # ===============================
    def geometry_changed(self, pts):
        """
        pts: Nx2
        return True if geometry changed more than threshold
        """
        if pts is None or len(pts.shape) != 2 or pts.shape[0] < 46:
            return True  # be conservative: if shapes unexpected, flag re-detect

        # eye corners indices (left eye outer corner 36, right eye outer corner 45)
        eye_L = pts[36]
        eye_R = pts[45]
        eye_dist = np.linalg.norm(eye_L - eye_R) + 1e-8

        # approximate nose bridge to tip (27 -> 33)
        nb = pts[27]
        nt = pts[33]
        nose_dist = np.linalg.norm(nb - nt) + 1e-8

        if self.ref_eye_dist is None or self.ref_nose_dist is None:
            return False

        if abs(eye_dist / self.ref_eye_dist - 1.0) > BioConfig.GEOMETRY_THRESHOLD:
            return True
        if abs(nose_dist / self.ref_nose_dist - 1.0) > BioConfig.GEOMETRY_THRESHOLD:
            return True
        return False

    # ===============================
    # Hybrid get landmarks: detection + KLT (with geometry check)
    # ===============================
    def get_landmarks_2d(self, image: np.ndarray, frame_idx: int, detect_if_needed=True):
        """
        Hybrid: use tracking when possible. Run detection every DETECTION_INTERVAL frames or if tracking fails.
        Returns landmarks (68,2) or None. Updates internal tracking buffers and ref geometry.
        """
        if image is None:
            return None

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Try tracking
        if self.prev_landmarks_pts is not None and self.prev_gray is not None and self.tracking_active:
            next_pts, status = self.track_landmarks_klt(self.prev_gray, gray, self.prev_landmarks_pts)
            if next_pts is not None and status is not None:
                status = status.reshape(-1)
                valid_ratio = np.sum(status == 1) / float(len(status))
                pts = next_pts.reshape(-1, 2)
                # geometry check
                if valid_ratio > (1.0 - BioConfig.TRACKING_FAILURE_RATIO) and not self.geometry_changed(pts):
                    self.prev_landmarks_pts = next_pts
                    self.prev_gray = gray
                    return pts
                else:
                    # tracking degraded or geometry drift -> force re-detect
                    self.prev_landmarks_pts = None
                    self.tracking_active = False

        # Decide whether to detect
        do_detect = detect_if_needed and ((frame_idx - self.last_detection_frame_idx) >= BioConfig.DETECTION_INTERVAL or self.prev_landmarks_pts is None)
        if do_detect:
            lm = self.detect_landmarks(image)
            if lm is None:
                return None
            # update reference geometry
            try:
                self.ref_eye_dist = np.linalg.norm(lm[36] - lm[45]) + 1e-8
                self.ref_nose_dist = np.linalg.norm(lm[27] - lm[33]) + 1e-8
            except Exception:
                self.ref_eye_dist = None
                self.ref_nose_dist = None

            pts = lm.astype(np.float32).reshape(-1, 1, 2)
            self.prev_landmarks_pts = pts
            self.prev_gray = gray
            self.tracking_active = True
            self.last_detection_frame_idx = frame_idx
            return lm
        else:
            return None

    def align_face_affine(self, frame, landmarks):
        """
        Stabilized Affine Alignment.
        Averages landmarks over SMOOTHING_WINDOW frames to prevent
        alignment jitter from destroying the BCG signal.
        """
        if landmarks is None:
            return frame, None
        landmarks = landmarks.astype(np.float32)
        # 1. Smooth the landmarks used for alignment (Indices 36, 45, 33)
        # We only need to smooth the anchor points, but smoothing all is easier
        self.lm_history_queue.append(landmarks)
        if len(self.lm_history_queue) > self.SMOOTHING_WINDOW:
            self.lm_history_queue.pop(0)
        
        # Average the landmarks over time
        avg_lm = np.mean(np.array(self.lm_history_queue), axis=0)

        dst_w, dst_h = BioConfig.AFFINE_W, BioConfig.AFFINE_H

        # Source points from SMOOTHED landmarks
        src = np.float32([
            avg_lm[36], # L Eye
            avg_lm[45], # R Eye
            avg_lm[33]  # Nose
        ])

        # Destination points (Canonical)
        dst = np.float32([
            [0.30 * dst_w, 0.40 * dst_h],
            [0.70 * dst_w, 0.40 * dst_h],
            [0.50 * dst_w, 0.65 * dst_h],
        ])

        # Affine transform
        M = cv2.getAffineTransform(src, dst)
        aligned = cv2.warpAffine(frame, M, (dst_w, dst_h))

        # Transform the CURRENT landmarks (not smoothed) using the smoothed Matrix
        # We want to know where the actual features are in the new frame
        ones = np.ones((landmarks.shape[0], 1))
        pts_h = np.hstack([landmarks, ones])
        aligned_lm = (M @ pts_h.T).T

        return aligned, aligned_lm

    # ===============================
    #      FOREHEAD & CHEEK ROI POLYGONS
    # ===============================
    def build_forehead_polygon(self, landmarks):
        brow_idxs = list(range(17, 27))
        brow_pts = landmarks[brow_idxs, :2]

        center = np.mean(brow_pts, axis=0)
        nose_tip = landmarks[30, :2]
        chin = landmarks[8, :2]

        face_h = np.linalg.norm(chin - nose_tip)
        shift = int(max(10, 0.25 * face_h))

        direction = center - nose_tip
        direction = direction / (np.linalg.norm(direction) + 1e-8)
        shift_vec = direction * shift

        forehead_pts = [(int(p[0] + shift_vec[0]), int(p[1] + shift_vec[1])) for p in brow_pts]
        base = [(int(x), int(y)) for x, y in brow_pts[::-1]]
        return forehead_pts + base

    def build_cheek_polygon(self, landmarks, side='left'):
        if side == 'left':
            idxs = [1, 2, 3, 31, 48, 33]
        else:
            idxs = [15, 14, 13, 35, 54, 33]
        pts = landmarks[idxs, :2]
        return [(int(x), int(y)) for x, y in pts]

    # ===============================
    # ROI mean color
    # ===============================
    def mask_from_polygon(self, frame_shape, poly):
        h, w = frame_shape
        mask = np.zeros((h, w), dtype=np.uint8)
        if len(poly) < 3:
            return mask
        pts = np.array(poly, dtype=np.int32)
        cv2.fillConvexPoly(mask, pts, 255)
        mask = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=1)
        return mask

    def get_roi_average(self, frame, poly):
        h, w = frame.shape[:2]
        if poly is None or len(poly) < 3:
            return np.zeros(3, dtype=np.float32)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, np.array(poly, dtype=np.int32), 255)
        return np.array(cv2.mean(frame, mask=mask)[:3], dtype=np.float32)

    # ===============================
    # POS rPPG
    # ===============================
    def pos_algorithm(self, rgb_signals, fs):
        if len(rgb_signals) < 3:
            return np.zeros(len(rgb_signals))

        rgb = rgb_signals[:, ::-1]
        mean_rgb = np.mean(rgb, axis=0)
        norm_rgb = rgb / (mean_rgb + 1e-8)

        S1 = norm_rgb[:, 1] - norm_rgb[:, 2]
        S2 = norm_rgb[:, 1] + norm_rgb[:, 2] - 2 * norm_rgb[:, 0]
        alpha = (np.std(S1) + 1e-8) / (np.std(S2) + 1e-8)
        P = S1 + alpha * S2

        nyq = 0.5 * fs
        low = BioConfig.MIN_HZ / nyq
        high = min(BioConfig.MAX_HZ / nyq, 0.999)

        b, a = signal.butter(3, [low, high], btype='bandpass')
        try:
            filt = signal.filtfilt(b, a, P)
        except:
            filt = signal.lfilter(b, a, P)
        return filt

    # ===============================
    # KLT + PCA BCG motion (with trajectory smoothing)
    # ===============================
    def extract_motion_pulse(self, frames: List[np.ndarray], landmarks_list: List[Optional[np.ndarray]]) -> np.ndarray:
        T = len(frames)
        if T < 10:
            return np.zeros(T, dtype=np.float32)

        # first valid landmarks
        first_lm = next((lm for lm in landmarks_list if lm is not None), None)
        if first_lm is None:
            return np.zeros(T, dtype=np.float32)

        fh_poly = self.build_forehead_polygon(first_lm)
        lc_poly = self.build_cheek_polygon(first_lm, 'left')
        rc_poly = self.build_cheek_polygon(first_lm, 'right')

        h, w = frames[0].shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        for poly in [fh_poly, lc_poly, rc_poly]:
            if len(poly) >= 3:
                cv2.fillConvexPoly(mask, np.array(poly, dtype=np.int32), 255)
        mask = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=1)
        mask_u = (mask > 0).astype(np.uint8)

        # Build corner features on gradient magnitude (more illumination-invariant)
        gray0 = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        gx = cv2.Sobel(gray0, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray0, cv2.CV_64F, 0, 1, ksize=3)
        gmag0 = cv2.magnitude(gx, gy).astype(np.uint8)

        corners = cv2.goodFeaturesToTrack(gmag0,
                                          maxCorners=BioConfig.MAX_FEATURES,
                                          qualityLevel=BioConfig.KLT_QUALITY,
                                          minDistance=BioConfig.KLT_MIN_DIST,
                                          blockSize=BioConfig.KLT_BLOCK_SIZE,
                                          mask=mask_u)
        if corners is None:
            ys, xs = np.where(mask_u > 0)
            if len(xs) == 0:
                return np.zeros(T, dtype=np.float32)
            pts = np.array(list(zip(xs, ys)), dtype=np.float32).reshape(-1, 1, 2)
            corners = pts[:min(BioConfig.MAX_FEATURES, len(pts))]

        lk_params = dict(winSize=(21, 21), maxLevel=3,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

        prev_pts = corners
        prev_gray = gray0
        trajectories = [prev_pts.reshape(-1, 2)]

        for i in range(1, T):
            gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)

            # small temporal high-pass on intensity to reduce slow illumination
            blur = cv2.GaussianBlur(gray, (31, 31), 0)
            gray_hp = cv2.subtract(gray, blur)

            next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None, **lk_params)

            if next_pts is None:
                trajectories.append(prev_pts.reshape(-1, 2))
            else:
                status = status.reshape(-1)
                pts_all = np.full((prev_pts.shape[0], 2), np.nan, dtype=np.float32)
                pts_all[status == 1] = next_pts[status == 1].reshape(-1, 2)
                trajectories.append(pts_all)

            prev_pts = next_pts if next_pts is not None else prev_pts
            prev_gray = gray

        traj = np.stack(trajectories, axis=0)  # (T, N, 2)

        # fill NaNs by interpolation
        for p in range(traj.shape[1]):
            for d in range(2):
                col = traj[:, p, d]
                if np.all(np.isnan(col)):
                    traj[:, p, d] = 0.0
                    continue
                mask_nan = np.isnan(col)
                if np.any(mask_nan):
                    xp = np.flatnonzero(~mask_nan)
                    fp = col[~mask_nan]
                    col[mask_nan] = np.interp(np.flatnonzero(mask_nan), xp, fp)
                    traj[:, p, d] = col

        # center trajectories (subtract per-frame mean to remove global shifts)
        traj = traj - np.mean(traj, axis=1, keepdims=True)

        # ----- SMOOTH TRAJECTORIES BEFORE PCA (REDUCES DETECTOR JITTER) -----
        try:
            for p in range(traj.shape[1]):
                for d in range(2):
                    # apply Savgol with window 7 (must be odd and <= T)
                    w = min(7, traj.shape[0] if traj.shape[0] % 2 == 1 else traj.shape[0] - 1)
                    if w >= 5:
                        traj[:, p, d] = savgol_filter(traj[:, p, d], w, 3)
        except Exception:
            pass

        T_len, N_pts, _ = traj.shape
        data_for_pca = traj.reshape(T_len, N_pts * 2)

        # use only first 3 components to avoid noise capture
        pca = PCA(n_components=min(3, data_for_pca.shape[1], data_for_pca.shape[0]))
        try:
            comps = pca.fit_transform(data_for_pca)
        except Exception:
            data_for_pca = data_for_pca[:, :min(50, data_for_pca.shape[1])]
            pca = PCA(n_components=min(3, data_for_pca.shape[1], data_for_pca.shape[0]))
            comps = pca.fit_transform(data_for_pca)

        best_sig = np.zeros(T_len, dtype=np.float32)
        max_energy = -1.0

        nyq = 0.5 * BioConfig.FS
        low = BioConfig.MIN_HZ / nyq
        high = min(BioConfig.MAX_HZ / nyq, 0.999)
        b, a = signal.butter(3, [low, high], btype='bandpass')

        for i in range(comps.shape[1]):
            sig = comps[:, i]
            try:
                filt = signal.filtfilt(b, a, sig)
            except Exception:
                filt = signal.lfilter(b, a, sig)
            energy = np.sum(filt ** 2)
            if energy > max_energy:
                max_energy = energy
                best_sig = filt

        if np.std(best_sig) > 1e-8:
            best_sig = (best_sig - np.mean(best_sig)) / (np.std(best_sig) + 1e-8)
        return best_sig.astype(np.float32)

    # ===============================
    #           PROCESS VIDEO
    # ===============================
    def process_video(self, video_path, max_frames=300, resize_width=None):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or math.isnan(fps):
            fps = BioConfig.FS
        BioConfig.FS = fps

        # --- CLEAR ROI BUFFERS FOR NEW VIDEO ---
        self.buffer_forehead = []
        self.buffer_cheek_l = []
        self.buffer_cheek_r = []

        raw_rppg_data = []
        frames_buffer = []
        landmarks_history = []

        frame_idx = 0
        pbar = tqdm(total=max_frames, desc="Extracting Frames & Landmarks")

        # Needed for stability when detection fails
        last_aligned_frame = None
        last_aligned_lm = None

        while cap.isOpened() and frame_idx < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if resize_width is not None:
                h, w = frame.shape[:2]
                scale = resize_width / float(w)
                frame = cv2.resize(frame, (resize_width, int(h * scale)))

            lm = self.get_landmarks_2d(frame, frame_idx, detect_if_needed=True)

            if lm is not None:
                # ---- AFFINE ALIGNMENT ----
                frame_aligned, lm_aligned = self.align_face_affine(frame, lm)
                last_aligned_frame = frame_aligned.copy()
                last_aligned_lm = lm_aligned.copy()

                # ROI values
                fh_poly = self.build_forehead_polygon(lm_aligned)
                lc_poly = self.build_cheek_polygon(lm_aligned, 'left')
                rc_poly = self.build_cheek_polygon(lm_aligned, 'right')

                val_fh = self.get_roi_average(frame_aligned, fh_poly)
                val_lc = self.get_roi_average(frame_aligned, lc_poly)
                val_rc = self.get_roi_average(frame_aligned, rc_poly)

                # Store ROI time-series values (**FIX**)
                self.buffer_forehead.append(val_fh[1])  # green channel
                self.buffer_cheek_l.append(val_lc[1])
                self.buffer_cheek_r.append(val_rc[1])

                avg = (val_fh + val_lc + val_rc) / 3.0

                raw_rppg_data.append(avg)
                frames_buffer.append(frame_aligned.copy())
                landmarks_history.append(lm_aligned)

            else:
                # ---- FIX: Reuse last aligned frame to avoid mixed coordinate spaces ---
                if last_aligned_frame is not None:
                    frames_buffer.append(last_aligned_frame.copy())
                    landmarks_history.append(last_aligned_lm.copy() if last_aligned_lm is not None else None)

                    # Keep ROI values stable
                    if last_aligned_lm is not None:
                        fh_poly = self.build_forehead_polygon(last_aligned_lm)
                        lc_poly = self.build_cheek_polygon(last_aligned_lm, 'left')
                        rc_poly = self.build_cheek_polygon(last_aligned_lm, 'right')

                        val_fh = self.get_roi_average(last_aligned_frame, fh_poly)
                        val_lc = self.get_roi_average(last_aligned_frame, lc_poly)
                        val_rc = self.get_roi_average(last_aligned_frame, rc_poly)

                        self.buffer_forehead.append(val_fh[1])
                        self.buffer_cheek_l.append(val_lc[1])
                        self.buffer_cheek_r.append(val_rc[1])

                        avg = (val_fh + val_lc + val_rc) / 3.0
                        raw_rppg_data.append(avg)
                else:
                    # First frames with no landmarks
                    frames_buffer.append(frame)
                    landmarks_history.append(None)

            frame_idx += 1
            pbar.update(1)


        cap.release()
        pbar.close()

        if len(raw_rppg_data) == 0:
            print("Error: No face landmarks detected.")
            return None, None, None, None, fps

        raw_rppg = np.array(raw_rppg_data)
        rppg_sig = self.pos_algorithm(raw_rppg, fps)
        motion_sig = self.extract_motion_pulse(frames_buffer, landmarks_history)

        min_len = min(len(rppg_sig), len(motion_sig), len(frames_buffer))
        return rppg_sig[:min_len], motion_sig[:min_len], frames_buffer[:min_len], landmarks_history[:min_len], fps

# ==========================================
# MODULE 3: Wavelet Analyzer
# ==========================================
class BioAnalyzer:
    def compute_wavelet_power(self, sig: np.ndarray, fs: float):
        if len(sig) < 3:
            return np.zeros((1, len(sig))), np.array([0.0])
        sig = (sig - np.mean(sig)) / (np.std(sig) + 1e-8)
        scales = BioConfig.SCALES
        coeffs, freqs = pywt.cwt(sig, scales, BioConfig.WAVELET_NAME, sampling_period=1.0 / fs)
        power = np.abs(coeffs) ** 2
        return power, freqs

    def extract_instantaneous_hr(self, power: np.ndarray, freqs: np.ndarray, fs: float):
        if power.size == 0:
            return np.zeros(power.shape[1] if power.ndim > 1 else 0)
        idxs = np.argmax(power, axis=0)
        hr = freqs[idxs]
        return hr

    def dominant_freq_from_power(self, power: np.ndarray, freqs: np.ndarray):
        # returns dominant frequency over time as a single scalar (median of per-frame peaks)
        if power.size == 0:
            return 0.0
        idxs = np.argmax(power, axis=0)
        dom = freqs[idxs]
        return float(np.median(dom))

# ==========================================
# BioSyncFusion: orchestration, fusion, visualization
# ==========================================
class BioSyncFusion:
    def __init__(self, device: Optional[str] = None):
        self.processor = BioSignalProcessor(device=device)
        self.analyzer = BioAnalyzer()
        self.evm = LaplacianEVM()

    def compute_phase_coherence(self, sig1, sig2, fs):
        if len(sig1) < 3 or len(sig2) < 3:
            return 0.0, np.array([]), np.array([])
        f, Cxy = signal.coherence(sig1, sig2, fs=fs, nperseg=min(256, len(sig1)))
        band_idx = np.where((f >= BioConfig.MIN_HZ) & (f <= BioConfig.MAX_HZ))[0]
        if band_idx.size == 0:
            return 0.0, f, Cxy
        return float(np.mean(Cxy[band_idx])), f, Cxy

    def compute_correlation_and_lag(self, sig1, sig2, fs):
        if len(sig1) < 3 or len(sig2) < 3:
            return 0.0, 0.0
        s1 = (sig1 - np.mean(sig1)) / (np.std(sig1) + 1e-8)
        s2 = (sig2 - np.mean(sig2)) / (np.std(sig2) + 1e-8)
        corrcoef = float(np.corrcoef(s1, s2)[0, 1])
        xc = signal.correlate(s1, s2, mode='full')
        lags = signal.correlation_lags(len(s1), len(s2), mode='full')
        lag_idx = np.argmax(xc)
        lag_samples = lags[lag_idx]
        return corrcoef, float(lag_samples / fs)

    def multi_roi_consistency(self, roi_signals: dict):
        keys = list(roi_signals.keys())
        cors = []
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                a = np.array(roi_signals[keys[i]])
                b = np.array(roi_signals[keys[j]])
                m = min(len(a), len(b))
                if m < 10:
                    continue
                c = np.corrcoef(a[:m], b[:m])[0, 1]
                cors.append(c)
        return float(np.mean(cors)) if len(cors) > 0 else 0.0

    def compute_hrv(self, rppg_sig, fps):
        """
        Returns SDNN in milliseconds (standard deviation of RR intervals).
        Only computed for sufficiently long clips (>= 60s). Otherwise returns 0.
        """
        if len(rppg_sig) < int(60 * fps):
            return 0.0  # avoid noisy HRV on short clips

        # Bandpass rPPG to HR band first (0.75-4.0 Hz)
        nyq = 0.5 * fps
        low = max(0.75 / nyq, 1e-6)
        high = min(4.0 / nyq, 0.999)
        try:
            b, a = signal.butter(3, [low, high], btype='bandpass')
            filt = signal.filtfilt(b, a, rppg_sig)
        except Exception:
            filt = signal.lfilter(*signal.butter(3, [low, high], btype='bandpass'), rppg_sig)

        # Smooth a bit
        try:
            filt_s = signal.savgol_filter(filt, min(len(filt)-1 if len(filt)%2==0 else len(filt), 31), 3)
        except Exception:
            filt_s = filt

        # Peak detection (min distance = fastest HR -> 180 bpm)
        min_dist = int(max(1, fps * 60.0 / 180.0))
        peaks, _ = signal.find_peaks(filt_s, distance=min_dist, prominence=np.std(filt_s)*0.3)

        if len(peaks) < 3:
            return 0.0

        rr = np.diff(peaks) / float(fps)  # seconds
        if len(rr) < 2:
            return 0.0

        sdnn = np.std(rr) * 1000.0  # ms
        return float(sdnn)

    def sliding_corr(self, a, b, window=90):
        m = min(len(a), len(b))
        a, b = a[:m], b[:m]
        if m < window + 5:
            return 0.0

        local_corrs = []
        for i in range(m - window):
            c = np.corrcoef(a[i:i+window], b[i:i+window])[0, 1]
            local_corrs.append(c)

        return float(np.mean(local_corrs)) if len(local_corrs) > 0 else 0.0

    def true_coherence(self, a, b, fps):
        f, Cxy = signal.coherence(a, b, fs=fps, nperseg=128)
        idx = np.where((f > 0.75) & (f < 4.0))
        if len(idx[0]) == 0:
            return 0.0
        return float(np.mean(Cxy[idx]))

    def run_analysis(self, video_path: str, model=None, label: str = "Unknown", max_frames: int = 300, resize_width: int = 480):
        print(f"\n>>> Starting Bio-Sync-Fusion Analysis on: {label}")
        rppg, motion, frames, landmarks_history, fps = self.processor.process_video(video_path, max_frames=max_frames, resize_width=resize_width)
        if rppg is None:
            print("Extraction failed.")
            return

        # --- Wavelet analysis ---
        power_rppg, freqs_rppg = self.analyzer.compute_wavelet_power(rppg, fps)
        power_motion, freqs_motion = self.analyzer.compute_wavelet_power(motion, fps)
        hr_rppg = self.analyzer.extract_instantaneous_hr(power_rppg, freqs_rppg, fs=fps)
        hr_motion = self.analyzer.extract_instantaneous_hr(power_motion, freqs_motion, fs=fps)

        # --- Features ---
        corr_coef_raw, lag_seconds = self.compute_correlation_and_lag(rppg, motion, fps)
        abs_corr = abs(corr_coef_raw)
        phase_coh, f_coh, Cxy = self.compute_phase_coherence(rppg, motion, fps)
        roi_consistency = self.multi_roi_consistency({
            'forehead': self.processor.buffer_forehead,
            'cheek_l': self.processor.buffer_cheek_l,
            'cheek_r': self.processor.buffer_cheek_r
        })
        hrv = self.compute_hrv(rppg, fps)
        local_corr = self.sliding_corr(rppg, motion)
        coh_true = self.true_coherence(rppg, motion, fps)

        hr_band_idx = np.where((freqs_rppg >= BioConfig.MIN_HZ) & (freqs_rppg <= BioConfig.MAX_HZ))[0]
        if hr_band_idx.size > 0:
            energy_hr = np.sum(power_rppg[hr_band_idx, :])
            energy_total = np.sum(power_rppg) + 1e-8
            snr_rppg = float(energy_hr / energy_total)
        else:
            snr_rppg = 0.0

        dom_freq_rppg = self.analyzer.dominant_freq_from_power(power_rppg, freqs_rppg)
        dom_freq_motion = self.analyzer.dominant_freq_from_power(power_motion, freqs_motion)
        freq_diff = abs(dom_freq_rppg - dom_freq_motion)
        
        hrv_norm = np.clip(1.0 - min(hrv, 120)/120.0, 0.0, 1.0) if hrv > 0 else 0.0

        # ==========================================================
        # FUSION LOGIC (TRAINED vs HARDCODED)
        # ==========================================================
        if model is not None:
            # --- USE TRAINED MODEL ---
            # Must match order in extract_features: 
            # [abs_corr, local_corr, roi_consistency, coh_true, snr_rppg, freq_diff, hrv_norm]
            feats_vec = np.array([abs_corr, local_corr, roi_consistency, coh_true, snr_rppg, freq_diff, hrv_norm]).reshape(1, -1)
            feats_vec = np.nan_to_num(feats_vec)
            
            # Predict
            prob_real = model.predict_proba(feats_vec)[0][1]
            fusion_score = prob_real
            
            # Threshold (Logistic Regression standard is 0.5)
            is_fake = fusion_score < 0.5 
            verdict = "FAKE" if is_fake else "REAL"
            
            print(f"--- MODEL INFERENCE ---")
            print(f"Features: Corr={abs_corr:.2f}, FreqDiff={freq_diff:.2f}, Coh={coh_true:.2f}")
            print(f"Model Probability (REAL): {fusion_score:.3f}")
            
        else:
            # --- FALLBACK: HARDCODED LOGIC (Legacy) ---
            is_phase_locked = (abs_corr > 0.25) and (abs(lag_seconds) < 0.25)
            f_corr = np.clip(abs_corr, 0.0, 1.0)
            f_local = np.clip(local_corr, -1.0, 1.0)
            f_roi = np.clip(roi_consistency, -1.0, 1.0)
            f_coh = np.clip(coh_true, 0.0, 1.0)
            f_freq = np.clip(1.0 - (freq_diff / 0.5), 0.0, 1.0)

            if is_phase_locked:
                fusion_score = (0.40 * f_corr + 0.20 * f_freq + 0.20 * max(0.0, f_local) + 0.10 * f_coh + 0.10 * f_roi)
                fusion_score = min(fusion_score + 0.15, 1.0)
            else:
                fusion_score = (0.25 * f_corr + 0.40 * f_freq + 0.15 * max(0.0, f_local) + 0.10 * f_coh + 0.10 * f_roi)
                fusion_score = max(fusion_score - 0.1, 0.0)
            is_fake = fusion_score < 0.4
            verdict = "FAKE" if is_fake else "REAL"
            print(f"--- HARDCODED INFERENCE ---")
            print(f"Fusion score: {fusion_score:.3f}")

        print(f"Verdict: {verdict}")

        # --- Plotting ---
        fig, axs = plt.subplots(4, 1, figsize=(12, 14))
        t_axis = np.arange(len(rppg)) / fps
        axs[0].plot(t_axis, (rppg - np.mean(rppg)) / (np.std(rppg) + 1e-8), label='rPPG (POS)')
        axs[0].plot(t_axis, (motion - np.mean(motion)) / (np.std(motion) + 1e-8), label='BCG (PCA)')
        axs[0].set_title(f"{label}: Signals (Corr={corr_coef_raw:.2f}, Score={fusion_score:.2f})")
        axs[0].legend()

        ax1 = axs[1]
        im1 = ax1.imshow(power_rppg, aspect='auto', origin='lower', extent=[0, len(rppg)/fps, freqs_rppg[0], freqs_rppg[-1]])
        ax1.set_title("rPPG Wavelet Power")
        fig.colorbar(im1, ax=ax1, orientation='horizontal')

        ax2 = axs[2]
        im2 = ax2.imshow(power_motion, aspect='auto', origin='lower', extent=[0, len(motion)/fps, freqs_motion[0], freqs_motion[-1]])
        ax2.set_title("BCG Wavelet Power")
        fig.colorbar(im2, ax=ax2, orientation='horizontal')

        axs[3].plot(t_axis, hr_rppg, label='HR (rPPG)')
        axs[3].plot(t_axis, hr_motion, label='HR (BCG)')
        axs[3].set_title("Instantaneous HR (Hz)")
        axs[3].legend()
        plt.tight_layout()
        plt.show()

        # --- Explainability Video ---
        print("Generating forensic explainability video...")
        explain_frames = []
        snippet_len = min(150, len(frames))
        n_inst = min(power_rppg.shape[1], power_motion.shape[1]) if power_rppg.size and power_motion.size else 0
        inst_freq_diff = np.zeros(n_inst)
        if n_inst > 0:
            idxs_r = np.argmax(power_rppg[:, :n_inst], axis=0)
            idxs_m = np.argmax(power_motion[:, :n_inst], axis=0)
            fr_r = freqs_rppg[idxs_r]
            fr_m = freqs_motion[idxs_m]
            inst_freq_diff = np.abs(fr_r - fr_m)
            inst_heat = np.clip(inst_freq_diff / 0.4, 0.0, 1.0)
        else:
            inst_heat = np.zeros(snippet_len)

        for i in range(snippet_len):
            frame = frames[i].copy()
            lm = landmarks_history[i] if i < len(landmarks_history) else None
            if lm is not None:
                fh_poly = self.processor.build_forehead_polygon(lm)
                lc_poly = self.processor.build_cheek_polygon(lm, 'left')
                rc_poly = self.processor.build_cheek_polygon(lm, 'right')
                cv2.polylines(frame, [np.array(fh_poly)], True, (0,255,0), 2)
                cv2.polylines(frame, [np.array(lc_poly)], True, (0,255,0), 2)
                cv2.polylines(frame, [np.array(rc_poly)], True, (0,255,0), 2)

            heat = float(inst_heat[min(i, len(inst_heat)-1)]) if len(inst_heat) > 0 else 0.0
            color = (0, int((1-heat)*255), int(heat*255))
            cv2.rectangle(frame, (0,0), (frame.shape[1], 45), color, -1) 
            tag = "REAL" if fusion_score >= 0.5 else "FAKE"
            cv2.putText(frame, f"{tag} ({fusion_score:.2f})", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"dF: {freq_diff:.2f}Hz", (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1, cv2.LINE_AA)
            explain_frames.append(frame)

        out_name = f"explained_{label}.mp4"
        self.save_video(explain_frames, out_name, fps)
        print(f"Saved explainability video to: {out_name}")
    def save_video(self, frames: List[np.ndarray], path: str, fps: float):
        if not frames:
            return
        H, W = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(path, fourcc, fps, (W, H))
        for f in frames:
            if f.shape[:2] != (H, W):
                f = cv2.resize(f, (W, H))
            if f.dtype != np.uint8:
                f = np.clip(f, 0, 255).astype(np.uint8)
            out.write(f)
        out.release()

    # ===============================
    # NEW METHOD: PURE FEATURE EXTRACTION
    # ===============================
    def extract_features(self, video_path: str, max_frames: int = 300, resize_width: int = 480):
        """
        Returns a numpy array of features for training:
        [abs_corr, local_corr, roi_consistency, coh_true, snr_rppg, freq_diff, hrv_norm]
        """
        # 1. Process Video
        rppg, motion, frames, _, fps = self.processor.process_video(video_path, max_frames=max_frames, resize_width=resize_width)
        if rppg is None or len(rppg) < 10:
            return None # Extraction failed

        # 2. Wavelet & HR
        power_rppg, freqs_rppg = self.analyzer.compute_wavelet_power(rppg, fps)
        power_motion, freqs_motion = self.analyzer.compute_wavelet_power(motion, fps)
        
        # 3. Compute Metrics
        corr_coef_raw, lag_seconds = self.compute_correlation_and_lag(rppg, motion, fps)
        abs_corr = abs(corr_coef_raw)
        
        local_corr = self.sliding_corr(rppg, motion)
        
        roi_consistency = self.multi_roi_consistency({
            'forehead': self.processor.buffer_forehead,
            'cheek_l': self.processor.buffer_cheek_l,
            'cheek_r': self.processor.buffer_cheek_r
        })
        
        coh_true = self.true_coherence(rppg, motion, fps)
        
        # SNR
        hr_band_idx = np.where((freqs_rppg >= BioConfig.MIN_HZ) & (freqs_rppg <= BioConfig.MAX_HZ))[0]
        if hr_band_idx.size > 0:
            energy_hr = np.sum(power_rppg[hr_band_idx, :])
            energy_total = np.sum(power_rppg) + 1e-8
            snr_rppg = float(energy_hr / energy_total)
        else:
            snr_rppg = 0.0

        # Frequency Difference
        dom_freq_rppg = self.analyzer.dominant_freq_from_power(power_rppg, freqs_rppg)
        dom_freq_motion = self.analyzer.dominant_freq_from_power(power_motion, freqs_motion)
        freq_diff = abs(dom_freq_rppg - dom_freq_motion)

        # HRV (Normalized 0-1 for stability)
        hrv = self.compute_hrv(rppg, fps)
        hrv_norm = np.clip(1.0 - min(hrv, 120)/120.0, 0.0, 1.0) if hrv > 0 else 0.0

        # Feature Vector
        # We handle NaN just in case
        features = [abs_corr, local_corr, roi_consistency, coh_true, snr_rppg, freq_diff, hrv_norm]
        features = [0.0 if np.isnan(x) else x for x in features]
        
        return np.array(features, dtype=np.float32)

# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":
    bsf = BioSyncFusion()
    
    # ---------------------------------------------------------
    # CONFIGURATION AREA
    # ---------------------------------------------------------
    TRAIN_MODE = True  # <--- SET TO TRUE TO TRAIN, FALSE TO TEST
    MODEL_PATH = "biosync_model.pkl"
    
    # Dataset Directories (Update these paths)
    REAL_DIR = "dataset/real_sequences" 
    FAKE_DIR = "dataset/fake_sequences"
    
    # Test Video (for inference mode)
    TEST_VIDEO = "test_video.mp4" 
    # ---------------------------------------------------------

    if TRAIN_MODE:
        print("\n>>> STARTING TRAINING PHASE...")
        X = []
        y = []
        
        # 1. Load Real (Label = 1)
        real_paths = glob.glob(os.path.join(REAL_DIR, "*.mp4"))
        print(f"Found {len(real_paths)} Real videos.")
        for p in tqdm(real_paths, desc="Processing Real"):
            feats = bsf.extract_features(p)
            if feats is not None:
                X.append(feats)
                y.append(1)
        
        # 2. Load Fake (Label = 0)
        fake_paths = glob.glob(os.path.join(FAKE_DIR, "*.mp4"))
        print(f"Found {len(fake_paths)} Fake videos.")
        for p in tqdm(fake_paths, desc="Processing Fake"):
            feats = bsf.extract_features(p)
            if feats is not None:
                X.append(feats)
                y.append(0)

        X = np.array(X)
        y = np.array(y)
        
        if len(X) == 0:
            print("No valid data extracted. Check paths.")
            sys.exit()

        print(f"Total samples: {len(X)}")

        # 3. Train
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        print(f"Training Logistic Regression on {len(X_train)} samples...")
        clf = LogisticRegression(class_weight='balanced', max_iter=1000)
        clf.fit(X_train, y_train)
        
        # 4. Evaluate
        preds = clf.predict(X_test)
        print("\nTest Set Report:")
        print(classification_report(y_test, preds, target_names=['FAKE', 'REAL']))
        
        # 5. Save
        joblib.dump(clf, MODEL_PATH)
        print(f"Model saved to {MODEL_PATH}")
        
        # Show weights
        feats_names = ["AbsCorr", "LocalCorr", "ROI_Cons", "CohTrue", "SNR", "FreqDiff", "HRV"]
        print("\n--- Learned Weights ---")
        for n, w in zip(feats_names, clf.coef_[0]):
            print(f"{n}: {w:.4f}")
        print(f"Bias: {clf.intercept_[0]:.4f}")

    else:
        # --- INFERENCE MODE ---
        print("\n>>> STARTING INFERENCE PHASE...")
        if not os.path.exists(MODEL_PATH):
            print("Model file not found! Please run with TRAIN_MODE = True first.")
        else:
            model = joblib.load(MODEL_PATH)
            print(f"Loaded model from {MODEL_PATH}")
            
            if os.path.exists(TEST_VIDEO):
                # Call run_analysis to get plots and video, passing the trained model
                bsf.run_analysis(TEST_VIDEO, model=model, label="TEST_INPUT", max_frames=300)
            else:
                print(f"Test video not found: {TEST_VIDEO}")
            
