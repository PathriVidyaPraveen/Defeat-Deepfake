# utils.py
import cv2
import pywt
import numpy as np
import torch

def frame_dwt(frame, wavelet='haar'):
    coeffs2 = pywt.dwt2(frame, wavelet)
    cA, (cH, cV, cD) = coeffs2
    return cA, cH, cV, cD

def video_dwt(video_path, wavelet='haar', sample_interval=16):
    """
    Extracts DWT components from video frames.
    sample_interval: Process every Nth frame to save time/memory.
    """
    cap = cv2.VideoCapture(video_path)
    dwt_frames = []
    
    if not cap.isOpened():
        return []

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Optimization: Only process the specific frames we need
        if frame_count % sample_interval == 0:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cA, cH, cV, cD = frame_dwt(gray_frame, wavelet)
            # Stack cA, cH, cV to mimic 3-channel image for ViT
            # Note: We ignore cD (Diagonal) to fit into 3 channels
            combined = np.stack([cA, cH, cV], axis=0).astype(np.float32)
            dwt_frames.append(combined)
            
        frame_count += 1

    cap.release()
    return dwt_frames

def apply_transforms_to_frames(frames, transform):
    """Apply same transform to all frames in batch"""
    if transform is None:
        return frames
    transformed_frames = []
    for frame in frames:
        transformed_frames.append(transform(frame))
    return torch.stack(transformed_frames, dim=0)