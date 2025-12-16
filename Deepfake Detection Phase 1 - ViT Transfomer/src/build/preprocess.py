import os
import torch
import numpy as np
import cv2
import pywt
from tqdm import tqdm
import argparse

def frame_dwt(frame, wavelet='haar'):
    coeffs2 = pywt.dwt2(frame, wavelet)
    cA, (cH, cV, cD) = coeffs2
    return cH, cV, cD

def process_video(video_path, num_frames=8, wavelet='haar'):
    """Reads video, extracts DWT, resizes, and returns a tensor."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    frames_list = []
    total_frames_in_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Smart sampling
    if total_frames_in_video >= num_frames:
        indices = np.linspace(0, total_frames_in_video - 1, num_frames, dtype=int)
    else:
        indices = np.arange(total_frames_in_video)
        
    current_frame = 0
    idx_pointer = 0
    target_indices = set(indices)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if current_frame in target_indices:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cH, cV, cD = frame_dwt(gray_frame, wavelet)
            
            # Stack components (3 channels)
            combined = np.stack([cH, cV, cD], axis=0).astype(np.float32)
            tensor_frame = torch.from_numpy(combined)
            
            # Resize for the vit transformer
            tensor_frame = torch.nn.functional.interpolate(
                tensor_frame.unsqueeze(0), 
                size=(224, 224), 
                mode='bilinear', 
                align_corners=False
            ).squeeze(0)
            
            frames_list.append(tensor_frame)
            
        current_frame += 1
        if len(frames_list) >= len(indices):
            break

    cap.release()

    if not frames_list:
        return None

    # Handle padding if video was too short
    video_tensor = torch.stack(frames_list) # (T, 3, 224, 224)
    if video_tensor.shape[0] < num_frames:
        padding = num_frames - video_tensor.shape[0]
        last_frame = video_tensor[-1].unsqueeze(0).repeat(padding, 1, 1, 1)
        video_tensor = torch.cat([video_tensor, last_frame], dim=0)

    return video_tensor

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='Path to original Celeb_df')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save .pt files')
    args = parser.parse_args()

    categories = ['0_Celeb-real', '1_Celeb-synthesis']
    
    for category in categories:
        input_dir = os.path.join(args.data_path, category)
        videos = [f for f in os.listdir(input_dir) if f.endswith('.mp4')]
        
        # Shuffle and split into train/val/test
        videos = sorted(videos)
        np.random.seed(42) # Seed for reproducibility
        np.random.shuffle(videos)

        total_videos = len(videos)
        train_split = int(0.7 * total_videos)
        val_split = int(0.15 * total_videos)

        train_videos = videos[:train_split]
        val_videos = videos[train_split:train_split+val_split]
        test_videos = videos[train_split+val_split:]

        
       # Process and save to respective folders
        for split_name, split_videos in [('train', train_videos), ('val', val_videos), ('test', test_videos)]:
            output_dir = os.path.join(args.output_path, split_name, category)
            os.makedirs(output_dir, exist_ok=True)
            
            for video_file in split_videos:
                video_path = os.path.join(input_dir, video_file)
                save_path = os.path.join(output_dir, video_file.replace('.mp4', '.pt'))

                try:
                    tensor = process_video(video_path)
                    if tensor is not None:
                        torch.save(tensor, save_path)
                except Exception as e:
                    print(f"Error processing {video_path}: {e}")

if __name__ == "__main__":
    main()