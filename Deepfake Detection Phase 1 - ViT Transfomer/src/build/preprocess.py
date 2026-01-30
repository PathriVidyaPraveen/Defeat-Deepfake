import os
import torch
import numpy as np
import cv2
import pywt
from tqdm import tqdm
import argparse
import torch.nn.functional as F
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
from supervision import Detection
from PIL import Image


model_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection",
                 filename="model.pt")

face_model = YOLO(model_path)


def yolo_crop(frame, target_size=(224, 224)):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)

    # YOLO inference
    output = face_model(pil_image)
    results = Detections.from_ultralytics(output[0])

    if len(results) > 0:
        # Extract results
        xyxy = results.xyxy[0]
        x1, y1, x2, y2 = map(int, xyxy)

        face_crop = frame[y1:y2, x1:x2]

        face_resized = cv2.resize(face_crop, target_size, interpolation=cv2.INTER_LINEAR)
        return face_resized
    else:
        print(f"YOLO crop failed")
        return center_crop(frame, target_size)
        






def frame_dwt(frame, wavelet='haar'):
    coeffs2 = pywt.dwt2(frame, wavelet)
    cA, (cH, cV, cD) = coeffs2
    
    dwt_stack = np.stack([cA, cH, cV], axis=0).astype(np.float32)
    return dwt_stack

def center_crop(img, target_size=(224, 224)):
    """
    Crops the center of the image/frame.
    img: Numpy array (H, W, C) or (H, W)
    """
    h, w = img.shape[:2]
    th, tw = target_size
    
    # If image is smaller than target, pad it instead of crashing
    if h < th or w < tw:
        # Calculate padding
        pad_h = max(0, th - h)
        pad_w = max(0, tw - w)
        # Pad evenly on sides
        img = np.pad(img, ((pad_h//2, pad_h - pad_h//2), (pad_w//2, pad_w - pad_w//2), (0,0)) if len(img.shape)==3 
                     else ((pad_h//2, pad_h - pad_h//2), (pad_w//2, pad_w - pad_w//2)), mode='constant')
        h, w = img.shape[:2] # Update new dims

    i = int(round((h - th) / 2.))
    j = int(round((w - tw) / 2.))
    
    if len(img.shape) == 3:
        return img[i:i+th, j:j+tw, :]
    else:
        return img[i:i+th, j:j+tw]

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
    target_indices = set(indices)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if current_frame in target_indices:
            # Center crop
            # Modify to use a YOLO model for cropping
            frame_cropped = yolo_crop(frame, target_size = (224, 224))

            # Convert to RGB
            rgb_frame = cv2.cvtColor(frame_cropped, cv2.COLOR_BGR2RGB)
            
            # Convert to tensor
            rgb_tensor = torch.from_numpy(rgb_frame).permute(2, 0, 1).float()

            # Process DWT
            gray_frame = cv2.cvtColor(frame_cropped, cv2.COLOR_BGR2GRAY)
            dwt_numpy = frame_dwt(gray_frame, wavelet)
            dwt_tensor = torch.from_numpy(dwt_numpy)

            # Upsample DWT to match RGB
            dwt_resized = F.interpolate(
                dwt_tensor.unsqueeze(0),
                size=(224, 224),
                mode='nearest'
            ).squeeze(0)

            # Combine 
            combined_frame = torch.cat([rgb_tensor, dwt_resized], dim=0)
            frames_list.append(combined_frame)
            
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

    categories = ['real_sequences', 'fake_sequences']
    
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