import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageChops
import os

def generate_ela_from_video(video_path, frame_number=15):
    """
    Extracts a frame from a video and generates an ELA visualization.
    """
    # 1. Capture the video and grab a specific frame
    cap = cv2.VideoCapture(video_path)
    # Check if video opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Jump to the specific frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Error: Could not read the frame.")
        return

    # 2. Convert to RGB (OpenCV uses BGR)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 3. ELA requires saving to disk to simulate 're-compression'
    # We save the extracted frame temporarily
    original_pil = Image.fromarray(frame_rgb)
    original_pil.save("temp_original.jpg", "JPEG", quality=100)
    original_pil.save("temp_resaved.jpg", "JPEG", quality=90) # Re-compress

    # 4. Load them back to compare
    img_original = Image.open("temp_original.jpg")
    img_resaved = Image.open("temp_resaved.jpg")

    # 5. Calculate the difference (The "Error")
    ela_image = ImageChops.difference(img_original, img_resaved)

    # 6. Enhance the difference (Make it visible)
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1
    scale = 255.0 / max_diff
    ela_enhanced = ImageChops.multiply(ela_image, scale)

    # Clean up temp files
    os.remove("temp_original.jpg")
    os.remove("temp_resaved.jpg")

    return frame_rgb, ela_enhanced

# --- Usage ---
video_file = "deployment/assets/fake_video.mp4"  # <--- REPLACE with your video path

try:
    original, forensic_view = generate_ela_from_video(video_file, frame_number=30)

    # Plotting for your slide
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Left: The normal frame
    axes[0].imshow(original)
    axes[0].set_title("Original Video Frame", fontsize=14)
    axes[0].axis('off')

    # Right: The Forensic View
    axes[1].imshow(forensic_view)
    axes[1].set_title("Error Level Analysis (Compression Artifacts)", fontsize=14)
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig("forensic_slide_image.png", dpi=300)
    plt.show()
    print("Image saved as 'forensic_slide_image.png'")

except Exception as e:
    print(f"An error occurred: {e}")