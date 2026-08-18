import cv2
import numpy as np
import os

def create_simulated_video(image_path, output_path, duration_sec=6, fps=30):
    """
    Creates a simulated moving/panning camera video from a static image.
    This gives the tracking model some visual displacement to track flaws.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to read image: {image_path}")
        return False
        
    h, w, c = img.shape
    
    # Target square frame size
    crop_size = min(h, w, 512)
    
    # Setup Video Writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (crop_size, crop_size))
    
    total_frames = duration_sec * fps
    max_x_offset = w - crop_size
    max_y_offset = h - crop_size
    
    for i in range(total_frames):
        t = i / total_frames
        
        # Pan back and forth smoothly using cosine
        pan_factor = 0.5 - 0.5 * np.cos(t * 2 * np.pi)
        
        x_offset = int(pan_factor * max_x_offset) if max_x_offset > 0 else 0
        y_offset = int(pan_factor * max_y_offset) if max_y_offset > 0 else 0
        
        # Crop the sub-frame
        frame = img[y_offset:y_offset+crop_size, x_offset:x_offset+crop_size]
        
        # Add a subtle zoom/scale effect to simulate depth changes
        zoom_factor = 1.0 + 0.05 * np.sin(t * 4 * np.pi) # zoom between 1.0 and 1.05
        new_size = int(crop_size * zoom_factor)
        
        # Perform resize and re-crop to maintain dimensions
        resized = cv2.resize(frame, (new_size, new_size))
        diff = (new_size - crop_size) // 2
        final_frame = resized[diff:diff+crop_size, diff:diff+crop_size]
        
        out.write(final_frame)
        
    out.release()
    print(f"Simulated test video created: {output_path}")
    return True

if __name__ == "__main__":
    # Create the test_videos directory
    os.makedirs("test_videos", exist_ok=True)
    
    # Generate videos from sample images
    samples = ["sample1.png", "sample2.png", "sample3.png"]
    for idx, sample in enumerate(samples, 1):
        img_path = os.path.join("test_images", sample)
        if os.path.exists(img_path):
            out_path = os.path.join("test_videos", f"demo_defect_video_{idx}.mp4")
            create_simulated_video(img_path, out_path)
        else:
            print(f"Sample image not found: {img_path}")
