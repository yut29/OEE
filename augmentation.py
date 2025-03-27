import os
import glob
import random
import shutil
import cv2
import numpy as np
from tqdm import tqdm

# Define paths
train_images_dir = r"E:\FAU\Stillstand\merged_dataset\images\train"
train_labels_dir = r"E:\FAU\Stillstand\merged_dataset\labels\train"

# Target class and number
TARGET_CLASS = 5  # Druckluftpistole (Air gun)
TARGET_COUNT = 150  # Target number of images

def check_label_for_class(label_file, target_class):
    """Check if the label file contains the target class"""
    with open(label_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if parts and int(parts[0]) == target_class:
                return True, lines
    return False, []

def find_samples_with_class(labels_dir, target_class):
    """Find all samples containing the specified class"""
    class_samples = []
    label_files = glob.glob(os.path.join(labels_dir, "*.txt"))
    
    for label_file in label_files:
        has_class, lines = check_label_for_class(label_file, target_class)
        if has_class:
            basename = os.path.basename(label_file)
            class_samples.append((basename, lines))
    
    return class_samples

def get_image_path(image_basename, images_dir):
    """Get image file path (supporting multiple formats)"""
    base_name = os.path.splitext(image_basename)[0]
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    
    for ext in image_extensions:
        img_path = os.path.join(images_dir, base_name + ext)
        if os.path.exists(img_path):
            return img_path, ext
    
    return None, None

def augment_image(image, label_lines, aug_type):
    """Augment images and labels"""
    height, width = image.shape[:2]
    new_image = image.copy()
    new_labels = label_lines.copy()
    
    if aug_type == "flip_horizontal":
        # Horizontal flip of the image
        new_image = cv2.flip(image, 1)  # 1 means horizontal flip
        
        # Update label coordinates
        for i in range(len(new_labels)):
            parts = new_labels[i].strip().split()
            if len(parts) >= 5:  # YOLO format: class x y w h
                class_id, x, y, w, h = parts[0], float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                # Flip x coordinate (x = 1 - x)
                new_x = 1 - x
                new_labels[i] = f"{class_id} {new_x} {y} {w} {h}\n"
    
    elif aug_type == "brightness":
        # Random brightness adjustment
        value = random.uniform(0.7, 1.3)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv = hsv.astype(np.float32)
        hsv[:,:,2] = hsv[:,:,2] * value
        hsv[:,:,2] = np.clip(hsv[:,:,2], 0, 255)
        hsv = hsv.astype(np.uint8)
        new_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    elif aug_type == "contrast":
        # Random contrast adjustment
        alpha = random.uniform(0.8, 1.2)
        beta = random.randint(-10, 10)
        new_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    
    elif aug_type == "rotate":
        # Small angle rotation (±15 degrees)
        angle = random.uniform(-15, 15)
        center = (width // 2, height // 2)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        new_image = cv2.warpAffine(image, rot_mat, (width, height), 
                                  flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        
        # Update label coordinates (simplified handling, only for small angle rotation)
        # For small angle rotations, we can approximately keep label coordinates unchanged
        # For large angle rotations, more complex coordinate transformation is needed
    
    elif aug_type == "noise":
        # Add Gaussian noise
        mean = 0
        sigma = random.uniform(5, 15)
        gauss = np.random.normal(mean, sigma, image.shape)
        gauss = gauss.astype(np.uint8)
        new_image = cv2.add(image, gauss)
    
    elif aug_type == "blur":
        # Slight blur
        kernel_size = random.choice([3, 5])
        new_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        
    return new_image, new_labels

def save_augmented_file(image, label_lines, original_basename, aug_type, index, images_dir, labels_dir, ext):
    """Save augmented image and label files"""
    # Create new filename
    base_name = os.path.splitext(original_basename)[0]
    new_basename = f"{base_name}_aug_{aug_type}_{index}"
    
    # Save image
    image_path = os.path.join(images_dir, new_basename + ext)
    cv2.imwrite(image_path, image)
    
    # Save label
    label_path = os.path.join(labels_dir, new_basename + ".txt")
    with open(label_path, 'w') as f:
        f.writelines(label_lines)
    
    return new_basename + ext, new_basename + ".txt"

def main():
    # Step 1: Find all samples containing class 5
    print(f"Finding samples containing class {TARGET_CLASS}...")
    class_samples = find_samples_with_class(train_labels_dir, TARGET_CLASS)
    original_count = len(class_samples)
    
    print(f"Found {original_count} samples containing class {TARGET_CLASS}")
    
    if original_count == 0:
        print(f"Error: No samples containing class {TARGET_CLASS} found!")
        return
    
    if original_count >= TARGET_COUNT:
        print(f"Original sample count ({original_count}) has already reached or exceeded the target count ({TARGET_COUNT}), no augmentation needed")
        return
    
    # Step 2: Calculate how many samples need to be augmented
    augment_count = TARGET_COUNT - original_count
    print(f"Need to augment {augment_count} samples to reach target count {TARGET_COUNT}")
    
    # Step 3: Create augmented samples
    aug_methods = ["flip_horizontal", "brightness", "contrast", "rotate", "noise", "blur"]
    created_samples = 0
    existing_filenames = set(os.listdir(train_images_dir))
    
    # Create progress bar
    with tqdm(total=augment_count) as pbar:
        while created_samples < augment_count:
            # Randomly select a sample
            sample_basename, label_lines = random.choice(class_samples)
            
            # Get corresponding image file
            image_path, ext = get_image_path(sample_basename, train_images_dir)
            if not image_path:
                print(f"Warning: Could not find image file for sample {sample_basename}, skipped")
                continue
            
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Warning: Could not read image {image_path}, skipped")
                continue
            
            # Randomly select an augmentation method
            aug_type = random.choice(aug_methods)
            
            # Perform augmentation
            try:
                aug_image, aug_labels = augment_image(image, label_lines, aug_type)
                
                # Save augmented files
                new_img_name, new_lbl_name = save_augmented_file(
                    aug_image, aug_labels, sample_basename, aug_type, 
                    created_samples, train_images_dir, train_labels_dir, ext
                )
                
                # Update counter
                created_samples += 1
                pbar.update(1)
                
                if created_samples % 10 == 0:
                    print(f"Created {created_samples}/{augment_count} augmented samples")
                
            except Exception as e:
                print(f"Error while augmenting sample {sample_basename}: {e}")
    
    print(f"\nData augmentation completed!")
    print(f"Original sample count: {original_count}")
    print(f"Newly added sample count: {created_samples}")
    print(f"Current total sample count for class {TARGET_CLASS}: {original_count + created_samples}")

if __name__ == "__main__":
    main()
