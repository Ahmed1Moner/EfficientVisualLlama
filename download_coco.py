import json
import os
import numpy as np
import requests
from tqdm import tqdm
import shutil
from PIL import Image
import psutil

def check_disk_space(path, required_space_mb):
    """Check if there is enough disk space in MB."""
    disk = psutil.disk_usage(path)
    free_space_mb = disk.free / (1024 * 1024)
    if free_space_mb < required_space_mb:
        raise OSError(f"Not enough disk space. Required: {required_space_mb} MB, Available: {free_space_mb:.2f} MB")
    return free_space_mb

def verify_image(image_path):
    """Verify if an image file is valid."""
    try:
        with Image.open(image_path) as img_pil:
            img_pil.verify()
        return True
    except Exception:
        return False

def create_coco_subsets(annotation_path, output_base_dir, percentages=[0.001], num_seeds=4, num_images_base=123287, batch_size=50):
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Check disk space (~10 GB for annotations, validation, and subsets)
    check_disk_space(output_base_dir, 10000)
    
    # Load full annotations
    if not os.path.exists(annotation_path):
        raise FileNotFoundError(f"Annotation file {annotation_path} not found.")
    with open(annotation_path, 'r') as f:
        annotations = json.load(f)
    
    images = annotations.get('images', [])
    all_annotations = annotations.get('annotations', [])
    
    if not images or not all_annotations:
        raise ValueError("Annotations file is missing 'images' or 'annotations' keys.")
    
    for percentage in percentages:
        num_images = max(1, int(num_images_base * percentage))  # e.g., 123 for 0.1%
        for seed in range(num_seeds):
            # Set random seed for reproducibility
            np.random.seed(seed)
            
            # Create subset directory structure
            subset_dir = os.path.join(output_base_dir, f"coco_{percentage:.1f}percent_seed{seed}")
            image_subset_dir = os.path.join(subset_dir, "train2017")
            annotation_subset_dir = os.path.join(subset_dir, "annotations")
            annotation_output_path = os.path.join(annotation_subset_dir, f"captions_train2017_{percentage:.1f}percent_seed{seed}.json")
            
            # Check if subset already exists
            if os.path.exists(annotation_output_path):
                try:
                    with open(annotation_output_path, 'r') as f:
                        subset_data = json.load(f)
                    if subset_data.get('images') and subset_data.get('annotations'):
                        print(f"Subset {subset_dir} already exists with valid annotations, skipping creation.")
                        continue
                except Exception:
                    print(f"Invalid annotation file at {annotation_output_path}, recreating subset.")
            
            # Select images randomly
            indices = np.random.choice(len(images), num_images, replace=False)
            selected_images = [images[i] for i in indices]
            selected_image_ids = set(img['id'] for img in selected_images)
            
            # Filter annotations for selected images
            selected_annotations = [ann for ann in all_annotations if ann['image_id'] in selected_image_ids]
            
            # Create directories
            os.makedirs(image_subset_dir, exist_ok=True)
            os.makedirs(annotation_subset_dir, exist_ok=True)
            
            # Save subset annotations JSON
            subset_data = {
                'images': selected_images,
                'annotations': selected_annotations,
                'info': annotations.get('info', {}),
                'licenses': annotations.get('licenses', []),
                'categories': annotations.get('categories', [])
            }
            with open(annotation_output_path, 'w') as f:
                json.dump(subset_data, f, indent=2)
            
            # Download images directly, skipping existing valid images
            base_url = "http://images.cocodataset.org/train2017/"
            dummy_count = 0
            images_to_download = []
            for img in selected_images:
                dst_path = os.path.join(image_subset_dir, img['file_name'])
                if os.path.exists(dst_path) and verify_image(dst_path):
                    print(f"Image {img['file_name']} already exists and is valid, skipping.")
                    continue
                images_to_download.append(img)
            
            for i in range(0, len(images_to_download), batch_size):
                batch_images = images_to_download[i:i + batch_size]
                for img in tqdm(batch_images, desc=f"Downloading images for {percentage:.1f}% seed {seed} (batch {i//batch_size + 1})"):
                    dst_path = os.path.join(image_subset_dir, img['file_name'])
                    img_url = base_url + img['file_name']
                    try:
                        response = requests.get(img_url, stream=True)
                        if response.status_code == 200:
                            with open(dst_path, 'wb') as f:
                                for chunk in response.iter_content(1024):
                                    f.write(chunk)
                            # Verify downloaded image
                            if not verify_image(dst_path):
                                print(f"Invalid image downloaded: {img['file_name']}, creating dummy.")
                                Image.new('RGB', (224, 224), color='gray').save(dst_path)
                                dummy_count += 1
                        else:
                            print(f"Failed to download {img['file_name']}: Status {response.status_code}")
                            Image.new('RGB', (224, 224), color='gray').save(dst_path)
                            dummy_count += 1
                    except Exception as e:
                        print(f"Error downloading {img['file_name']}: {e}")
                        Image.new('RGB', (224, 224), color='gray').save(dst_path)
                        dummy_count += 1
            
            print(f"Created subset: {subset_dir} with {len(selected_images)} images and {len(selected_annotations)} annotations")
            if dummy_count > 0:
                print(f"Warning: {dummy_count} dummy images created for {percentage:.1f}% seed {seed}")

def download_coco(data_dir, annotation_url, val_image_url):
    os.makedirs(data_dir, exist_ok=True)
    
    # Check disk space for annotations and validation (~7 GB)
    check_disk_space(data_dir, 7000)
    
    # Download annotations if not exist
    annotation_zip = os.path.join(data_dir, "annotations_trainval2017.zip")
    annotation_dir = os.path.join(data_dir, "annotations")
    if not os.path.exists(os.path.join(annotation_dir, "captions_train2017.json")):
        print("Downloading annotations...")
        response = requests.get(annotation_url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        with open(annotation_zip, 'wb') as f, tqdm(total=total_size, unit='B', unit_scale=True, desc="Annotations") as pbar:
            for chunk in response.iter_content(1024):
                f.write(chunk)
                pbar.update(len(chunk))
        shutil.unpack_archive(annotation_zip, data_dir)
        os.remove(annotation_zip)
        print(f"Deleted {annotation_zip} to save space")
    else:
        print(f"Annotations already exist in {annotation_dir}, skipping download.")
    
    # Download validation images if not exist
    val_image_zip = os.path.join(data_dir, "val2017.zip")
    val_image_dir = os.path.join(data_dir, "val2017")
    if not os.path.exists(val_image_dir):
        print("Downloading validation images...")
        response = requests.get(val_image_url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        with open(val_image_zip, 'wb') as f, tqdm(total=total_size, unit='B', unit_scale=True, desc="Val Images") as pbar:
            for chunk in response.iter_content(1024):
                f.write(chunk)
                pbar.update(len(chunk))
        shutil.unpack_archive(val_image_zip, data_dir)
        os.remove(val_image_zip)
        print(f"Deleted {val_image_zip} to save space")
    else:
        print(f"Validation images already exist in {val_image_dir}, skipping download.")

def clear_cache(cache_dir="/home/yazan/VisualLlama/cache"):
    """Clear temporary cache directory."""
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir, ignore_errors=True)
        print(f"Cleared cache directory: {cache_dir}")

def main():
    data_dir = "/home/yazan/VisualLlama/coco"
    output_base_dir = "/home/yazan/VisualLlama/coco_subsets"
    annotation_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    val_image_url = "http://images.cocodataset.org/zips/val2017.zip"
    
    # Step 1: Clear any existing cache
    clear_cache(os.path.join(data_dir, "cache"))
    
    # Step 2: Download annotations and validation images
    download_coco(data_dir, annotation_url, val_image_url)
    
    # Step 3: Create only 0.1% subsets with 4 seeds
    annotation_path = os.path.join(data_dir, "annotations/captions_train2017.json")
    create_coco_subsets(annotation_path, output_base_dir, percentages=[0.001], num_seeds=4, batch_size=50)
    
    print(f"0.1% subsets with 4 seeds created in {output_base_dir}")

if __name__ == "__main__":
    main()
