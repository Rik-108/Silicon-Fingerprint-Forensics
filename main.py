# main.py

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import exposure
import prnu

# Import the comprehensive YOLO function from the other file
from yolo_analysis import run_yolo_analysis

# --- 1. SETUP PROJECT DIRECTORIES ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(BASE_DIR, 'images')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')

os.makedirs(OUTPUT_DIR, exist_ok=True)


# --- 2. ANALYSIS FUNCTIONS ---

def analyze_noise(image_sets, output_dir):
    """
    Calculates one PRNU fingerprint for each phone folder.
    Saves to section_ii_prnu/{phone_name}/
    Also runs identification on unknown images if present.
    """
    print("\n--- Section II: Calculating PRNU Fingerprints per Phone ---")
    prnu_base_dir = os.path.join(output_dir, 'section_ii_prnu')
    os.makedirs(prnu_base_dir, exist_ok=True)

    # Dictionary to store fingerprint paths
    fingerprint_paths = {}

    for phone_name, paths in image_sets.items():
        print(f"Processing fingerprint for '{phone_name}'...")
        # Find min dimensions
        min_h, min_w = float('inf'), float('inf')
        for img_path in paths:
            try:
                img_info = cv2.imread(img_path)
                if img_info is not None:
                    h, w, _ = img_info.shape
                    if h < min_h: min_h = h
                    if w < min_w: min_w = w
            except Exception as e:
                print(f"  Warning: Could not read shape of {img_path}. Error: {e}")

        if min_h == float('inf') or min_w == float('inf'):
            print(f"  Skipping '{phone_name}': Could not determine valid dimensions.")
            continue

        target_size = (min_w, min_h)
        loaded_images = [cv2.resize(cv2.imread(p), target_size, interpolation=cv2.INTER_AREA) for p in paths if
                         cv2.imread(p) is not None]

        if len(loaded_images) < 2:
            print(f"  Skipping '{phone_name}': Not enough valid images.")
            continue

        fingerprint = prnu.extract_multiple_aligned(loaded_images, processes=os.cpu_count())

        # Create phone-specific dir
        phone_dir = os.path.join(prnu_base_dir, phone_name)
        os.makedirs(phone_dir, exist_ok=True)

        npy_path = os.path.join(phone_dir, f"{phone_name}_prnu.npy")
        np.save(npy_path, fingerprint)
        fingerprint_paths[phone_name] = npy_path

        if fingerprint.ndim == 3:
            fingerprint_gray = cv2.cvtColor(fingerprint.astype(np.float32), cv2.COLOR_BGR2GRAY)
        else:
            fingerprint_gray = fingerprint

        fingerprint_visual = cv2.normalize(fingerprint_gray, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        png_path = os.path.join(phone_dir, f"{phone_name}_prnu_visualization.png")
        cv2.imwrite(png_path, fingerprint_visual)

        print(f"  Fingerprint saved to '{phone_dir}'.")

    print(f"✅ PRNU analysis complete. Results saved in '{prnu_base_dir}'.")

    # Identification System: "Who Took This?"
    print("\n--- Identification: Who Took This? ---")
    unknown_dir = os.path.join(os.path.dirname(output_dir), 'images', 'unknown')  # Assume images/unknown folder
    if not os.path.exists(unknown_dir):
        print(f"  No unknown images folder found at '{unknown_dir}'. Skipping identification.")
        return

    unknown_paths = [os.path.join(unknown_dir, f) for f in os.listdir(unknown_dir) if
                     f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not unknown_paths:
        print(f"  No unknown images found in '{unknown_dir}'. Skipping identification.")
        return

    PCE_THRESHOLD = 50  # Common threshold for PRNU match; adjust as needed

    results = []
    for unknown_path in unknown_paths:
        unknown_filename = os.path.basename(unknown_path)
        print(f"  Identifying source for '{unknown_filename}'...")

        unknown_img = cv2.imread(unknown_path)
        if unknown_img is None:
            print(f"    Could not load '{unknown_filename}'. Skipping.")
            continue

        # Extract noise from unknown (match target_size for consistency)
        loaded_unknown = [cv2.imread(unknown_path)]
        if len(loaded_unknown) < 1 or loaded_unknown[0] is None:
            continue
        unknown_noise = prnu.extract_multiple_aligned(loaded_unknown, processes=1)

        max_pce = -np.inf
        best_phone = None
        pce_scores = {}

        for phone_name, npy_path in fingerprint_paths.items():
            fingerprint = np.load(npy_path)

            # Resize unknown noise to match fingerprint dimensions
            fh, fw = fingerprint.shape[:2]
            unknown_resized_to_fp = cv2.resize(unknown_noise, (fw, fh), interpolation=cv2.INTER_AREA)

            # Compute cross-correlation
            cc2d = prnu.crosscorr_2d(fingerprint, unknown_resized_to_fp)

            # Compute PCE on cross-correlation
            pce_result = prnu.pce(cc2d)
            pce = pce_result['pce']
            pce_scores[phone_name] = pce
            if pce > max_pce:
                max_pce = pce
                best_phone = phone_name

        if max_pce > PCE_THRESHOLD:
            match = f"{best_phone} (PCE: {max_pce:.2f})"
        else:
            match = "No Match Found"

        results.append({
            'Image': unknown_filename,
            'Matched Phone': match,
            'All PCE Scores': str(pce_scores)
        })
        print(f"    Result: {match}")

    # Save results to .txt notebook file
    if results:
        import pandas as pd

        df = pd.DataFrame(results)

        txt_content = "# PRNU Identification Results\n\n"
        txt_content += "# Who Took This? Matching System\n\n"
        txt_content += "This report summarizes the camera identification results for the unknown images using PRNU fingerprints from the 5 trained phones.\n\n"
        txt_content += "Key Notes:\n"
        txt_content += "- 'Matched Phone' shows the most likely phone if PCE > 50; otherwise, 'No Match Found'.\n"
        txt_content += "- PCE (Peak-to-Correlation Energy) measures match strength—higher is better.\n"
        txt_content += "- Scores for all phones are listed for transparency.\n\n"

        txt_content += "=== RESULTS ===\n\n"
        for idx, row in df.iterrows():
            txt_content += f"Image: {row['Image']}\n"
            txt_content += f"Best Match: {row['Matched Phone']}\n"
            txt_content += "PCE Scores by Phone:\n"
            scores_str = row['All PCE Scores'].replace("'", "").replace("np.float32(", "").replace(")", "")
            for score in scores_str.split(', '):
                phone, val = score.split(': ')
                txt_content += f"  - {phone}: {val}\n"
            txt_content += "\n" + "="*50 + "\n\n"

        txt_path = os.path.join(prnu_base_dir, 'identification_results.txt')
        with open(txt_path, 'w') as f:
            f.write(txt_content)
        print(f"  Results saved to notebook: '{txt_path}'.")
    else:
        print("  No results to save.")


def analyze_hog(image_sets, output_dir):
    """
    Applies HOG to every image from every phone.
    Saves to section_iii_hog/{phone_name}/{filename}_hog.png
    """
    print("\n--- Section III: Analyzing HOG Features for All Images ---")
    hog_base_dir = os.path.join(output_dir, 'section_iii_hog')
    os.makedirs(hog_base_dir, exist_ok=True)

    for phone_name, paths in image_sets.items():
        if not paths:
            continue

        phone_dir = os.path.join(hog_base_dir, phone_name)
        os.makedirs(phone_dir, exist_ok=True)

        print(f"Processing HOG for '{phone_name}'...")

        for image_path in paths:
            filename = os.path.basename(image_path)
            print(f"  Processing '{filename}'...")
            image = cv2.imread(image_path)
            if image is None:
                continue

            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            fd, hog_image = hog(image_gray, orientations=8, pixels_per_cell=(16, 16),
                                cells_per_block=(1, 1), visualize=True)
            hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

            output_filename = f"{os.path.splitext(filename)[0]}_hog.png"
            output_path = os.path.join(phone_dir, output_filename)
            plt.imsave(output_path, hog_image_rescaled, cmap=plt.cm.gray)

        print(f"  HOG results for '{phone_name}' saved to '{phone_dir}'.")

    print(f"✅ HOG analysis complete. Results saved in '{hog_base_dir}'.")


def apply_noise_removal(image_sets, output_dir):
    """
    Applies Gaussian and Median filters to every image from every phone.
    Saves to section_iv_noise_removal/{phone_name}/{base}_gaussian.png etc.
    Returns original_sets and filtered_sets dicts for YOLO.
    """
    print("\n--- Section IV: Applying Noise Removal to All Images ---")
    noise_base_dir = os.path.join(output_dir, 'section_iv_noise_removal')
    os.makedirs(noise_base_dir, exist_ok=True)

    filtered_sets = {phone: [] for phone in image_sets}

    for phone_name, paths in image_sets.items():
        if not paths:
            continue

        phone_dir = os.path.join(noise_base_dir, phone_name)
        os.makedirs(phone_dir, exist_ok=True)

        print(f"Applying filters for '{phone_name}'...")

        for image_path in paths:
            filename = os.path.basename(image_path)
            print(f"  Applying filters to '{filename}'...")
            image = cv2.imread(image_path)
            if image is None:
                continue

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Apply Gaussian Smoothing
            gaussian_filtered = cv2.GaussianBlur(image_rgb, (5, 5), 0)

            # Apply Median Filter
            median_filtered = cv2.medianBlur(image_rgb, 5)

            # Save filtered images
            base, ext = os.path.splitext(filename)

            gauss_path = os.path.join(phone_dir, f"{base}_gaussian{ext}")
            median_path = os.path.join(phone_dir, f"{base}_median{ext}")

            cv2.imwrite(gauss_path, cv2.cvtColor(gaussian_filtered, cv2.COLOR_RGB2BGR))
            cv2.imwrite(median_path, cv2.cvtColor(median_filtered, cv2.COLOR_RGB2BGR))

            filtered_sets[phone_name].extend([gauss_path, median_path])

        print(f"  Filtered images for '{phone_name}' saved to '{phone_dir}'.")

    print(f"✅ Noise removal complete. Results saved in '{noise_base_dir}'.")
    return image_sets, filtered_sets  # original_sets is image_sets


# --- 3. MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    print("Starting Computer Vision Analysis Pipeline...")

    image_sets = {}
    # The keys will now be phone names, e.g., 'Phone_1_iPhone14'
    for phone_folder in os.listdir(IMAGE_DIR):
        if phone_folder == 'unknown':
            continue  # Skip unknown folder for analysis (used only in PRNU identification)
        phone_path = os.path.join(IMAGE_DIR, phone_folder)
        if os.path.isdir(phone_path):
            image_files = [os.path.join(phone_path, f) for f in os.listdir(phone_path)]
            image_sets[phone_folder] = image_files

    if not image_sets:
        print("\nERROR: No phone folders found in the 'images' directory.")
        print("Please create subdirectories like 'Phone_1_iPhone14' and add your 5 images.")
    else:
        # Run the standard analysis steps
        analyze_noise(image_sets, OUTPUT_DIR)
        analyze_hog(image_sets, OUTPUT_DIR)

        # This function now returns sets for the YOLO step
        original_sets, filtered_sets = apply_noise_removal(image_sets, OUTPUT_DIR)

        # Combine original and filtered per phone for YOLO
        all_yolo_sets = {}
        for phone in image_sets:
            all_yolo_sets[phone] = image_sets[phone] + filtered_sets.get(phone, [])

        # Run the comprehensive YOLO analysis on all original and filtered images, organized by phone
        run_yolo_analysis(all_yolo_sets, OUTPUT_DIR)

    print("\n--- Pipeline Finished ---")
    print(f"All outputs have been saved to the '{OUTPUT_DIR}' directory, organized by section and phone.")