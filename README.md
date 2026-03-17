# The Silicon Fingerprint: Image Forensics & Multi-Task YOLOv11 👁️🔍

## 1. Project Overview
An advanced Computer Vision pipeline that operates on two levels: 
1. **Forensic Level:** Identifying the unique "Silicon Fingerprint" (PRNU) of a smartphone camera to verify image authenticity.
2. **Intelligence Level:** Deploying the latest **YOLOv11** architecture to perform five distinct computer vision tasks simultaneously.

## 2. Forensic Analysis: PRNU Identification
Every camera sensor has microscopic physical imperfections called **Photo Response Non-Uniformity (PRNU)**. 
- **The Process:** This system extracts the sensor noise from a set of images to create a unique device fingerprint.
- **The Goal:** To prove if an "Unknown" image was taken by a specific physical device, a technique used in high-stakes digital forensics and deepfake detection.

## 3. Computer Vision Suite (YOLOv11)
The project utilizes a comprehensive YOLOv11 pipeline to analyze forensic artifacts and original images across five domains:
- **Object Detection:** Real-time identification of 80+ classes.
- **Instance Segmentation:** Pixel-level masking of individual objects.
- **Image Classification:** High-accuracy category labeling.
- **Pose Estimation:** Human skeleton and keypoint tracking.
- **Oriented Object Detection (OBB):** Detecting rotated objects for better spatial awareness.

## 4. Feature Engineering: HOG & Noise Removal
To understand how AI "sees" forensic artifacts, the project implements:
- **HOG (Histogram of Oriented Gradients):** Visualizing the structural edges the model uses for detection.
- **Denoising Pipeline:** Comparing model performance on raw vs. filtered images to measure the impact of sensor noise on AI accuracy.

## 5. Technology Stack
- **Deep Learning:** Ultralytics YOLOv11
- **Image Processing:** OpenCV, Scikit-Image
- **Forensics:** PRNU (Photo Response Non-Uniformity) Analysis
- **Mathematics:** NumPy, SciPy (Signal Processing)

## 6. How to Run
1. **Setup Images:** Place 5 images from each device in `images/Phone_Name/`.
2. **Install Dependencies:** `pip install -r requirements.txt`
3. **Run Pipeline:** ```bash
   python src/main.py
