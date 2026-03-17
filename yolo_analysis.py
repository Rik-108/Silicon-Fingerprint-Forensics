# yolo_analysis.py

import os
from ultralytics import YOLO
import cv2

def run_yolo_analysis(image_sets, output_dir):
    """
    Runs all 5 required YOLO tasks on a provided dict of {phone: list of image paths}.
    Organizes outputs by section_v_yolo/{task_name}_results/{phone_name}/
    """
    print("\n--- Section V: Running All YOLO Tasks ---")

    if not image_sets:
        print("Error: No image sets provided for YOLO analysis.")
        return

    # Define the tasks and their corresponding model files
    tasks = {
        'object_detection': 'yolo11n.pt',
        'instance_segmentation': 'yolo11n-seg.pt',
        'image_classification': 'yolo11n-cls.pt',
        'pose_estimation': 'yolo11n-pose.pt',
        'oriented_object_detection': 'yolo11n-obb.pt'
    }

    yolo_base_dir = os.path.join(output_dir, 'section_v_yolo')
    os.makedirs(yolo_base_dir, exist_ok=True)

    for task_name, model_name in tasks.items():
        print(f"\n----- Running: {task_name.replace('_', ' ').title()} -----")

        task_output_dir = os.path.join(yolo_base_dir, f'{task_name}_results')
        os.makedirs(task_output_dir, exist_ok=True)

        try:
            model = YOLO(model_name)
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            continue

        for phone_name, paths in image_sets.items():
            if not paths:
                continue

            phone_task_dir = os.path.join(task_output_dir, phone_name)
            os.makedirs(phone_task_dir, exist_ok=True)

            print(f"  Processing images for '{phone_name}'...")

            for img_path in paths:
                img_name = os.path.basename(img_path)

                print(f"    Processing '{img_name}'...")

                results = model.predict(img_path, verbose=False)

                base_name = os.path.splitext(img_name)[0]

                if task_name == "image_classification":
                    # Load the original image
                    img = cv2.imread(img_path)
                    if img is None:
                        print(f"      Could not load image '{img_name}'.")
                        continue

                    # Get top 5 classes
                    if hasattr(results[0], 'probs'):
                        probs = results[0].probs
                        names = results[0].names
                        for i in range(min(5, len(probs.top5))):
                            class_name = names[probs.top5[i]]
                            conf = probs.top5conf[i].item()
                            text = f"{i+1}. {class_name}: {conf:.2f}"
                            y_pos = 50 + i * 45  # Increased starting y and spacing for better placement
                            cv2.putText(img, text, (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0),
                                        3)  # Increased scale to 1.0 and thickness to 3

                    output_filename = f"{base_name}_{task_name}.jpg"
                    output_path = os.path.join(phone_task_dir, output_filename)
                    cv2.imwrite(output_path, img)
                    print(f"      Classification image saved.")
                else:
                    output_filename = f"{base_name}_{task_name}.png"
                    output_path = os.path.join(phone_task_dir, output_filename)
                    results[0].save(filename=output_path)
                    print(f"      Visual result saved.")

        print(f"  {task_name.replace('_', ' ').title()} results saved under '{task_output_dir}'.")

    print(f"\n✅ All YOLO tasks complete. Results saved in '{yolo_base_dir}'.")