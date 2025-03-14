import torch
import torchvision.transforms as T
from torchvision.models.segmentation import DeepLabV3_ResNet101_Weights
import numpy as np
import cv2
import gradio as gr
import mediapipe as mp
from PIL import Image
import os

class FaceHairSegmenter:
    def __init__(self):
        # Use MediaPipe for face detection
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1,  # Use full range model
            min_detection_confidence=0.6
        )
        
        # Load DeepLabV3+ model
        self.model = self.load_model()
        
        # Define transforms
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def load_model(self):
        try:
            model = torch.hub.load(
                "pytorch/vision:v0.10.0",
                "deeplabv3_resnet101",
                weights=DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1
            )
            model.eval()
            if torch.cuda.is_available():
                model = model.cuda()
            print("DeepLabV3+ model loaded successfully")
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None

    def detect_faces(self, image):
        """Detect faces using MediaPipe (expects image in RGB)."""
        # Since Gradio returns an image in RGB when type="numpy", no need to convert.
        image_rgb = image  # Already RGB
        h, w = image.shape[:2]
        
        # Process with MediaPipe
        results = self.face_detection.process(image_rgb)
        
        bboxes = []
        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                x_min = max(0, int(bbox.xmin * w))
                y_min = max(0, int(bbox.ymin * h))
                x_max = min(w, int((bbox.xmin + bbox.width) * w))
                y_max = min(h, int((bbox.ymin + bbox.height) * h))
                bboxes.append((x_min, y_min, x_max, y_max))
        
        if len(bboxes) > 1:
            bboxes = self.remove_overlapping_boxes(bboxes)
            
        return len(bboxes), bboxes

    def remove_overlapping_boxes(self, boxes, overlap_threshold=0.5):
        if not boxes:
            return []
        def box_area(box):
            return (box[2] - box[0]) * (box[3] - box[1])
        boxes = sorted(boxes, key=box_area, reverse=True)
        keep = []
        for current in boxes:
            is_duplicate = False
            for kept_box in keep:
                x1 = max(current[0], kept_box[0])
                y1 = max(current[1], kept_box[1])
                x2 = min(current[2], kept_box[2])
                y2 = min(current[3], kept_box[3])
                if x1 < x2 and y1 < y2:
                    intersection = (x2 - x1) * (y2 - y1)
                    area1 = box_area(current)
                    area2 = box_area(kept_box)
                    union = area1 + area2 - intersection
                    iou = intersection / union
                    if iou > overlap_threshold:
                        is_duplicate = True
                        break
            if not is_duplicate:
                keep.append(current)
        return keep

    def segment_face_hair(self, image):
        """Segment face and hair while preserving correct color."""
        if self.model is None:
            return image, "Model not loaded correctly."
        if image is None or image.size == 0:
            return image, "Invalid image provided."
        
        # Detect faces (input image is already in RGB)
        num_faces, bboxes = self.detect_faces(image)
        if num_faces == 0:
            return image, "No face detected! Please upload an image with a clear face."
        elif num_faces > 1:
            debug_img = image.copy()
            for (x_min, y_min, x_max, y_max) in bboxes:
                cv2.rectangle(debug_img, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
            return debug_img, f"{num_faces} faces detected! Please upload an image with exactly ONE face."
        
        bbox = bboxes[0]
        # Use the input image directly as it is already RGB.
        original_rgb = image.copy()

        # Run semantic segmentation using DeepLabV3
        pil_image = Image.fromarray(original_rgb)
        input_tensor = self.transform(pil_image).unsqueeze(0)
        if torch.cuda.is_available():
            input_tensor = input_tensor.cuda()
        with torch.no_grad():
            output = self.model(input_tensor)["out"][0]
        output_predictions = output.argmax(0).byte().cpu().numpy()
        person_mask = (output_predictions == 15).astype(np.uint8) * 255

        h, w = image.shape[:2]
        x_min, y_min, x_max, y_max = bbox
        face_height = y_max - y_min
        face_width = x_max - x_min

        # Expand bounding box for hair region
        y_min_exp = max(0, y_min - int(face_height * 0.8))
        x_min_exp = max(0, x_min - int(face_width * 0.3))
        x_max_exp = min(w, x_max + int(face_width * 0.3))
        y_max_exp = min(h, y_max + int(face_height * 0.1))
        
        face_mask = np.zeros((h, w), dtype=np.uint8)
        face_mask[y_min_exp:y_max_exp, x_min_exp:x_max_exp] = 255
        
        final_mask = cv2.bitwise_and(person_mask, face_mask)
        kernel = np.ones((5, 5), np.uint8)
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)
        final_mask = cv2.GaussianBlur(final_mask, (5, 5), 0)
        _, final_mask = cv2.threshold(final_mask, 127, 255, cv2.THRESH_BINARY)
        
        # Create the RGBA image (the image is already in correct RGB order)
        rgba = np.dstack((original_rgb, final_mask))
        return rgba, "Face and hair segmented successfully!"

# Gradio process function
def process_image(input_img):
    if input_img is None:
        return np.zeros((100, 100, 4), dtype=np.uint8), "Please upload an image."
    segmenter = FaceHairSegmenter()
    result, message = segmenter.segment_face_hair(input_img)
    return result, message

# Gradio interface
def create_interface():
    with gr.Blocks(title="Face & Hair Segmentation") as interface:
        gr.Markdown("""
        # Face-Segmentation Tool
        Upload an image to extract the face and hair with a transparent background.
        ## Guidelines:
        - Upload an image with **exactly one face**
        - The face should be clearly visible
        - For best results, use images with good lighting
        """)
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(label="Input Image", type="numpy")
                process_btn = gr.Button("Extract Face & Hair", variant="primary")
            with gr.Column():
                output_image = gr.Image(label="Segmented Result", type="numpy")
                status_text = gr.Textbox(label="Status")
        process_btn.click(fn=process_image, inputs=input_image, outputs=[output_image, status_text])
    return interface

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    create_interface().launch(server_name="0.0.0.0", server_port=port)