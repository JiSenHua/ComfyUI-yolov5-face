import os
import sys
import torch
import numpy as np
import cv2
from PIL import Image
import folder_paths

NODE_DIR = os.path.dirname(os.path.abspath(__file__))
YOLOV5_FACE_PATH = os.path.join(NODE_DIR, "yolov5face")

sys.path.insert(0, NODE_DIR)
sys.path.insert(0, YOLOV5_FACE_PATH)

from yolov5face.models.experimental import attempt_load
from yolov5face.utils.general import non_max_suppression, scale_coords
from yolov5face.utils.torch_utils import select_device

COMFYUI_ROOT = os.path.dirname(os.path.dirname(NODE_DIR))
YOLOV5_MODELS_DIR = os.path.join(COMFYUI_ROOT, "models", "yolov5")

def get_yolov5_model_list():
    if not os.path.exists(YOLOV5_MODELS_DIR):
        os.makedirs(YOLOV5_MODELS_DIR, exist_ok=True)
        return []
    
    model_files = []
    for file in os.listdir(YOLOV5_MODELS_DIR):
        if file.endswith(('.pt', '.pth')):
            model_files.append(file)
    
    return sorted(model_files) if model_files else ["yolov5s-face.pt"]

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    shape = img.shape[:2]  
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup: 
        r = min(r, 1.0)

    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto: 
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  
    elif scaleFill:  
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, ratio, (dw, dh)

class FaceDetector:
    def __init__(self, weights_name, imgsz=640):
        weights_path = os.path.join(YOLOV5_MODELS_DIR, weights_name)
        if not os.path.isfile(weights_path):
            raise FileNotFoundError(f"model not found: {weights_path}")
        self.device = select_device('')
        self.model = attempt_load(weights_path, map_location=self.device)
        self.model.eval()
        self.imgsz = imgsz
        dummy = torch.zeros(1, 3, imgsz, imgsz).to(self.device)
        with torch.no_grad():
            _ = self.model(dummy)

    def get_most_centered_face(self, faces, img_shape):
        """选择最居中且大小合适的人脸"""
        img_center = np.array([img_shape[1] / 2, img_shape[0] / 2])
        best_score = float('-inf')
        best_box = None
        
        for box in faces:
            x1, y1, x2, y2 = box
            face_center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
            
            dist = np.linalg.norm(face_center - img_center)
            max_dist = np.linalg.norm(img_center)
            normalized_dist = 1 - (dist / max_dist)
            
            face_area = (x2 - x1) * (y2 - y1)
            img_area = img_shape[0] * img_shape[1]
            normalized_size = min(face_area / img_area, 0.5)
            
            score = normalized_dist * 0.7 + normalized_size * 0.3
            
            if score > best_score:
                best_score = score
                best_box = box
        
        return best_box

    def draw_face_box(self, img, box, color=(0, 255, 0), thickness=3):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        return img

    def crop_face_square(self, img, box, expand_ratio=1.5, offset_x=0, offset_y=0):
        x1, y1, x2, y2 = map(int, box)
        
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        face_width = x2 - x1
        face_height = y2 - y1

        offset_x_pixels = int(offset_x * max(face_width, face_height))
        offset_y_pixels = int(offset_y * max(face_width, face_height))
        
        center_x += offset_x_pixels
        center_y += offset_y_pixels
        
        expanded_size = int(max(face_width, face_height) * expand_ratio)
        
        half_size = expanded_size // 2
        crop_x1 = center_x - half_size
        crop_y1 = center_y - half_size
        crop_x2 = center_x + half_size
        crop_y2 = center_y + half_size
        
        square_size = expanded_size
        square_img = np.zeros((square_size, square_size, 3), dtype=np.uint8)
        
        src_x1 = max(0, crop_x1)
        src_y1 = max(0, crop_y1)
        src_x2 = min(img.shape[1], crop_x2)
        src_y2 = min(img.shape[0], crop_y2)
        
        dst_x1 = src_x1 - crop_x1
        dst_y1 = src_y1 - crop_y1
        dst_x2 = dst_x1 + (src_x2 - src_x1)
        dst_y2 = dst_y1 + (src_y2 - src_y1)
        
        dst_x1 = max(0, dst_x1)
        dst_y1 = max(0, dst_y1)
        dst_x2 = min(square_size, dst_x2)
        dst_y2 = min(square_size, dst_y2)
        
        if dst_x2 > dst_x1 and dst_y2 > dst_y1:
            src_x2 = src_x1 + (dst_x2 - dst_x1)
            src_y2 = src_y1 + (dst_y2 - dst_y1)
            
            if src_x1 < img.shape[1] and src_y1 < img.shape[0] and src_x2 > 0 and src_y2 > 0:
                square_img[dst_y1:dst_y2, dst_x1:dst_x2] = img[src_y1:src_y2, src_x1:src_x2]
        
        face_512 = cv2.resize(square_img, (512, 512), interpolation=cv2.INTER_LINEAR)
        
        return face_512

    def detect_and_mark_face(self, pil_img, expand_ratio=1.5, offset_x=0, offset_y=0):
        if isinstance(pil_img, torch.Tensor):
            tensor_img = pil_img
            
            if tensor_img.dim() == 4:
                tensor_img = tensor_img.squeeze(0)
            
            if tensor_img.dim() == 3 and tensor_img.shape[0] in [1, 3]:  # CHW 格式
                tensor_img = tensor_img.permute(1, 2, 0)  # 转为 HWC
            
            if tensor_img.max() <= 1.0:  
                img_np = (tensor_img.cpu().numpy() * 255).astype(np.uint8)
            else:  
                img_np = tensor_img.cpu().numpy().astype(np.uint8)
            
            pil_img = Image.fromarray(img_np)

        img0 = np.array(pil_img.convert("RGB"))[:, :, ::-1]  # RGB to BGR
        
        orig_h, orig_w = img0.shape[:2]
        
        img, ratio, pad = letterbox(img0, new_shape=(self.imgsz, self.imgsz), auto=False)
        
        img_in = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
        img_in = np.ascontiguousarray(img_in)
        img_in = torch.from_numpy(img_in).to(self.device, dtype=torch.float32)
        img_in /= 255.0
        if img_in.ndimension() == 3:
            img_in = img_in.unsqueeze(0)
        
        with torch.no_grad():
            pred = self.model(img_in, augment=False)[0]
        pred = non_max_suppression(pred, 0.25, 0.45, classes=0, agnostic=False)
        
        faces = []
        for det in pred:
            if det is not None and len(det):
                for *xyxy, conf, cls in det:
                    x1 = (xyxy[0] - pad[0]) / ratio[0]
                    y1 = (xyxy[1] - pad[1]) / ratio[1]
                    x2 = (xyxy[2] - pad[0]) / ratio[0]
                    y2 = (xyxy[3] - pad[1]) / ratio[1]
                    
                    x1 = max(0, min(orig_w, x1))
                    y1 = max(0, min(orig_h, y1))
                    x2 = max(0, min(orig_w, x2))
                    y2 = max(0, min(orig_h, y2))
                    
                    faces.append([int(x1), int(y1), int(x2), int(y2)])
        
        if not faces:
            face_rgb = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
            face_tensor = torch.from_numpy(face_rgb).float() / 255.0
            face_tensor = face_tensor.unsqueeze(0)
            
            black_img = np.zeros((512, 512, 3), dtype=np.uint8)
            black_tensor = torch.from_numpy(black_img).float() / 255.0
            black_tensor = black_tensor.unsqueeze(0)
            
            return face_tensor, black_tensor
        
        # 选择最佳人脸
        best_box = self.get_most_centered_face(faces, img0.shape)
        
        result_img = img0.copy()
        result_img = self.draw_face_box(result_img, best_box, color=(0, 255, 0), thickness=3)
        
        face_crop = self.crop_face_square(img0, best_box, expand_ratio, offset_x, offset_y)
        
        marked_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
        marked_tensor = torch.from_numpy(marked_rgb).float() / 255.0
        marked_tensor = marked_tensor.unsqueeze(0)  # 添加 batch 维度
        
        face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        face_tensor = torch.from_numpy(face_rgb).float() / 255.0
        face_tensor = face_tensor.unsqueeze(0)  # 添加 batch 维度
        
        return marked_tensor, face_tensor


class NodeFaceDetect:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Input image to detect faces from."}),
                "weights_name": (get_yolov5_model_list(), {
                    "default": "yolov5s-face.pt",
                    "tooltip": "Select the YOLOv5 face detection model to use."
                }),
                "expand_ratio": ("FLOAT", {
                    "default": 1,
                    "min": 1.0,
                    "max": 3.0,
                    "step": 0.1,
                    "tooltip": "Expansion ratio for face cropping. 1.0 = original face box, 2.0 = double the size."
                }),
                "offset_x": ("FLOAT", {
                    "default": 0.0,
                    "min": -1.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Horizontal offset for face cropping. Positive = right, negative = left. Range is relative to face size."
                }),
                "offset_y": ("FLOAT", {
                    "default": 0.0,
                    "min": -1.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Vertical offset for face cropping. Positive = down, negative = up. Range is relative to face size."
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("marked_image", "face_crop")
    FUNCTION = "process"
    CATEGORY = "ComfyUI-yolov5-face"

    def __init__(self):
        self.detector = None
        self.last_weights = None

    def process(self, image, weights_name, expand_ratio, offset_x, offset_y):
        if self.detector is None or self.last_weights != weights_name:
            self.detector = FaceDetector(weights_name)
            self.last_weights = weights_name
        
        marked_img, face_crop = self.detector.detect_and_mark_face(image, expand_ratio, offset_x, offset_y)
        
        return (marked_img, face_crop)


NODE_CLASS_MAPPINGS = {
    "FaceDetect": NodeFaceDetect,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FaceDetect": "Yolov5 Face Detect",
}
