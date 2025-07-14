from .node import NodeFaceDetect

NODE_CLASS_MAPPINGS = {
    "FaceDetect": NodeFaceDetect,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FaceDetect": "Yolov5 Face Detect",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
