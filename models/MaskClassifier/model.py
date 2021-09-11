import module

import onnxruntime
import torch
import cv2

class EfficientNetModel(module):
    '''Takes image and returns classification'''
    def __init__(self):
        super(__class__, self)

        self.weights_path = "./weights/ONNX/efficientnet_b3.onnx"

        self.input_shape = (64, 64)

    def initialise_weights(self):
        providers = ['CPUExecutionProvider']
        if onnxruntime.get_device() == "CPU":
            print("Using CPU for inference - https://stackoverflow.com/questions/64452013/how-do-you-run-a-onnx-model-on-a-gpu")
        elif onnxruntime.get_device() == "GPU":
            print("ONNXruntime using GPU")
        
        return onnxruntime.InferenceSession(self.weights_path, providers=providers)

    def preprocess(self, img_tensor):
        return 

    def inference(self, img_tensor):
        return

