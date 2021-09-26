from module import Module
from general import rescale_img
from torchvision.transforms import transforms
import torch

import onnxruntime
import numpy as np
import cv2

class EfficientNetModel(Module):
    '''Takes image and returns classification'''
    def __init__(self):
        super(__class__, self).__init__()

        self.weights_path = "./weights/ONNX/efficientnet_b3.onnx"

        self.input_shape = (64, 64)

    def initialise_weights(self):
        providers = ['CPUExecutionProvider']
        if onnxruntime.get_device() == "CPU":
            print("Using CPU for inference - https://stackoverflow.com/questions/64452013/how-do-you-run-a-onnx-model-on-a-gpu")
        elif onnxruntime.get_device() == "GPU":
            print("ONNXruntime using GPU")
        
        return onnxruntime.InferenceSession(self.weights_path, providers=providers)

    def preprocess(self, rois):
        processed_rois = []
        for roi in rois:
            roi = roi.permute(2, 0, 1) 
            roi = rescale_img(roi, self.input_shape).float()
            roi /= 127.5
            roi -= 1.
            roi = roi.permute(1,2,0)
            processed_rois.append(roi)
        
        processed_rois = torch.stack(processed_rois)
        return processed_rois


    def batch_classify(self, img_tensors, batch_size=32):
        
        img_tensors = self.preprocess(img_tensors)

        preds = self.model.run(None, {self.model.get_inputs()[0].name: img_tensors.cpu().numpy()})[0]
        return preds

        

