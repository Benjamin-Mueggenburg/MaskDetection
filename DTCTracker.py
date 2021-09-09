import torch
import numpy as np

from models.YoloV5Face.model import Yolov5FaceModel, convert_tlbr_to_tlwh
from models.deepsort_torch.deep.feature_extractor import Extractor
from general import device

class DTCTracker:
    def __init__(self):
        self.faceDetection = Yolov5FaceModel()

        deepsort_path = './models/deepsort_torch/deep/checkpoint/original_ckpt.t7'
        self.extractor = Extractor(self.deepsort_path)

        self.init()
    def init(self):
        self.faceDetection.init()

    def update(self, frame, BGR=True):
        '''Frame is raw numpy frame'''
        if BGR:
            frame = frame[:, :, ::-1] #TO RGB
        self.image_tensor = torch.from_numpy(frame.copy()).to(device) #must copy() frame otherwise weird errors
        detections = self.get_detections(self.image_tensor)
    
    def get_detections(self, frame):
        '''Yolov5face'''
        preproc_frame = self.faceDetection.preprocess(frame)
        preds = self.faceDetection.inference(preproc_frame)
        preds = self.faceDetection.nms(preds)

        scaled_preds = self.faceDetection.postprocess_scale(preds, frame.shape[:2]) # dets are torch.Tensor with shape of [num_det, 5]. bbox in top left, bottom right format (tlbr)
        scaled_preds = convert_tlbr_to_tlwh(scaled_preds) #convert bboxs from top-left-bottom-right to top-left-width-height
        
        return scaled_preds

    def get_features(self, detections: torch.Tensor, frame: torch.Tensor):
        '''Using Deepsort to get features''' 

        boxes = [ detection[:4] for detection in detections]
        confidences = [ detection[4] for detection in detections]
        features = self.encoder(frame, boxes)
        detections = [ Detection(bbox, confidence, "face", feature) for bbox, confidence, feature in zip(boxes, confidences, features)]

    def update_tracker(self, detections: torch.Tensor):
        # detections is NUMPY
        # boxes = [ detection[:4] for detection in detections]
        # confidences = [ detection[4] for detection in detections]

        #detections are torch
        boxes = detections[:, :4].cpu().tolist()
        confidences = detections[:, :5].cpu().tolist()

        #Convert detections to Detection objects, while encoding
        
        features = self.encoder(frame, boxes)
        detections = [ Detection(bbox, confidence, "face", feature) for bbox, confidence, feature in zip(boxes, confidences, features)]

        #Run non-maxima supression

        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence] for d in detections)
        indices = preprocessing.non_max_suppression(boxes, self.nms_max_overlap, scores)
    
        self.detections = [detections[i] for i in indices]

        self.deepsort_tracker.predict()
        
        self.deepsort_tracker.update(detections)
