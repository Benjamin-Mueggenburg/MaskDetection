import torch
import numpy as np

from models.YoloV5Face.model import Yolov5FaceModel, convert_tlbr_to_tlwh, convert_tlwh_to_tlbr
from models.deepsort_torch.deep_sort_realtime.embedder.embedder_pytorch import MobileNetv2_Embedder as Embedder

from general import device

class DeepsortCfg:
    deepsort_path = './models/deepsort_torch/deep/checkpoint/original_ckpt.t7'
    half = True
    max_batch_size = 16
    bgr = False


class ImageTensor:
    def __init__(self, image_tensor):
        self.image_tensor = image_tensor
        self.image_tensor_cwh = image_tensor.permute(2,0,1)
    @property
    def channels_last(self):
        ''' Default - Image tensor shape [Width, Height, Channels]'''
        return self.image_tensor
    @property
    def channels_first(self):
        '''Image tensor shape [Channels, Width, Height]'''
        return self.image_tensor_cwh


class DTCTracker:
    def __init__(self):
        self.faceDetection = Yolov5FaceModel()
        self.embeder = Embedder(model_wts_path=DeepsortCfg.deepsort_path, half=DeepsortCfg.half, max_batch_size=DeepsortCfg.max_batch_size, bgr=False)

        self.init()
    def init(self):
        self.faceDetection.init()

    def update(self, frame, BGR=True):
        '''Frame is raw numpy frame'''
        if BGR:
            frame = frame[:, :, ::-1] #TO RGB
        self.image_tensor = ImageTensor(torch.from_numpy(frame.copy()).to(device)) #must copy() frame otherwise weird errors
        #self.image_tensor is of [Width, Height, Channel] shape, whereas image_tensor
        detections = self.get_detections(self.image_tensor.channels_last)
    
    def get_detections(self, frame):
        '''Yolov5face - frame is channels_last'''
        preproc_frame = self.faceDetection.preprocess(frame)
        preds = self.faceDetection.inference(preproc_frame)
        preds = self.faceDetection.nms(preds)

        scaled_preds = self.faceDetection.postprocess_scale(preds, frame.shape[:2]) # dets are torch.Tensor with shape of [num_det, 5]. bbox in top left, bottom right format (tlbr)
        scaled_preds = convert_tlbr_to_tlwh(scaled_preds) #convert bboxs from top-left-bottom-right to top-left-width-height
        
        return scaled_preds
    
    def get_image_patch(bbox, frame: ImageTensor):
        '''Crop frame to bbox TLWH'''
        tlbr = convert_tlwh_to_tlbr(bbox).cpu().tolist()
        xmin, ymin, xmax, ymax = tlbr
        return frame.channels_first[:, ymin:ymax, xmin:xmax]

    def get_features(self, bboxs: torch.Tensor, frame: ImageTensor):
        '''Given bboxs and frame, use deepsort to get features''' 

        #bboxs have shape of (<num of bboxs>, 4) where it's format is top left width height (tlwh)
        image_patches = []
        features = self.embeder.predict()
        detections = [ Detection(bbox, confidence, "face", feature) for bbox, confidence, feature in zip(boxes, confidences, features)]

    def update_tracker(self, detections: torch.Tensor):
        # detections is NUMPY
        # boxes = [ detection[:4] for detection in detections]
        # confidences = [ detection[4] for detection in detections]

        #detections are torch
        boxes = detections[:, :4]
        confidences = detections[:, 4]

        #Convert detections to Detection objects, while encoding
        
        features = self.get_features(bboxs, self.image_tensor)
        detections = [ Detection(bbox, confidence, "face", feature) for bbox, confidence, feature in zip(boxes, confidences, features)]

        #Run non-maxima supression

        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence] for d in detections)
        indices = preprocessing.non_max_suppression(boxes, self.nms_max_overlap, scores)
    
        self.detections = [detections[i] for i in indices]

        self.deepsort_tracker.predict()
        
        self.deepsort_tracker.update(detections)
