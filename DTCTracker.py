import torch
import numpy as np

from models.YoloV5Face.model import Yolov5FaceModel, convert_tlbr_to_tlwh, convert_tlwh_to_tlbr

from models.MaskClassifier.model import EfficientNetModel

from models.deepsort_torch.deep_sort_realtime.embedder.embedder_pytorch import MobileNetv2_Embedder as Embedder
from models.deepsort_torch.deep_sort_realtime.deep_sort.detection import Detection
from models.deepsort_torch.deep_sort_realtime.deep_sort.tracker import Tracker
from models.deepsort_torch.deep_sort_realtime.deep_sort import nn_matching
from models.deepsort_torch.deep_sort_realtime.utils.nms import non_max_suppression

from general import device


class DeepsortCfg:
    #Embedder cfg
    deepsort_path = './models/deepsort_torch/deep_sort_realtime/embedder/weights/mobilenetv2_bottleneck_wts.pt'
    half = True
    max_batch_size = 16
    bgr = False

    #Metric config
    max_cosine_distance = 0.3
    nn_budget = None

    #NMS config
    nms_max_overlap = 1.0


class ImageTensor:
    def __init__(self, image_tensor):
        self.image_tensor = image_tensor
        self.image_tensor_cwh = image_tensor.permute(2,0,1)
    @property
    def channels_last(self):
        ''' Default - Image tensor shape [Width, Height, Channels]'''
        return self.image_tensor.clone()
    @property
    def channels_first(self):
        '''Image tensor shape [Channels, Width, Height]'''
        return self.image_tensor_cwh.clone()


class DTCTracker:
    def __init__(self):
        self.faceDetection = Yolov5FaceModel()
        self.maskClassifier = EfficientNetModel()
        self.embeder = Embedder(model_wts_path=DeepsortCfg.deepsort_path, half=DeepsortCfg.half, max_batch_size=DeepsortCfg.max_batch_size, bgr=False)
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", DeepsortCfg.max_cosine_distance, DeepsortCfg.nn_budget)
        self.tracker = Tracker(metric)



        self.init()
    def init(self):
        self.faceDetection.init()
        self.maskClassifier.init()

    #Public functions to call
    def update(self, frame, BGR=True):
        '''Frame is raw numpy frame'''
        if BGR:
            frame = frame[:, :, ::-1] #TO RGB
        self.image_tensor = ImageTensor(torch.from_numpy(frame.copy()).to(device)) #must copy() frame otherwise weird errors
        #self.image_tensor is of [Width, Height, Channel] shape, whereas image_tensor

        detections = self.get_yolo_detections(self.image_tensor.channels_last)
        self.update_tracker(detections, self.image_tensor)
        #del detections #detections are no longer use. TODO use tensor of detections for as long as possbile
        self.classify()

    def getTracks(self):
        ''' Get Main detections including detect class - should have valid tracks + valid classification
        '''
        return [track for track in self.tracker.tracks if self.is_valid_track(track) and not np.any(track.classification == None)]

    def getAllTracks(self):
        '''Gets all tracks - including ones that haven't been classified'''
        return self.tracker.tracks

    #####
    #YoloV5-face model
    def get_yolo_detections(self, frame):
        '''Yolov5face - frame is channels_last'''
        preproc_frame = self.faceDetection.preprocess(frame)

        preds = self.faceDetection.inference(preproc_frame)
        preds = self.faceDetection.nms(preds)
        #preds is always a list with one Tensor regardless if it detects anything
        if preds[0].shape[0] == 0:
            #No faces detected
            return []
        scaled_preds = self.faceDetection.postprocess_scale(preds, preproc_frame.shape[-2:], frame.shape[:2]) # dets are torch.Tensor with shape of [num_det, 5]. bbox in top left, bottom right format (tlbr)
        scaled_preds = convert_tlbr_to_tlwh(scaled_preds) #convert bboxs from top-left-bottom-right to top-left-width-height

        return scaled_preds
    ######
    #Mask Detection classify
    def classify(self):
        current_tracks = self.getAllTracks()

        track_idx = []
        for i, track in enumerate(current_tracks):
            if self.is_valid_track(track):
                if self.should_update_classification(track):
                    track_idx.append(i)
                else:
                    track.time_since_classification += 1
        
        bboxs = [current_tracks[idx].to_tlbr() for idx in track_idx] #bboxs are numpy arrays

        preds = self.batch_classify(self.image_tensor, bboxs)
        #Modify tracks now
        for idx, pred in zip(track_idx, preds):
            current_tracks[idx].classification = pred
            current_tracks[idx].time_since_classification = 0

    def batch_classify(self, frame: ImageTensor, bboxs):
        #If possilbe implement batch sizes here, rather than processing all rois at once. Dunno if Ill need to cause how many faces are there in a picture?
        #bboxs list of numpy array
        if len(bboxs) == 0:
            return []
        
        #Could really just use the get_image_patch code below but who cares
        rois = [ ImageTensor.channels_last[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])] for bbox in bboxs]
        return self.maskClassifier.batch_classify(rois)

        

    def should_update_classification(self, track):
        '''
        Should track update classification - logic for determining whether track should update classification
        '''
        if np.any(track.classification == None):
            # Track has not been classified before
            return True
        else:
            return max(track.classification[0], track.classification[1]) < 0.8
    ######
    #Deepsort model
    def get_image_patch(self, bbox, frame: ImageTensor):
        '''Crop frame to bbox TLWH'''
        tlbr = convert_tlwh_to_tlbr(bbox).cpu().int().tolist()
        xmin, ymin, xmax, ymax = tlbr
        return frame.channels_first[:, ymin:ymax, xmin:xmax]

    def get_features(self, bboxs: torch.Tensor, frame: ImageTensor):
        '''Given bboxs and frame, use deepsort to get features, returns List[np.array with shape 1280]''' 

        #bboxs have shape of (<num of bboxs>, 4) where it's format is top left width height (tlwh), image patchs do have 
        image_patches = [self.get_image_patch(box, frame) for box in bboxs]
        return self.embeder.predict(image_patches)

    def is_valid_track(self, track):
        '''
        Is track confirmed and been updated
        '''
        return track.det_cls == "face" and track.is_confirmed() and not track.time_since_update > 1

    def update_tracker(self, detections: torch.Tensor, image_tensor: ImageTensor):
        # if detections is NUMPY
        # boxes = [ detection[:4] for detection in detections]
        # confidences = [ detection[4] for detection in detections]
        if detections == []:
            #No detections just update tracker
            self.tracker.predict()
            return

        #detections are torch

        #Convert detections to Detection objects, while encoding
        #detections = [ Detection(bbox, confidence, "face", feature) for bbox, confidence, feature in zip(boxes, confidences, features)]
        boxes = detections[:, :4]
        confidences = detections[:, 4]
        features = self.get_features(boxes, image_tensor) #list of numpy arrays

        #detections = [ Detection(bbox, "face", feature) for bbox, feature in enumerate(zip(detections, features))]
        detectionObjs = [ Detection(detections[i], "face", features[i]) for i in range(len(detections)) ]

        #Run non-maxima supression <again????>
        if DeepsortCfg.nms_max_overlap < 1.0:
            np_boxes = boxes.cpu().numpy()
            np_confidences = confidences.cpu().numpy()
            indices = non_max_suppression(np_boxes, DeepsortCfg.nms_max_overlap, np_confidences)
            detectionObjs = [detectionObjs[i] for i in indices]
            
        self.tracker.predict()
        self.tracker.update(detectionObjs)
