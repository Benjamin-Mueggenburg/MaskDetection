from models.YoloV5Face.model import Yolov5FaceModel, convert_tlbr_to_tlwh, convert_tlwh_to_tlbr
from general import device
from DTCTracker import ImageTensor

import torch
import cv2



frame_step = 1 #Every nth frame gets passed to DTCtracker
resize_factor = 2 # Resize each frame passed to DTCtracker

def get_detections(frame, model):
    image_tensor =  ImageTensor(torch.from_numpy(frame.copy()).to(device))
    image = image_tensor.channels_last #Un edited image

    preproc_image = model.preprocess(image)

    preds = model.inference(preproc_image)
    preds = model.nms(preds)


    if preds[0].shape[0] == 0:
        #No faces detected
        return []

    scaled_preds = model.postprocess_scale(preds, preproc_image.shape[-2:], image.shape[:2]) # dets are torch.Tensor with shape of [num_det, 5]. bbox in top left, bottom right format (tlbr)
    #Detections are in tlwh format
    return scaled_preds


def vis_preds(frame, detections):
    #TODO Visulise total stats - % of people wearing mask, total number of people detected
    for detection in detections:
        detection = detection.tolist()
        
        color = (0, 255, 0) 
        
        h,w,c = frame.shape
        xywh = detection[:4]  #Convert list of str to list of int
        x1 = int(xywh[0] * w)
        y1 = int(xywh[1] * h)
        x2 = int(x1 + xywh[2] * w)
        y2 = int(y1 + xywh[3] * h)
        
        detection_confidence = detection[4]



        text = str(round(detection_confidence, 3))
        text_location = ( int(x1 + 10), int(y1 + 20) )

        frame = cv2.putText(frame, text, text_location, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)
        frame = cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness=2)

    return frame


def detect_faces(cap, model):
    frame_idx = 0
    width  = cap.get(3)  # float `width`
    height = cap.get(4)

    while cap.isOpened():

        if frame_idx % frame_step == 0:
            ret, frame = cap.read()
            
            if not ret:
                break

            if resize_factor > 1:
                frame = cv2.resize(frame, (int(width//resize_factor), int(height//resize_factor)))

            detections = get_detections(frame[:, :, ::-1], model)


            cv2.imshow('frame', vis_preds(frame, detections))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == '__main__':
    source_video = "../videos/170111_040_Tokyo_TrainStation_1080p.mp4"
    cap = cv2.VideoCapture(source_video)

    if cap.isOpened() == False:
        raise Exception("Error opening video file")

    faceDetection = Yolov5FaceModel()
    faceDetection.init()

    detect_faces(cap, faceDetection)

    
    #detect_classify_track_one_frame()

    cap.release()
    cv2.destroyAllWindows()