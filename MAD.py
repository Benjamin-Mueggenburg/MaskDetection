import cv2
import argparse

from DTCTracker import DTCTracker

parser = argparse.ArgumentParser()
parser.add_argument('--source', '-s', type=str, default='../videos/170111_040_Tokyo_TrainStation_1080p.mp4', help='source') #File
parser.add_argument('--output', '-o', type=str, default='./results/dnnx_models.mp4', help='output mp4')

args = parser.parse_args()

def vis_preds(frame, tracks):
    #TODO Visulise total stats - % of people wearing mask, total number of people detected

    for track in tracks:
        (mask, withoutMask) = track.classification
        label = "Mask" if mask > withoutMask else "No Mask"

        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        confidence_score = max(mask, withoutMask) * 100
        bbox = track.to_tlbr()  
        bbox = list(map(int, bbox)) #Convert list of str to list of int

        detection_confidence = track.detection_confidence

        text = str(round(confidence_score / 100, 7))
        xy = tuple(bbox[:2])
        text_location = ( int(xy[0] + 10), int(xy[1] + 20) )

        frame = cv2.putText(frame, text, text_location, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)
        frame = cv2.rectangle(frame, tuple(bbox[:2]), tuple(bbox[2:]), color, thickness=2)

    return frame

def detect_classify_track_one_frame():
    frame = cv2.imread("../images/mask3.jpg")
    DTC = DTCTracker()
    
    width  = cap.get(3)  # float `width`
    height = cap.get(4)

    if resize_factor > 1:
        frame =  cv2.resize(frame, (int(width//resize_factor), int(height//resize_factor)))
    
    DTC.update(frame, BGR=True)
    tracks = DTC.getTracks()

def detect_classify_track(cap, writer):
    #Load DTC model/tracker
    DTC = DTCTracker()

    width  = cap.get(3)  # float `width`
    height = cap.get(4)

    #MAIN LOOP
    frame_idx = 0

    while cap.isOpened():

        if frame_idx % frame_step == 0:
            ret, frame = cap.read()
            
            if not ret:
                break

            if resize_factor > 1:
                frame = cv2.resize(frame, (int(width//resize_factor), int(height//resize_factor)))
            

            DTC.update(frame, BGR=True)

            tracks = DTC.getTracks()
            frame = vis_preds(frame, tracks)


            writer.write(frame)

            cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print("Done")


#CONFIG
frame_step = 1 #Every nth frame gets passed to DTCtracker
resize_factor = 2 # Resize each frame passed to DTCtracker

if __name__ == '__main__':
    cap = cv2.VideoCapture(args.source)

    if cap.isOpened() == False:
        raise Exception("Error opening video file")

    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter(args.output, fourcc, fps//frame_step, (int(cap.get(3))//resize_factor, int(cap.get(4))//resize_factor) )

    detect_classify_track(cap, writer)
    #detect_classify_track_one_frame()

    cap.release()
    writer.release()
    cv2.destroyAllWindows()