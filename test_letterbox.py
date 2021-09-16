import torch
import cv2

from DTCTracker import ImageTensor
from models.YoloV5Face.utils.datasets import letterbox_torch, letterbox
from models.YoloV5Face.model import Yolov5FaceModel
from general import rescale_img
import copy
from torchvision.utils import save_image

device = torch.device("cuda")

def test_letterbox_numpy(orgimg, model):

    #detect_video_onnx.py - detect_frame()
    img0 = copy.deepcopy(orgimg)
    h0, w0 = orgimg.shape[:2]
    r = 640 / max(h0, w0)
    if r != 1:  # always resize down, only resize up if training with augmentation
        interp = cv2.INTER_AREA if r < 1  else cv2.INTER_LINEAR
        img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)
    print("scaled_img", img0.shape)
    imgsz = 640 #imgsz = check_img_size(self.img_size, s=self.max_stride)  = 640
    img = letterbox(img0, new_shape=imgsz, auto=False)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1).copy()
    img = torch.from_numpy(img).to(device)
    img = img.float()
    img /= 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    
    return img




def test_letterbox_torch(frame, model):
    frame = frame[:, :, ::-1]
    image_tensor =  ImageTensor(torch.from_numpy(frame.copy()).to(device))
    _frame = image_tensor.channels_last
    #model.preprocess
    _img = _frame.detach().clone()
    h0, w0 = _img.shape[:2]
    _img = _img.permute(2, 0, 1)
    r = 640 / max(h0, w0)
    if r != 1:
        interp = "area" if r < 1 else "linear"
        _img = rescale_img(_img, (int(h0 * r), int(w0 * r)), mode=interp)
    print("scaled_img", _img.shape)
    #imgsz = check_img_size(self.img_size, s=self.max_stride)  = 640
    imgsz = 640
    _img = letterbox_torch(_img, new_shape=imgsz, auto=False)[0]
    _img = _img.float()
    _img /= 255.0  # 0 - 255 to 0.0 - 1.0
        
    if _img.ndimension() == 3:
        _img = _img.unsqueeze(0)

    return _img







def test_letterbox(model, cap):
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
                break
        numpy_test = test_letterbox_numpy(frame, model)
        torch_test = test_letterbox_torch(frame, model)

        print(numpy_test.shape)
        print(torch_test.shape)
        print(torch.equal(numpy_test, torch_test))
        save_image(numpy_test[0], "./numpy_test.jpg")
        save_image(torch_test[0], "./torch_test.jpg")
        exit()


if __name__ == "__main__":
    source_video = "../videos/170111_040_Tokyo_TrainStation_1080p.mp4"
    cap = cv2.VideoCapture(source_video)

   # faceDetection = Yolov5FaceModel()
   # faceDetection.init()
    faceDetection = None

    if cap.isOpened() == False:
        raise Exception("Error opening video file")


    test_letterbox(faceDetection, cap)

    
    #detect_classify_track_one_frame()

    cap.release()
    cv2.destroyAllWindows()