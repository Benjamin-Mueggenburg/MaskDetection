import module
import onnxruntime
import torch
import cv2

from general import rescale_img, device

from models.YoloV5Face.utils.datasets import letterbox_torch
from models.YoloV5Face.utils.general import check_img_size, non_max_suppression_face, apply_classifier, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging, increment_path

def convert_tlbr_to_tlwh(boxes):
    '''
    Convert bboxs in format of x1, y1, x2, y2 to x1, y1, w, h, confidence
    '''
    if isinstance(boxes, torch.Tensor):
        #Torch implementation
        if boxes.dim() == 1:
            #Single box rather than multiple
            boxes[2] = boxes[2] - boxes[0]  # width
            boxes[3] = boxes[3] - boxes[1] #height
            return boxes
        else:
            #Assuming two dimensions means mulitiple boxes
            boxes[:, 2] = boxes[:, 2] - boxes[:, 0]  # width
            boxes[:, 3] = boxes[:, 3] - boxes[:, 1]  # height
            return boxes
    else:
        #Numpy
        return [ [box[0], box[1], box[2] - box[0], box[3] - box[1], box[4]] for box in boxes ]

def convert_tlwh_to_tlbr(boxes: torch.Tensor):
    '''
    Convert bboxs in format of x1, y1, w, h to x1, y1, x2, y2
    '''
    if isinstance(boxes, torch.Tensor):
        if boxes.dim() == 1:
            boxes[2:4] += boxes[:2]
            return boxes
        else:
            #Assuming 2 dimensions
            boxes[:, 2:4] += boxes[:, :2]
            return boxes 

class Yolov5FaceModel(module):
    '''This model takes image -> {img: <originalInputImage>, dets: <scaled_detections>}
        dets are in form of xywh'''
    def __init__(self):
        super(__class__, self)

        #Constants
        self.weight_path = "./weights/ONNX/yolov5n-0.5_5-09-2021.onnx"

        self.stride8_shape=(1,3,80,80,16)
        self.stride16_shape=(1,3,40,40,16)
        self.stride32_shape=(1,3,20,20,16)

        self.img_size = 640

        self.conf_thres = 0.3
        self.iou_thres = 0.5

        self.max_stride = 32

        #VARS TO BE CLEARED AFTER EACH RUN
        self.data = {}

    def initialise_weights(self):
        providers = ['CPUExecutionProvider']
        if onnxruntime.get_device() == "CPU":
            print("Using CPU for inference - https://stackoverflow.com/questions/64452013/how-do-you-run-a-onnx-model-on-a-gpu")
        elif onnxruntime.get_device() == "GPU":
            print("ONNXruntime using GPU")

        return onnxruntime.InferenceSession(self.weight_path, providers=providers)

    def preprocess(self, img):
        '''Given a (image tensor) preprocess for inference'''
        _img = img.detach().clone()
        h0, w0 = _img.shape[:2]
    
        r = self.img_size / max(h0, w0)
        if r != 1:
            #CV2
            #interp = cv2.INTER_AREA if r < 1  else cv2.INTER_LINEAR
            #img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)
            #Torch
            interp = "area" if r < 1 else "linear"
            _img = rescale_img(_img, (int(w0 * r), int(h0 * r)), mode=interp)
        imgsz = check_img_size(self.img_size, s=self.max_stride)  # check img_size
        _img = letterbox_torch(_img, new_shape=imgsz, auto=False)[0]      
        # Convert
        _img = _img.permute(2, 0, 1)  # BGR to RGB, to 3x416x416
        _img = _img.float()  # uint8 to fp16/32

        _img /= 255.0  # 0 - 255 to 0.0 - 1.0
        
        if _img.ndimension() == 3:
            _img = _img.unsqueeze(0)

        return _img

    def inference(self, img):
        preds = self.model.run(None, {self.model.get_inputs()[0].name: img.cpu().numpy()})  
        stride_8 = preds[0].reshape(*self.stride8_shape) 
        stride_16 = preds[1].reshape(*self.stride16_shape) 
        stride_32 = preds[2].reshape(*self.stride32_shape) 

        strides = [stride_8,stride_16,stride_32]
        preds = self.detection_layer(strides)
        return preds

    def nms(self, preds):
        '''Non max supression'''
        return non_max_suppression_face(preds, self.conf_thres, self.iou_thres)

    def postprocess_scale(self, pred, orgimg_shape):
        '''Returns preds scaled to original image shape'''
        dets = torch.Tensor().to(device) #TORCH (bbox shape is 4, confidence shape is 1 = 5) empty tensor with no shape
        #dets = [] #NUMPY
        for i, det in enumerate(pred):  # detections per image
            gn = torch.tensor(orgimg_shape)[[1, 0, 1, 0]].to(device)  # normalization gain whwh
            gn_lks = torch.tensor(orgimg_shape)[[1, 0, 1, 0, 1, 0, 1, 0, 1, 0]].to(device)  # normalization gain landmarks
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], orgimg_shape).round()

                #WE DO NOT USE LANDMARKS SO SKIP
                #   det[:, 5:15] = self.scale_coords_landmarks(img.shape[2:], det[:, 5:15], orgimg_shape).round()

                for j in range(det.size()[0]):
                    #TORCH 
                    bbox = det[j, :4].view(-1)
                    conf = det[j, 4]
                    dets = torch.cat((dets, torch.cat(bbox, conf).unsqueeze(0))) #Assume torch.cat(bbox, conf) has a shape of [5] not [1,5]
                    
                    #NUMPY
                    # bbox = det[j, :4].view(-1).tolist()
                    # conf = det[j, 4].cpu().numpy()
                    
                    # dets.append([*bbox, *conf])

        return dets 

    def scale_coords_landmarks(img1_shape, coords, img0_shape, ratio_pad=None):
        # Rescale coords (xyxy) from img1_shape to img0_shape
        if ratio_pad is None:  # calculate from img0_shape
            gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
            pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
        else:
            gain = ratio_pad[0][0]
            pad = ratio_pad[1]

        coords[:, [0, 2, 4, 6, 8]] -= pad[0]  # x padding
        coords[:, [1, 3, 5, 7, 9]] -= pad[1]  # y padding
        coords[:, :10] /= gain
        #clip_coords(coords, img0_shape)
        coords[:, 0].clamp_(0, img0_shape[1])  # x1
        coords[:, 1].clamp_(0, img0_shape[0])  # y1
        coords[:, 2].clamp_(0, img0_shape[1])  # x2
        coords[:, 3].clamp_(0, img0_shape[0])  # y2
        coords[:, 4].clamp_(0, img0_shape[1])  # x3
        coords[:, 5].clamp_(0, img0_shape[0])  # y3
        coords[:, 6].clamp_(0, img0_shape[1])  # x4
        coords[:, 7].clamp_(0, img0_shape[0])  # y4
        coords[:, 8].clamp_(0, img0_shape[1])  # x5
        coords[:, 9].clamp_(0, img0_shape[0])  # y5
        return coords
        
    def detection_layer(self, strides):
        '''[stride_8, stride_16, stride_32] implements the ommited section of the Detection block when exporting PyTorch -> ONNX'''
        stride= torch.tensor([8.,16.,32.]).to(device)

        x=[torch.from_numpy(strides[0]).to(device),torch.from_numpy(strides[1]).to(device),torch.from_numpy(strides[2]).to(device)]

        no=16 #num outputs
        nl=3 #num layers

        grid=[torch.zeros(1).to(device)] * nl 

        anchor_grid=torch.tensor([[[[[[  4.,   5.]]],
            [[[  8.,  10.]]],
            [[[ 13.,  16.]]]]],
            [[[[[ 23.,  29.]]],
            [[[ 43.,  55.]]],
            [[[ 73., 105.]]]]],
            [[[[[146., 217.]]],
            [[[231., 300.]]],
            [[[335., 433.]]]]]]).to(device)
        
        z = [] 
        for i in range(len(x)):
        
            bs,ny, nx = x[i].shape[0],x[i].shape[2] ,x[i].shape[3] 
            if grid[i].shape[2:4] != x[i].shape[2:4]:
                grid[i] = self._make_grid(nx, ny).to(x[i].device)
            y = torch.full_like(x[i], 0)
            y[..., [0,1,2,3,4,15]] = x[i][..., [0,1,2,3,4,15]].sigmoid()
            y[..., 5:15] = x[i][..., 5:15]
            #y = x[i].sigmoid()

            y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + grid[i].to(x[i].device)) * stride[i]  # xy
            y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * anchor_grid[i]  # wh

            #y[..., 5:15] = y[..., 5:15] * 8 - 4
            y[..., 5:7]   = y[..., 5:7] *   anchor_grid[i] + grid[i].to(x[i].device) * stride[i] # landmark x1 y1
            y[..., 7:9]   = y[..., 7:9] *   anchor_grid[i] + grid[i].to(x[i].device) * stride[i]# landmark x2 y2
            y[..., 9:11]  = y[..., 9:11] *  anchor_grid[i] + grid[i].to(x[i].device) * stride[i]# landmark x3 y3
            y[..., 11:13] = y[..., 11:13] * anchor_grid[i] + grid[i].to(x[i].device) * stride[i]# landmark x4 y4
            y[..., 13:15] = y[..., 13:15] * anchor_grid[i] + grid[i].to(x[i].device) * stride[i]# landmark x5 y5

            #y[..., 5:7] = (y[..., 5:7] * 2 -1) * anchor_grid[i]  # landmark x1 y1
            #y[..., 7:9] = (y[..., 7:9] * 2 -1) * anchor_grid[i]  # landmark x2 y2
            #y[..., 9:11] = (y[..., 9:11] * 2 -1) * anchor_grid[i]  # landmark x3 y3
            #y[..., 11:13] = (y[..., 11:13] * 2 -1) * anchor_grid[i]  # landmark x4 y4
            #y[..., 13:15] = (y[..., 13:15] * 2 -1) * anchor_grid[i]  # landmark x5 y5

            z.append(y.view(bs, -1, no))
        return torch.cat(z, 1)

    def _make_grid(self,nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()



