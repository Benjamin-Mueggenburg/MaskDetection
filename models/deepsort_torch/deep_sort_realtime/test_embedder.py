import cv2
import torch

from embedder.embedder_pytorch import MobileNetv2_Embedder as Embedder

half = True

im = cv2.imread('./mask.jpg')
im = im[:, :, ::-1]

#img has format [1500, 1500, 3]
img = torch.from_numpy(im.copy()).float() #get weird At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)

embedder = Embedder(half=half, max_batch_size=16, bgr=False)
embeds = embedder.predict([img])

print(embeds[0].shape)

print(embeds[0])