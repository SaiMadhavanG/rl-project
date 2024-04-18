import torch
import torch.nn as nn
import numpy as np
import cv2


def createMLP(inDim, outDim, device):
    return nn.Sequential(
        nn.Linear(inDim, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, outDim),
    ).to(device)


def createCNN(m, outDim, device):
    """
    The input to the network is a 84x84x4
    tensor containing a rescaled, and gray-scale, version of the last four
    frames. The first convolution layer convolves the input with 32 filters
    of size 8 (stride 4), the second layer has 64 layers of size 4
    (stride 2), the final convolution layer has 64 filters of size 3 (stride
    1). This is followed by a fully-connected hidden layer of 512 units.
    All these layers are separated by Rectifier Linear Units (ReLu). Finally,
    a fully-connected linear layer projects to the output of the
    network, i.e., the Q-values.
    """
    return nn.Sequential(
        nn.Conv2d(m, 32, (8, 8), 4),
        nn.ReLU(),
        nn.Conv2d(32, 64, (4, 4), 2),
        nn.ReLU(),
        nn.Conv2d(64, 64, (3, 3)),
        nn.ReLU(),
        nn.Linear(3136, 512),
        nn.ReLU(),
        nn.Linear(512, outDim),
    ).to(device)


def preprocessFrame(frame, prev_frame):
    max_frame = np.maximum(frame, prev_frame) if prev_frame is not None else frame

    grayscale = max_frame[:, :, 0]
    resized = cv2.resize(grayscale, (84, 84), interpolation=cv2.INTER_AREA)

    return resized


def preprocessMap(frames, device):
    m = len(frames)
    prev_frame = None
    output = []
    for frame in frames:
        output.append(preprocessFrame(frame, prev_frame))
        prev_frame = frame
    return torch.Tensor(output).to(device)
