import torch
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset
import random
from torchvision.io import read_video
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import UniformTemporalSubsample
import torch.nn.functional as F
import io
import av
import os

import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from pytorchvideo.data.encoded_video import EncodedVideo
from torchvision.transforms import Compose
from pytorchvideo.transforms import (
    UniformTemporalSubsample,
    ShortSideScale,
    Normalize,
)
from pytorchvideo.models.hub import slowfast_r50
from pytorchvideo.models.head import ResNetBasicHead
from x3d.dataset import mapping_label

def preprocess(video_path):
    
    transform = Compose([
        ShortSideScale(224),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    video_frames, _, _ = read_video(video_path, start_pts=0, end_pts=None, pts_unit="sec")

    # 랜덤한 프레임 선택
    num_frames = video_frames.size(0)  # 전체 프레임 수
    random_frame_idx = random.randint(0, num_frames - 1)  # 랜덤 인덱스 선택
    random_frame = video_frames[random_frame_idx]  # 랜덤 프레임 선택
    pil_image = Image.fromarray(random_frame.numpy())  # PIL 이미지로 변환

    video = EncodedVideo.from_path(video_path, decoder='pyav')

    video_container = video._container
    video_stream = video_container.streams.video[0]
    frames = video_container.decode(video_stream)

    video_frames = [torch.from_numpy(frame.to_rgb().to_ndarray()) for frame in frames]
    video_tensor = torch.stack(video_frames).permute(3, 0, 1, 2).float() / 255.0

    # SlowFast 경로 생성
    slow_pathway = UniformTemporalSubsample(8)(video_tensor)
    fast_pathway = UniformTemporalSubsample(32)(video_tensor)

    slow_pathway = transform(slow_pathway)
    fast_pathway = transform(fast_pathway)
    slow_pathway = slow_pathway.unsqueeze(0)
    fast_pathway = fast_pathway.unsqueeze(0)
    vd_tensor_input = [slow_pathway, fast_pathway]

    return vd_tensor_input, pil_image



def load_trained_model(model, weights_path):
    model.load_state_dict(torch.load(weights_path))
    return model

# Validation 정확도 계산 함수
def evaluate_on_validation(model, video_tensor_, device="cuda"):
    model.to(device)
    model.eval()  # 모델을 평가 모드로 설정

    with torch.no_grad():  # 평가 시에는 그래디언트를 계산하지 않음
        video_tensor_ = [pathway.to(device) for pathway in video_tensor_]  # SlowFast 경로
        y_hat = model(video_tensor_)
        y_hat = y_hat.reshape(y_hat.size(0), -1)
        preds = y_hat.argmax(dim=-1)
        preds=preds.item()

    return preds

def final_pipeline(video_path, weights_path):
    vd_tensor_input, pil_image=preprocess(video_path)
    pretrained_model = slowfast_r50(pretrained=False)
    num_classes = 39
    pretrained_model.blocks[-1] = ResNetBasicHead(
        proj=torch.nn.Linear(pretrained_model.blocks[-1].proj.in_features, num_classes),
        dropout=nn.Dropout(0.5),
    )

    # 가중치 로드
    model = load_trained_model(pretrained_model, weights_path)
    pred = evaluate_on_validation(model, vd_tensor_input)
    pred_string = label_dict[int(pred)]
    
    return pred_string, pil_image

if __name__ == '__main__':
    # test code
    weights_path = r'C:\Final_project_Sesac\slow_fast_50\bfb_slowfast_r50.pth'
    video_path = r'C:\Final_project_Sesac\cut_videos\kinect_color\vp11\run1_2018-05-24-13-44-01_cut_6738.mp4'
    label_map_path = r'C:\Final_project_Sesac\cut_videos\label_map_midlevel.csv'

    label_dict = mapping_label(label_map_path, language = 'kor')