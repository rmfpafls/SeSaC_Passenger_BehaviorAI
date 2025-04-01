import random
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import pandas as pd
from torch.amp import autocast
from torchmetrics import Accuracy
from torch.utils.data import Dataset, DataLoader
from pytorchvideo.data.encoded_video import EncodedVideo
from sklearn.model_selection import train_test_split
from torchvision.io import read_video
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize
from pytorchvideo.transforms import UniformTemporalSubsample, ShortSideScale
from dataset import mapping_label
from PIL import Image

class VideoClassificationLightningModule(pl.LightningModule):
    def __init__(self, model, args):
        super().__init__()
        self.model = model
        self.args = args
        self.train_accuracy = Accuracy(task="multiclass", num_classes=args["num_classes"])
        self.val_accuracy = Accuracy(task="multiclass", num_classes=args["num_classes"])

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        assert isinstance(x, list), "Input data must be a list of tensors"
        assert len(x) == 2, "SlowFast model requires two pathways as input"

        # 데이터 타입 변환
        x = [pathway.float() for pathway in x]  # 모델 입력을 float32로 변환
        y = y.long()  # 라벨을 long 타입으로 변환

        y_hat = self.model(x)
        y_hat = y_hat.reshape(y_hat.size(0), -1)
        loss = F.cross_entropy(y_hat, y)
        preds = y_hat.argmax(dim=-1)
        acc = self.train_accuracy(preds, y)

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_acc", acc, prog_bar=True, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        # 데이터 타입 변환
        x = [pathway.float() for pathway in x]  # 모델 입력을 float32로 변환
        y = y.long()  # 라벨을 long 타입으로 변환

        y_hat = self.model(x)
        y_hat = y_hat.reshape(y_hat.size(0), -1)
        preds = y_hat.argmax(dim=-1)
        loss = F.cross_entropy(y_hat, y)
        acc = self.val_accuracy(preds, y)

        self.log("valid_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("valid_acc", acc, prog_bar=True, on_step=True, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch

        # 데이터 타입 변환
        x = [pathway.float() for pathway in x]  # 모델 입력을 float32로 변환
        y = y.long()  # 라벨을 long 타입으로 변환

        y_hat = self.model(x)
        y_hat = y_hat.reshape(y_hat.size(0), -1)
        preds = y_hat.argmax(dim=-1)
        loss = F.cross_entropy(y_hat, y)
        acc = self.val_accuracy(preds, y)

        self.log("test_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("test_acc", acc, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.args["lr"],
            momentum=self.args["momentum"],
            weight_decay=self.args["weight_decay"],
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, self.args["max_epochs"]
        )
        return [optimizer], [scheduler]


# 학습된 가중치 불러오기
def load_trained_model(model, weights_path):
    model.load_state_dict(torch.load(weights_path))
    return model


# Validation 정확도 계산 함수
def evaluate_on_validation(model, val_loader, device="cuda"):
    model.to(device)
    model.eval()  # 모델을 평가 모드로 설정
    correct = 0
    total = 0

    with torch.no_grad():  # 평가 시에는 그래디언트를 계산하지 않음
        for batch in val_loader:
            x, y = batch

            # 데이터와 라벨을 장치로 이동
            x = [pathway.to(device).float() for pathway in x]  # SlowFast 경로
            y = y.to(device).long()

            y_hat = model(x)
            y_hat = y_hat.reshape(y_hat.size(0), -1)
            preds = y_hat.argmax(dim=-1)
            correct += (preds == y).sum().item()
            total += y.size(0)

            print(f'pred =  {preds}, y =  {y}')

    if total > 0:
        accuracy = correct / total * 100  # 정확도 계산
    else:
        accuracy = 0.0  # 데이터가 없는 경우

    print(f"Validation Accuracy: {accuracy:.2f}%")
    return accuracy


# 비디오에서 액션 예측 함수
def predict_action(model, video_path, label_map_csv_path, device="cuda"):
    """
    Predicts the action label for a single video clip using a SlowFast model,
    and returns a random frame image from the clip along with the predicted label text.

    Args:
        model: The pretrained SlowFast model.
        video_path (str): Path to the video file.
        label_map_csv_path (str): Path to the CSV file containing label_index and activity.
        device (str): Device to run the model on (default: "cuda").

    Returns:
        str: The predicted action label text.
        PIL.Image.Image: A random frame image from the clip.
    """
    # Load the label map from the CSV file
    index_to_label = mapping_label(label_map_csv_path)

    model.to(device)
    model.eval()

    transform = Compose([
        ShortSideScale(224),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

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
    slow_pathway=slow_pathway.unsqueeze(0)
    fast_pathway =fast_pathway .unsqueeze(0)
    vd_tensor_input =[slow_pathway,fast_pathway]

    with torch.no_grad():  # 평가 시에는 그래디언트를 계산하지 않음
        video_tensor_ = [pathway.to(device) for pathway in vd_tensor_input]  # SlowFast 경로
        y_hat = model(video_tensor_)
        y_hat = y_hat.reshape(y_hat.size(0), -1)
        preds = y_hat.argmax(dim=-1)
        preds=preds.item()

    print(preds)

    # Map index to label
    predicted_label = index_to_label.get(preds, "Unknown")

    return predicted_label








