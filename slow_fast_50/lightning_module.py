import torch
import json
import torch.nn.functional as F
import torch.nn as nn
from torchmetrics import Accuracy
import pytorch_lightning as pl

class VideoClassificationLightningModule(pl.LightningModule):
    def __init__(self, model, args):
        super().__init__()
        self.model = model
        self.args = args
        self.train_accuracy = Accuracy(task="multiclass", num_classes=args["num_classes"])
        self.val_accuracy = Accuracy(task="multiclass", num_classes=args["num_classes"])
        self.misclassified_samples = []  # 잘못 예측된 샘플을 저장할 리스트 초기화

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        assert isinstance(x, list), "Input data must be a list of tensors"
        assert len(x) == 2, "SlowFast model requires two pathways as input"

        y_hat = self.model(x)
        y_hat = y_hat.reshape(y_hat.size(0), -1)

        loss = F.cross_entropy(y_hat, y)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)

        # Use reshape instead of view
        y_hat = y_hat.reshape(y_hat.size(0), -1)

        loss = F.cross_entropy(y_hat, y)
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        y_hat = y_hat.reshape(y_hat.size(0), -1)
        
        loss = F.cross_entropy(y_hat, y)
        preds = torch.argmax(y_hat, dim=1)
        acc = self.val_accuracy(preds, y)

        # 잘못 예측된 샘플 저장
        misclassified_indices = (preds != y).nonzero(as_tuple=True)[0]
        for idx in misclassified_indices:
            self.misclassified_samples.append({
                'true_label': y[idx].item(),
                'predicted_label': preds[idx].item()
            })

        # 예측 레이블과 정답 레이블 출력
        for i in range(len(y)):
            print(f"Predicted Label: {preds[i].item()}, True Label: {y[i].item()}")

        self.log("test_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("test_acc", acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def on_test_epoch_end(self):
        # 잘못 예측된 샘플 출력
        if self.misclassified_samples:
            print("\n잘못 예측된 샘플 목록:")
            for sample in self.misclassified_samples:
                print(f"실제 레이블: {sample['true_label']}, 예측 레이블: {sample['predicted_label']}")

            # JSON 파일로 저장
            with open("misclassified_samples.json", "w") as f:
                json.dump(self.misclassified_samples, f, indent=4)
            print("\n잘못 예측된 샘플이 'misclassified_samples.json' 파일에 저장되었습니다.")
        else:
            print("\n모든 샘플이 정확하게 분류되었습니다.")
        # 리스트 초기화
        self.misclassified_samples.clear()

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