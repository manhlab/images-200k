import random
from typing import Callable, Tuple

from kornia import augmentation as aug
from kornia import filters
from kornia.geometry import transform as tf
import torch
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2
import numpy as np
from copy import deepcopy
from itertools import chain
from typing import Dict, List

import pytorch_lightning as pl
from torch import optim
import torch.nn.functional as f
from pathlib import Path

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Union
from torch.optim import Adam
from os import cpu_count
from torch.optim import Adam
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers
import torch

class RandomApply(nn.Module):
    def __init__(self, fn: Callable, p: float):
        super().__init__()
        self.fn = fn
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        return x if random.random() > self.p else self.fn(x)


def default_augmentation(image_size: Tuple[int, int] = (224, 224)) -> nn.Module:
    return nn.Sequential(
        tf.Resize(size=image_size),
        RandomApply(aug.ColorJitter(0.8, 0.8, 0.8, 0.2), p=0.8),
        aug.RandomGrayscale(p=0.2),
        aug.RandomHorizontalFlip(),
        RandomApply(filters.GaussianBlur2d((3, 3), (1.5, 1.5)), p=0.1),
        aug.RandomResizedCrop(size=image_size),
        aug.Normalize(
            mean=torch.tensor([0.485, 0.456, 0.406]),
            std=torch.tensor([0.229, 0.224, 0.225]),
        ),
    )


def mlp(dim: int, projection_size: int = 256, hidden_size: int = 4096) -> nn.Module:
    return nn.Sequential(
        nn.Linear(dim, hidden_size),
        nn.BatchNorm1d(hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, projection_size),
    )


class EncoderWrapper(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        projection_size: int = 256,
        hidden_size: int = 4096,
        layer: Union[str, int] = -2,
    ):
        super().__init__()
        self.model = model
        self.projection_size = projection_size
        self.hidden_size = hidden_size
        self.layer = layer

        self._projector = None
        self._projector_dim = None
        self._encoded = torch.empty(0)
        self._register_hook()

    @property
    def projector(self):
        if self._projector is None:
            self._projector = mlp(
                self._projector_dim, self.projection_size, self.hidden_size
            )
        return self._projector

    def _hook(self, _, __, output):
        output = output.flatten(start_dim=1)
        if self._projector_dim is None:
            self._projector_dim = output.shape[-1]
        self._encoded = self.projector(output)

    def _register_hook(self):
        if isinstance(self.layer, str):
            layer = dict([*self.model.named_modules()])[self.layer]
        else:
            layer = list(self.model.children())[self.layer]

        layer.register_forward_hook(self._hook)

    def forward(self, x: Tensor) -> Tensor:
        _ = self.model(x)
        return self._encoded



# import utils
# from model import Model

BATCH_SIZE = 128
tau_plus = 0.1
temperature= 0.5
debiased = True
feature_dim = 256

def get_negative_mask(batch_size):
    negative_mask = torch.ones((batch_size, 2 * batch_size), dtype=bool)
    for i in range(batch_size):
        negative_mask[i, i] = 0
        negative_mask[i, i + batch_size] = 0

    negative_mask = torch.cat((negative_mask, negative_mask), 0)
    return negative_mask

def normalized_mse(x: Tensor, y: Tensor) -> Tensor:
    x = f.normalize(x, dim=-1)
    y = f.normalize(y, dim=-1)
    return 2 - 2 * (x * y).sum(dim=-1)


class BYOL(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        image_size: Tuple[int, int] = (128, 128),
        hidden_layer: Union[str, int] = -2,
        projection_size: int = 256,
        hidden_size: int = 4096,
        augment_fn: Callable = None,
        batch_size: int = 128,
        beta: float = 0.999,
    ):
        super().__init__()
        self.augment = default_augmentation(image_size) if augment_fn is None else augment_fn
        self.beta = beta
        self.encoder = EncoderWrapper(
            model, projection_size, hidden_size, layer=hidden_layer
        )
        self.predictor = nn.Linear(projection_size, projection_size, hidden_size)
        self._target = None
        self.batch_size = batch_size

        self.encoder(torch.zeros(2, 3, *image_size))

    def forward(self, x: Tensor) -> Tensor:
        return self.predictor(self.encoder(x))

    @property
    def target(self):
        if self._target is None:
            self._target = deepcopy(self.encoder)
        return self._target

    def update_target(self):
        for p, pt in zip(self.encoder.parameters(), self.target.parameters()):
            pt.data = self.beta * pt.data + (1 - self.beta) * p.data

    # --- Methods required for PyTorch Lightning only! ---

    def configure_optimizers(self):
        optimizer =Adam(self.parameters(), lr= 1e-4, weight_decay=1e-6, )
        # optimizer = LARS(
        #         self.parameters(),
        #         lr=1e-3,
        #         momentum=0.9,
        #         weight_decay=1e-6,
        #         trust_coefficient=0.001,
        #     )
        return optimizer

    def training_step(self, batch, *_) -> Dict[str, Union[Tensor, Dict]]:
        x = batch
        with torch.no_grad():
            x1, x2 = self.augment(x), self.augment(x)
        out_1, out_2 = self.forward(x1), self.forward(x2)
        with torch.no_grad():
            targ1, targ2 = self.target(x1), self.target(x2)

        out = torch.cat([out_1, out_2], dim=0)
        neg = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
        mask = get_negative_mask(self.batch_size).cuda()
        neg = neg.masked_select(mask).view(2 * self.batch_size, -1)

        # pos score
        pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
        pos = torch.cat([pos, pos], dim=0)

        # estimator g()
        if debiased:
            N = self.batch_size * 2 - 2
            Ng = (-tau_plus * N * pos + neg.sum(dim = -1)) / (1 - tau_plus)
            # constrain (optional)
            Ng = torch.clamp(Ng, min = N * np.e**(-1 / temperature))
        else:
            Ng = neg.sum(dim=-1)

        # contrastive loss
        loss = (- torch.log(pos / (pos + Ng) )).mean()
        # loss = torch.mean(normalized_mse(pred1, targ2) + normalized_mse(pred2, targ1))
        self.log("train_loss", loss.item())
        self.update_target()

        return {"loss": loss}

    @torch.no_grad()
    def validation_step(self, batch, *_) -> Dict[str, Union[Tensor, Dict]]:
        x = batch
        with torch.no_grad():
            x1, x2 = self.augment(x), self.augment(x)
        out_1, out_2 = self.forward(x1), self.forward(x2)
        with torch.no_grad():
            targ1, targ2 = self.target(x1), self.target(x2)

        out = torch.cat([out_1, out_2], dim=0)
        neg = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
        mask = get_negative_mask(BATCH_SIZE).cuda()
        neg = neg.masked_select(mask).view(2 * BATCH_SIZE, -1)

        # pos score
        pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
        pos = torch.cat([pos, pos], dim=0)

        # estimator g()
        if debiased:
            N = BATCH_SIZE * 2 - 2
            Ng = (-tau_plus * N * pos + neg.sum(dim = -1)) / (1 - tau_plus)
            # constrain (optional)
            Ng = torch.clamp(Ng, min = N * np.e**(-1 / temperature))
        else:
            Ng = neg.sum(dim=-1)
        # contrastive loss
        loss = (- torch.log(pos / (pos + Ng) )).mean()
        return {"loss": loss}

    @torch.no_grad()
    def validation_epoch_end(self, outputs: List[Dict]) -> Dict:
        val_loss = sum(x["loss"] for x in outputs) / len(outputs)
        self.log("val_loss", val_loss.item())


class ImageLoader(Dataset):
  def __init__(self, path, transform=None):
    self.path = path
    self.transform = transform
  def __len__(self):
    return len(self.path)
  def __getitem__(self, index):
    img =Image.open(self.path[index])
    if self.transform:
        if random.random() < 0.4:
            img = self.transform(img)
    img = np.array(img.convert('RGB'))
    img = cv2.resize(img, (96, 96), interpolation = cv2.INTER_AREA)
    img = torch.tensor(np.moveaxis(img, 2, 0)).float()/256
    return img

if "__name__"== "__main__":
    temp_dir  ="/content/data/data_origin"
    image_paths = [p for p in Path(temp_dir).rglob('*.jpg')]
    dataset = ImageLoader(image_paths)
    train_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        num_workers=4,
        drop_last=True,
    )
    torch.hub.list('zhanghang1989/ResNeSt', force_reload=True)
    # load pretrained models, using ResNeSt-50 as an example
    model = torch.hub.load('zhanghang1989/ResNeSt', 'resnest50', pretrained=True)
    # model = resnet18(pretrained=True)
    byol = BYOL(model, image_size=(96, 96), batch_size=BATCH_SIZE)
    dirpath = "/content/gdrive/MyDrive/Kaggle/image-200k"
    checkpoint_callback = ModelCheckpoint(dirpath=dirpath, filename='5-9part-resnest50-{epoch:02d}-{loss:.2f}')
    tb_logger = pl_loggers.TensorBoardLogger(dirpath)
    trainer = pl.Trainer(
        logger=tb_logger,
        max_epochs=50, 
        gpus=1,
        weights_summary=None,
        # fast_dev_run=True,
        # resume_from_checkpoint=ckpt,
        limit_val_batches=0.01,
        precision=16,
        callbacks=[checkpoint_callback]
    )
    trainer.fit(byol, train_loader, val_loader)
