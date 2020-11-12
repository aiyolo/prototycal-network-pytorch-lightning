import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import pytorch_lightning as pl
from argparse import ArgumentParser, Namespace
from dataset.mini_imagenet import MiniImageNet
from dataset.sampler import CategoriesSampler
from utils.metric import euclidean_metric
from model.convnet import Convnet
from pytorch_lightning.metrics.functional.classification import accuracy
import yaml


class LitModel(pl.LightningModule):
    def __init__(self, backbone, hparams):
        super().__init__()
        self.hparams = hparams
        self.backbone = backbone

    def forward(self, x):
        embedding = self.backbone(x)
        return embedding

    def training_step(self, batch, batch_idx):
        data, y = batch
        p = self.hparams.shot * self.hparams.train_way
        data_shot, data_query = data[:p], data[p:]
        proto = self(data_shot)
        proto = proto.reshape(self.hparams.shot, self.hparams.train_way, -1).mean(dim=0)

        label = torch.arange(self.hparams.train_way, device=self.device).repeat(self.hparams.query)

        logits = euclidean_metric(self(data_query), proto)
        loss = F.cross_entropy(logits, label)
        pred = torch.argmax(logits, dim=1)
        acc = accuracy(pred, label)

        self.log_dict({'train_loss': loss, 'train_acc': acc}, prog_bar=True,on_epoch=True)
        return loss


    def validation_step(self, batch, batch_idx):
        data, y = batch
        p = self.hparams.shot * self.hparams.test_way
        data_shot, data_query = data[:p], data[p:]
        proto = self(data_shot)
        proto = proto.reshape(self.hparams.shot, self.hparams.test_way, -1).mean(dim=0)

        label = torch.arange(self.hparams.test_way, device=self.device).repeat(self.hparams.query)

        logits = euclidean_metric(self(data_query), proto)
        loss = F.cross_entropy(logits, label)
        pred = torch.argmax(logits, dim=1)
        acc = accuracy(pred, label)

        self.log_dict({'valid_loss': loss, 'val_acc': acc}, prog_bar=True, on_step=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
        return [optimizer], [lr_scheduler]


def main():
    config = yaml.load(open('config/default.yaml'), Loader=yaml.FullLoader)
    args = Namespace(**config)

    trainset = MiniImageNet('train')
    valset = MiniImageNet('val')

    train_sampler = CategoriesSampler(trainset.label, 100,
                                      args.train_way, args.shot + args.query)
    train_loader = DataLoader(dataset=trainset, batch_sampler=train_sampler,num_workers=12, pin_memory=True)

    val_sampler = CategoriesSampler(valset.label, 400,
                                    args.test_way, args.shot + args.query)
    val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler,
                            num_workers=12, pin_memory=True)

    convnet = Convnet()
    model = LitModel(convnet, args)
    trainer = pl.Trainer(max_epochs=args.max_epochs, gpus=[0])
    trainer.fit(model, train_loader, val_loader)

if __name__=='__main__':
    main()
