import torchvision
from torchvision import models
import torch
import torch.nn as nn
from torch.optim import Optimizer
import math
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from data.data import get_loader


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = models.convnext_tiny()

        self.classifier = nn.Sequential(
            nn.LayerNorm([768, 1, 1], eps=1e-6, elementwise_affine=True),
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(768, 60)
        )

    def forward(self, x):
        x = self.model.features(x)
        x = self.model.avgpool(x)
        return self.classifier(x)


def get_cosine_schedule_with_warmup(
        optimizer: Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        num_cycles: float = 0.5,
        last_epoch: int = -1,
):
    def lr_lambda(now_step):
        if now_step < num_warmup_steps:  # 小于的话 学习率比单调递增
            return float(now_step) / float(max(1, num_warmup_steps))
        # 大于的话，换一个比例继续调整学习率
        progress = float(now_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def train(train_loader,
          valid_loader,
          lr=4e-3,
          weight_decay=0.05,
          total_epoch=100,
          label_smoothing=0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                num_warmup_steps=20,
                                                num_training_steps=total_epoch)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing).to(device)
    for epoch in range(1, total_epoch + 1):
        # train
        model.train()
        train_loss = 0
        train_acc = 0
        best_acc = 0
        for x, y in tqdm(train_loader):
            x = x.to(device)
            y = y.to(device)

            x = model(x)  # N, 60
            _, pre = torch.max(x, dim=1)
            train_acc += torch.sum(pre == y).item()
            loss = criterion(x, y)
            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss /= len(train_loader)
        train_acc /= len(train_loader)

        print(f'epoch {epoch}, train loss = {train_loss}, train acc = {train_acc}')

        # valid
        model.eval()
        with torch.no_grad():
            valid_loss = 0
            valid_acc = 0
            for x, y in tqdm(valid_loader):
                x = x.to(device)
                y = y.to(device)
                x = model(x)  # N, 60
                _, pre = torch.max(x, dim=1)
                valid_acc += torch.sum(pre == y).item()
                loss = criterion(x, y)
                valid_loss += loss.item()

            valid_loss /= len(valid_loader)
            valid_acc /= len(valid_loader)

            print(f'epoch {epoch}, valid loss = {valid_loss}, valid acc = {valid_acc}')

            if valid_acc > best_acc:
                best_acc = valid_acc
                torch.save(model.state_dict(), 'model.pth')

        scheduler.step()


if __name__ == '__main__':
    train_loader, valid_loader = get_loader(batch_size=1,
                                            train_image_path='./data/public_dg_0416/train/',
                                            valid_image_path='./data/public_dg_0416/train/',
                                            label2id_path='./data/dg_label_id_mapping.json')

    train(train_loader, valid_loader)
