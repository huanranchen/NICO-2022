import os
import numpy as np
import torchvision
from torchvision import models
import torch
import torch.nn as nn
from torch.optim import Optimizer
import math
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from data.data import get_loader
from torch.cuda.amp import autocast, GradScaler
from PIL import Image
import torch.nn.functional as F
from convnext import convnext_tiny, LayerNorm


def hook(module, fea_in, fea_out):
    for name, param in module.named_modules():
        print('-' * 100)
        print(param, )
        break
    # if linear , N, ?, ?, C
    # print(f'in std = {fea_in[0].std()}, out = {fea_out[0].std()}')

    print(fea_in[0], fea_out[0])


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = models.resnet50(num_classes=60, )

        # self.apply(self._init_weights)
        print('resnet' * 100)

    def forward(self, x):
        x = self.model(x)
        return x

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
          valid_loader = None,
          lr=4e-3,
          weight_decay=0.05,
          total_epoch=120,
          label_smoothing=0.2,
          fp16_training=True,
          warmup_epoch=1,
          warmup_cycle=12000,
          ):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    # scheduler = get_cosine_schedule_with_warmup(optimizer,
    #                                             num_warmup_steps=len(train_loader)*warmup_epoch,
    #                                             num_training_steps=total_epoch*len(train_loader),
    #                                             num_cycles=warmup_cycle)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing).to(device)

    if os.path.exists('model.pth'):
        model.load_state_dict(torch.load('model.pth', map_location=device))
        print('using loaded model')
        print('-' * 100)

    best_acc = 0
    for epoch in range(1, total_epoch + 1):
        # train
        model.train()
        train_loss = 0
        train_acc = 0
        step = 0
        pbar = tqdm(train_loader)

        for x, y in pbar:

            x = x.to(device)
            y = y.to(device)

            # x=x.permute(0,2,3,1)
            # x = x[0].numpy() *255
            # print(x.shape)
            # x = Image.fromarray(np.uint8(x))
            # x.show()
            # assert False
            # x = model.model.features(x)
            # # x = model.model.avgpool(x)
            # torch.set_printoptions(threshold=np.inf)
            # a,p = torch.max(x, dim = 1)
            # print(a,p)
            # print(x[0],x[0,718,0,0])
            # assert False
            # for name, m in model.named_modules():
            #     print(name)
            #     if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) or isinstance(m, LayerNorm):
            #         m.register_backward_hook(hook)

            x = model(x)  # N, 60
            _, pre = torch.max(x, dim=1)
            train_acc += (torch.sum(pre == y).item()) / y.shape[0]
            loss = criterion(x, y)
            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()

            # if step == 0:
            #     for name, param in model.named_parameters():
            #         print(name,torch.sum(param.data**2), torch.sum(param.grad ** 2))
            #assert False
            # if step == 0:
            #
            #     for name, w in model.named_parameters():
            #         print(name,torch.sum(torch.abs(w)),torch.sum(w).item())
            # assert False

            nn.utils.clip_grad_value_(model.parameters(), 0.1)
            optimizer.step()
            step += 1
            # scheduler.step()

            if step % 10 == 0:
                pbar.set_postfix_str(f'loss = {train_loss / step}, acc = {train_acc / step}')

        train_loss /= len(train_loader)
        train_acc /= len(train_loader)

        print(f'epoch {epoch}, train loss = {train_loss}, train acc = {train_acc}')

        # valid
        if valid_loader is not None:
            model.eval()
            with torch.no_grad():
                valid_loss = 0
                valid_acc = 0
                for x, y in tqdm(valid_loader):
                    x = x.to(device)
                    y = y.to(device)
                    x = model(x)  # N, 60
                    _, pre = torch.max(x, dim=1)
                    valid_acc += (torch.sum(pre == y).item()) / y.shape[0]
                    loss = criterion(x, y)
                    valid_loss += loss.item()

                valid_loss /= len(valid_loader)
                valid_acc /= len(valid_loader)

                print(f'epoch {epoch}, valid loss = {valid_loss}, valid acc = {valid_acc}')

                if valid_acc > best_acc:
                    best_acc = valid_acc
                    torch.save(model.state_dict(), 'model.pth')
        else:
            torch.save(model.state_dict(), 'model.pth')


def noisy_student(train_loader,
          test_loader,
          lr=4e-3,
          weight_decay=0.05,
          total_epoch=120,
          label_smoothing=0.2,
          fp16_training=True,
          warmup_epoch=1,
          warmup_cycle=12000,
          ):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    # scheduler = get_cosine_schedule_with_warmup(optimizer,
    #                                             num_warmup_steps=len(train_loader)*warmup_epoch,
    #                                             num_training_steps=total_epoch*len(train_loader),
    #                                             num_cycles=warmup_cycle)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing).to(device)

    if os.path.exists('model.pth'):
        model.load_state_dict(torch.load('model.ckpt', map_location=device))
        print('using loaded model')
        print('-' * 100)

    for epoch in range(1, total_epoch + 1):
        # train
        model.train()
        train_loss = 0
        train_acc = 0
        step = 0
        pbar = tqdm(train_loader)

        for x, y in pbar:

            x = x.to(device)
            y = y.to(device)
            x = model(x)  # N, 60
            _, pre = torch.max(x, dim=1)
            train_acc += (torch.sum(pre == y).item()) / y.shape[0]
            loss = criterion(x, y)
            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()

            nn.utils.clip_grad_value_(model.parameters(), 0.1)
            optimizer.step()
            step += 1
            # scheduler.step()
            if step % 10 == 0:
                pbar.set_postfix_str(f'loss = {train_loss / step}, acc = {train_acc / step}')

        train_loss /= len(train_loader)
        train_acc /= len(train_loader)

        print(f'epoch {epoch}, train loss = {train_loss}, train acc = {train_acc}')




if __name__ == '__main__':
    train_loader, valid_loader = get_loader(batch_size=10,
                                            train_image_path='./public_dg_0416/train/',
                                            valid_image_path='./public_dg_0416/train/',
                                            label2id_path='./dg_label_id_mapping.json')

    train(train_loader, valid_loader)
