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
import torch.nn.functional as F
from PIL import Image


class MixLoader():
    def __init__(self, iteraters, probs=None):
        '''

        :param iteraters: list of dataloader
        '''
        self.loaders = iteraters
        self.iteraters = [iter(loader) for loader in iteraters]
        self.index = [i for i, _ in enumerate(iteraters)]
        lens = [len(i) for i in self.iteraters]
        if probs is not None:
            self.probs = probs
        else:
            self.probs = self.compute_prob(lens)
        self.max_time = sum(lens)
        self.iteration_times = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.iteration_times >= self.max_time:
            self.iteration_times = 0
            raise StopIteration
        else:
            choice = np.random.choice(self.index, p=self.probs)
            try:
                data = next(self.iteraters[choice])
                self.iteration_times += 1
                return data
            except:
                self.iteraters[choice] = iter(self.loaders[choice])
                return self.__next__()

    @staticmethod
    def compute_prob(x):
        total = sum(x)
        result = [i / total for i in x]
        return result

    def __len__(self):
        return self.max_time


def smoothing_cross_entropy(x, y, ):
    '''
    -y log pre
    :param x: N, D
    :param y: N, D
    :return:
    '''
    if x.shape != y.shape:
        return F.cross_entropy(x, y)
    return F.kl_div(F.log_softmax(x, dim=1), y, reduction='batchmean')


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = models.resnet50(num_classes=60, )
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.apply(self._init_weights)
        print('mix' * 100)

    def forward(self, x):
        x = self.model(x)
        return x

    def load_model(self):
        start_state = torch.load('model.pth', map_location=self.device)
        self.model.load_state_dict(start_state['model_state'])
        # self.load_state_dict(start_state)
        print('using loaded model')
        print('-' * 100)

    def save_model(self):
        # result = self.model.state_dict()
        result = self.model.state_dict()
        torch.save(result, 'model.pth')


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


class NoisyStudent():
    def __init__(self,
                 batch_size=64,
                 lr=1e-4,
                 weight_decay=0,
                 train_image_path='./public_dg_0416/train/',
                 valid_image_path='./public_dg_0416/train/',
                 label2id_path='./dg_label_id_mapping.json',
                 test_image_path='./public_dg_0416/public_test_flat/'
                 ):
        self.result = {}
        from data.data import get_loader, get_test_loader
        self.train_loader = get_loader(batch_size=batch_size,
                                       valid_category=None,
                                       train_image_path=train_image_path,
                                       valid_image_path=valid_image_path,
                                       label2id_path=label2id_path)
        self.test_loader_predict, _ = get_test_loader(batch_size=batch_size,
                                                      transforms=None,
                                                      label2id_path=label2id_path,
                                                      test_image_path=test_image_path)
        self.test_loader_student, self.label2id = get_test_loader(batch_size=batch_size,
                                                                  transforms='train',
                                                                  label2id_path=label2id_path,
                                                                  test_image_path=test_image_path)
        self.train_loader = MixLoader([self.train_loader, self.test_loader_student],
                                      probs=[0.5, 0.5])
        del self.test_loader_student
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = Model().to(self.device)

        if os.path.exists('model.pth'):
            self.model.load_model()

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)

    def save_result(self, epoch=None):
        from data.data import write_result
        result = {}
        for name, pre in list(self.result.items()):
            _, y = torch.max(pre, dim=1)
            result[name] = y.item()

        if epoch is not None:
            write_result(result, path='prediction' + str(epoch) + '.json')
        else:
            write_result(result)

        return result

    def predict(self):
        with torch.no_grad():
            print('teacher are giving his predictions!')
            self.model.eval()
            for x, names in tqdm(self.test_loader_predict):
                x = x.to(self.device)
                x = self.model(x)
                x = F.softmax(x, dim=1)
                for i, name in enumerate(list(names)):
                    self.result[name] = x[i, :].unsqueeze(0)  # 1, D

            print('teacher have given his predictions!')
            print('-' * 100)

    def get_label(self, names):
        y = []
        for name in list(names):
            y.append(self.result[name])

        return torch.cat(y, dim=0).to(self.device)

    def train(self,
              total_epoch=3,
              label_smoothing=0.2,
              fp16_training=True,
              warmup_epoch=1,
              warmup_cycle=12000,
              ):
        # scheduler = get_cosine_schedule_with_warmup(optimizer,
        #                                             num_warmup_steps=len(train_loader)*warmup_epoch,
        #                                             num_training_steps=total_epoch*len(train_loader),
        #                                             num_cycles=warmup_cycle)
        criterion = smoothing_cross_entropy
        for epoch in range(1, total_epoch + 1):
            # first, predict
            self.model.eval()
            self.predict()
            self.save_result(epoch)

            self.model.train()
            train_loss = 0
            train_acc = 0
            step = 0
            pbar = tqdm(self.train_loader)
            for x, y in pbar:
                x = x.to(self.device)
                if isinstance(y, tuple):
                    y = self.get_label(y)
                y = y.to(self.device)
                x = self.model(x)  # N, 60
                _, pre = torch.max(x, dim=1)
                loss = criterion(x, y)
                if pre.shape != y.shape:
                    _, y = torch.max(y, dim=1)
                train_acc += (torch.sum(pre == y).item()) / y.shape[0]
                train_loss += loss.item()
                self.optimizer.zero_grad()
                loss.backward()

                nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
                self.optimizer.step()
                step += 1
                # scheduler.step()
                if step % 10 == 0:
                    pbar.set_postfix_str(f'loss = {train_loss / step}, acc = {train_acc / step}')

            train_loss /= len(self.train_loader)
            train_acc /= len(self.train_loader)

            print(f'epoch {epoch}, test loader loss = {train_loss}, acc = {train_acc}')

            self.model.save_model()


if __name__ == '__main__':
    x = NoisyStudent(batch_size=1)
    x.train(total_epoch=10)

    # train_image_path = './public_dg_0416/train/'
    # valid_image_path = './public_dg_0416/train/'
    # label2id_path = './dg_label_id_mapping.json'
    # test_image_path = './public_dg_0416/public_test_flat/'
    # from data.data import get_loader
    # train_loader = get_loader(batch_size=10,
    #                                valid_category=None,
    #                                train_image_path=train_image_path,
    #                                valid_image_path=valid_image_path,
    #                                label2id_path=label2id_path)
    # train(train_loader)
