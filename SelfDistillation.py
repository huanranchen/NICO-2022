import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from tqdm import tqdm
import os
from pyramidnet import pyramidnet272


class SelfDistillation():
    """
    核心思想，teacher蒸馏student，teacher EMA更新但有ground truth所以和BYOL那种靠不同样本间互相预测来学习内部特征不同，
    这个更像是supervised distillation的连续版本，并且EMA的momentum越小越倾向于硬标签
    其实也是个SAM
    """

    def __init__(self, train_loader, test_loader_predict,
                 lr=0.1, weight_decay=1e-4, test_loader_student=None
                 ):
        self.momentum = 0.9
        self.lr = lr
        self.train_loader = train_loader
        self.test_loader_predict = test_loader_predict
        self.test_loader_student = test_loader_student
        self.result = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.student = pyramidnet272(num_classes=60).to(self.device)
        self.teacher = pyramidnet272(num_classes=60).to(self.device)
        self.load_model()

        self.optimizer = torch.optim.SGD(self.student.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)

        self.teacher.load_state_dict(self.student.state_dict())
        for s in self.teacher.parameters():
            s.requires_grad = False
        self.teacher.eval()

    def load_model(self):
        if os.path.exists('teacher.pth'):
            self.teacher.load_state_dict(torch.load('teacher.pth', map_location=self.device))
            print('managed to load teacher')
            print('-' * 100)
        if os.path.exists('student.pth'):
            self.student.load_state_dict(torch.load('student.pth', map_location=self.device))
            print('managed to load student')
            print('-' * 100)

    @torch.no_grad()
    def momentum_sychronize(self, m=0.999):
        for s, t in zip(self.student.parameters(), self.teacher.parameters()):
            t.data = m * t.data + (1 - m) * s.data

    @staticmethod
    def chr_loss(s_x, t_x, label, alpha=1, beta=0.2, t=5):
        return alpha * F.cross_entropy(s_x, label) + \
               beta * F.kl_div(F.log_softmax(s_x / t, dim=1), F.softmax(t_x / t, dim=1),
                               reduction='batchmean')

    @torch.no_grad()
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
            print('student are giving his predictions!')
            self.student.eval()
            for x, names in tqdm(self.test_loader_predict):
                x = x.to(self.device)
                x = self.student(x)
                for i, name in enumerate(list(names)):
                    self.result[name] = x[i, :].unsqueeze(0)  # 1, D

            print('student have given his predictions!')
            print('-' * 100)

    def train(self,
              total_epoch=3,
              label_smoothing=0.2,
              fp16=True,
              warmup_epoch=1,
              warmup_cycle=12000,
              ):
        from torch.cuda.amp import autocast, GradScaler
        scaler = GradScaler()
        prev_loss = 999
        train_loss = 99
        criterion = self.chr_loss
        self.teacher.eval()
        for epoch in range(1, total_epoch + 1):
            # first, predict
            #             self.predict()
            #             self.result = self.save_result(epoch)

            self.student.train()
            self.warm_up(epoch, now_loss=train_loss, prev_loss=prev_loss)
            prev_loss = train_loss
            train_loss = 0
            train_acc = 0
            step = 0
            pbar = tqdm(self.train_loader)
            for x, y in pbar:
                x = x.to(self.device)
                y = y.to(self.device)

                with torch.no_grad():
                    x_t = self.teacher(x)
                if fp16:
                    with autocast():
                        x_s = self.student(x)  # N, 60
                        _, pre = torch.max(x_s, dim=1)
                        loss = criterion(x_s, x_t, y)
                else:
                    raise NotImplementedError
                    x_s = self.student(x)  # N, 60
                    _, pre = torch.max(x, dim=1)
                    loss = criterion(x_s, x_t, y)

                if pre.shape != y.shape:
                    _, y = torch.max(y, dim=1)
                train_acc += (torch.sum(pre == y).item()) / y.shape[0]
                train_loss += loss.item()
                self.optimizer.zero_grad()

                if fp16:
                    scaler.scale(loss).backward()
                    scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_value_(self.student.parameters(), 0.1)
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_value_(self.student.parameters(), 0.1)
                    self.optimizer.step()

                step += 1
                # scheduler.step()
                if step % 10 == 0:
                    pbar.set_postfix_str(f'loss = {train_loss / step}, acc = {train_acc / step}')

                self.momentum_sychronize()

            train_loss /= len(self.train_loader)
            train_acc /= len(self.train_loader)

            print(f'epoch {epoch}, test loader loss = {train_loss}, acc = {train_acc}')

            self.save_model()

    def warm_up(self, epoch, now_loss=None, prev_loss=None):
        if epoch <= 10:
            self.optimizer.param_groups[0]['lr'] = self.lr * epoch / 10
        elif now_loss is not None and prev_loss is not None:
            delta = prev_loss - now_loss
            if delta / now_loss < 0.02 and delta < 0.02:
                self.optimizer.param_groups[0]['lr'] *= 0.9

        p_lr = self.optimizer.param_groups[0]['lr']
        print(f'lr = {p_lr}')

    @torch.no_grad()
    def TTA(self, total_epoch=10, aug_weight=0.5):
        self.predict()
        print('now we are doing TTA')
        for epoch in range(1, total_epoch + 1):
            self.student.eval()
            for x, names in tqdm(self.test_loader_student):
                x = x.to(self.device)
                x = self.student(x)
                for i, name in enumerate(list(names)):
                    self.result[name] += x[i, :].unsqueeze(0) * aug_weight  # 1, D

        print('TTA finished')
        self.save_result()
        print('-' * 100)

    def save_model(self):
        torch.save(self.student.state_dict(), 'student.pth')
        torch.save(self.teacher.state_dict(), 'teacher.pth')


if __name__ == '__main__':
    train_image_path = './public_dg_0416/train/'
    valid_image_path = './public_dg_0416/train/'
    label2id_path = './dg_label_id_mapping.json'
    test_image_path = './public_dg_0416/public_test_flat/'
    from data.data import get_loader, get_test_loader
    import argparse

    paser = argparse.ArgumentParser()
    paser.add_argument('-b', '--batch_size', default=2)
    paser.add_argument('-t', '--total_epoch', default=10)
    paser.add_argument('-l', '--lr', default=1e-4)
    args = paser.parse_args()
    batch_size = int(args.batch_size)
    total_epoch = int(args.total_epoch)
    lr = float(args.lr)

    train_loader = get_loader(batch_size=batch_size,
                              valid_category=None,
                              train_image_path=train_image_path,
                              valid_image_path=valid_image_path,
                              label2id_path=label2id_path)
    test_loader_predict, _ = get_test_loader(batch_size=batch_size,
                                             transforms=None,
                                             label2id_path=label2id_path,
                                             test_image_path=test_image_path)
    test_loader_student, label2id = get_test_loader(batch_size=batch_size,
                                                    transforms='train',
                                                    label2id_path=label2id_path,
                                                    test_image_path=test_image_path)

    M = SelfDistillation(train_loader, test_loader_predict, test_loader_student=test_loader_student,
                         lr=lr)
    M.train(total_epoch=total_epoch)
