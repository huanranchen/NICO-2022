import torch
import torch.nn as nn
from tqdm import tqdm
from train import Model
from data.dataUtils import write_result
from data.data import get_test_loader, get_loader


def test(loader, id2label, model_path='model.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    result = {}

    with torch.no_grad():
        for x, name in tqdm(loader):
            x = x.to(device)
            y = model(x)  # N, D
            _, y = torch.max(y, dim=1)  # (N,)

            for i, name in enumerate(list(name)):
                result[name] = y[i].item()

    write_result(result)


if __name__ == '__main__':
    loader, id2label = get_test_loader()
    # loader, _ = get_loader(batch_size=1,
    #                                         train_image_path='./data/public_dg_0416/train/',
    #                                         valid_image_path='./data/public_dg_0416/train/',
    #                                         label2id_path='./data/dg_label_id_mapping.json')
    test(loader, id2label, model_path='resnet18.pth')
