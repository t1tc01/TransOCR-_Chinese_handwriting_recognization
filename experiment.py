import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
from model.transocr import Transformer
from utils import get_data_package, converter, tensor2str, get_alphabet
import zhconv
import cv2 as cv

from data.dataset import lmdbDataset, resizeNormalize
import pickle as pkl
import torchvision.transforms as transforms
from PIL import Image
import gc

class resizeNormalize(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img

IMAGE_H = 32
IMAGE_W = 256


parser = argparse.ArgumentParser(description='')
parser.add_argument('--exp_name', type=str, default="test", help='')
parser.add_argument('--image_path', type=str, default="", help='')
parser.add_argument('--batch_size', type=int, default=32, help='')
parser.add_argument('--lr', type=float, default=1.0, help='')
parser.add_argument('--epoch', type=int, default=1000, help='')
parser.add_argument('--radical', action='store_true', default=False)
parser.add_argument('--test', action='store_true', default=False)
parser.add_argument('--resume', type=str, default='', help='')
parser.add_argument('--train_dataset', type=str, default='', help='')
parser.add_argument('--test_dataset', type=str, default='', help='')
parser.add_argument('--imageH', type=int, default=32, help='')
parser.add_argument('--imageW', type=int, default=256, help='')
parser.add_argument('--coeff', type=float, default=1.0, help='')
parser.add_argument('--alpha_path', type=str, default='./data/benchmark.txt', help='')
parser.add_argument('--alpha_path_radical', type=str, default='./data/radicals.txt', help='')
parser.add_argument('--decompose_path', type=str, default='./data/decompose.txt', help='')
args = parser.parse_args()

PATH = args.image_path

alphabet = get_alphabet(args, 'char')
print('Number of characters: ',len(alphabet))

model = Transformer(args).cuda()
model = nn.DataParallel(model)


model.load_state_dict(torch.load(args.resume))
print('loading pretrained model！！！')

torch.cuda.empty_cache()

@torch.no_grad()
def test(epoch):

    torch.cuda.empty_cache()

    print("Start Eval!")
    model.eval()

    for iteration in range(1):
        image = Image.open(PATH).convert('RGB')
        transform = resizeNormalize((args.imageW, args.imageH))
        image = transform(image)
        image = image.unsqueeze(0)
        image = torch.nn.functional.interpolate(image, size=(args.imageH, args.imageW))
        max_length = 17
        
        # print(image.shape) #torch.Size([3, 32, 256])
        # print(type(image)) #torch.Tensor
        # print(type(max_length)) #17 torch.Tensor

        batch = image.shape[0]
        pred = torch.zeros(batch,1).long().cuda()
        image_features = None
        prob = torch.zeros(batch, max_length).float()
        for i in range(max_length):
            length_tmp = torch.zeros(batch).long().cuda() + i + 1
            result = model(image, length_tmp, pred, conv_feature=image_features, test=True)

            prediction = result['pred']
            now_pred = torch.max(torch.softmax(prediction,2), 2)[1]
            prob[:,i] = torch.max(torch.softmax(prediction,2), 2)[0][:,-1]
            pred = torch.cat((pred, now_pred[:,-1].view(-1,1)), 1)
            image_features = result['conv']

        text_pred_list = []
        for i in range(batch):
            now_pred = []
            for j in range(max_length):
                if pred[i][j] != len(alphabet) - 1:
                    now_pred.append(pred[i][j])
                else:
                    break
            text_pred_list.append(torch.Tensor(now_pred)[1:].long().cuda())

        for i in range(batch):
            state = False
            pred = zhconv.convert(tensor2str(text_pred_list[i], args),'zh-cn')
            print('{}'.format(pred))

if __name__ == '__main__':
    print('-------------')
    test(-1)
    exit(0)