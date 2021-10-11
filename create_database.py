# 制作人脸特征向量的数据库 最后会保存两个文件，分别是数据库中的人脸特征向量和对应的名字。当然也可以保存在一起
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import numpy as np
import pandas as pd
import os
workers = 0 if os.name == 'nt' else 4
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))
mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device=device
)
# InceptionResnetV1提供了两个预训练模型，分别在vggface数据集和casia数据集上训练的。
# 预训练模型如果不手动下载，可能速度会很慢，可以从作者给的谷歌云链接下载，然后放到C:\Users\你的用户名\.cache\torch\checkpoints这个文件夹下面
# 如果是linux系统，那么存放在/home/你的用户名/.cache/torch/checkpoints下面
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def collate_fn(x):
    return x[0]
# 将所有的单人照图片放在各自的文件夹中，文件夹名字就是人的名字,存放格式如下
'''
--orgin
  |--zhangsan
     |--1.jpg
     |--2.jpg
  |--lisi
     |--1.jpg
     |--2.jpg
'''
dataset = datasets.ImageFolder('./face_recognition_raw_pic/origin')  #加载数据库
dataset.idx_to_class = {i:c for c, i in dataset.class_to_idx.items()}
loader = DataLoader(dataset, collate_fn=collate_fn, num_workers=workers)
aligned = []  # aligned就是从图像上抠出的人脸，大小是之前定义的image_size=160
names = []
i= 1
for x, y in loader:
    path = '.\\face_recognition_raw_pic\\aligned\\{}\\'.format(dataset.idx_to_class[y])  # 这个是要保存的人脸路径
    if not os.path.exists(path):
        i = 1
        os.mkdir(path)
    # 如果要保存识别到的人脸，在save_path参数指明保存路径即可,不保存可以用None
    x_aligned, prob = mtcnn(x, return_prob=True,save_path= path+ '/{}.jpg'.format(i))
    i = i+1
    if x_aligned is not None:
        print('Face detected with probability: {:8f}'.format(prob))
        aligned.append(x_aligned)
        names.append(dataset.idx_to_class[y])

aligned = torch.stack(aligned).to(device)
embeddings = resnet(aligned).detach().cpu()   # 提取所有人脸的特征向量，每个向量的长度是512
# 两两之间计算混淆矩阵
dists = [[(e1 - e2).norm().item() for e2 in embeddings] for e1 in embeddings]
print(names)
print(pd.DataFrame(dists, columns=names, index=names))
torch.save(embeddings, 'face_recognition_db/face_recognition_raw_pic.pt')  # 当然也可以保存在一个文件
torch.save(names, 'face_recognition_db/names.pt')
