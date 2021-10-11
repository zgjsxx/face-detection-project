import torch
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image
import numpy as np
from face_property_models.resnet_model import *
from detect_face_position import detect_face_position
import numpy
import cv2


labels = ['Male','Mouth_Slightly_Open']

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_path = r'pretrained_models/face_property/model-resnet-50-state.ptn'
model = ResNet50(class_num=2)
model.to(device)
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint)
model = model.eval()


def get_tensor(img):
    tfms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    return tfms(img).unsqueeze(0)


def predict(img, label_lst, model):
    tnsr = get_tensor(img)
    op = model(tnsr.to(device))

    op_b = torch.round(op)
    print(op_b)
    op_b_np = torch.Tensor.cpu(op_b).detach().numpy()

    preds = np.where(op_b_np == 1)[1]

    sigs_op = torch.Tensor.cpu(torch.round((op) * 100)).detach().numpy()[0]
    # o_p = np.argsort(torch.Tensor.cpu(op).detach().numpy())[0][::-1]
    # print(o_p)
    #
    # label = []
    # for i in preds:
    #     label.append(label_lst[i])
    #
    # arg_s = {}
    # for i in o_p:
    #     arg_s[label_lst[int(i)]] = sigs_op[int(i)]

    return sigs_op

def detect_face_property(img):
    width = img.size[0]
    height = img.size[1]
    boxes,name_list = detect_face_position(img)
    print(boxes.shape)
    if type(boxes) is not numpy.ndarray:
        return None,None
    face_property = []
    for box in boxes:
        crop_width = box[2] - box[0]
        crop_height = box[3] - box[1]

        face = img.crop((box[0], box[1], box[2], box[3]))

        result = predict(face, labels, model)
        result = result.tolist()
        face_property.append(result)
        face = cv2.cvtColor(numpy.asarray(face), cv2.COLOR_RGB2BGR)

        face = cv2.resize(face, (224, 224))
        # cv2.imshow("11", face)
        # cv2.waitKey(0)
    return face_property

if __name__=="__main__":
    img = Image.open(r".\test_img\test20.jpg")
    result = detect_face_property(img)
    print(result)
