from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy
import torch
# If required, create a face detection pipeline using MTCNN:
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=True, device=device)
# Create an inception resnet (in eval mode):
resnet = InceptionResnetV1(pretrained='vggface2').eval().to('cuda')
names = torch.load("./face_recognition_db/names.pt")
embeddings = torch.load("./face_recognition_db/face_recognition_raw_pic.pt").to('cuda')
fontStyle = ImageFont.truetype("SIMYOU.TTF", 40,encoding="utf-8")

def detect_face_position(img):
    faces = mtcnn(img)
    boxes, _ = mtcnn.detect(img)  # 检测出人脸框 返回的是位置
    if type(boxes) is not numpy.ndarray:
        return None,None

    #draw = ImageDraw.Draw(img)
    name_list = []
    # if boxes is None:
    #     return None,None
    #print(boxes)
    for i, box in enumerate(boxes):
        #draw.rectangle(box.tolist(), outline=(255, 0, 0))  # 绘制框
        face_embedding = resnet(faces[i].unsqueeze(0).to('cuda'))
        probs = [(face_embedding - embeddings[i]).norm().item() for i in range(embeddings.size()[0])]
        #print(probs)
        # 我们可以认为距离最近的那个就是最有可能的人，但也有可能出问题，数据库中可以存放一个人的多视角多姿态数据，对比的时候可以采用其他方法，如投票机制决定最后的识别人脸
        index = probs.index(min(probs))   # 对应的索引就是判断的人脸
        name = names[index] # 对应的人脸
        name_list.append(name)
    return boxes,name_list

if __name__ == "__main__":
    img = Image.open(r".\test_img\1.jpg")
    boxes,name_list = detect_face_position(img)

    draw = ImageDraw.Draw(img)
    for i, box in enumerate(boxes):
        draw.rectangle(box.tolist(), outline=(255, 0, 0))  # 绘制框
        draw.text( (int(box[0]),int(box[1])), str(name_list[i]), fill=(255,0,0),font=fontStyle)

    img = cv2.cvtColor(numpy.asarray(img),cv2.COLOR_RGB2BGR)
    cv2.imshow("capture", img)
    cv2.waitKey(0)

