from face_key_point_models.vgg_model import vgg
import torch
import numpy as np
import cv2
from detect_face_position import detect_face_position
from PIL import Image
import numpy
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

#load model
model_name = "vgg16"
net = vgg(model_name=model_name, num_classes=136, init_weights=True)
net.to(device)
model_dir= r"pretrained_models/face_key_point/face-keypoint-vgg16-0.pth"
checkpoint = torch.load(model_dir)
net.load_state_dict(checkpoint)

# predict use
net.eval()
def detect_face_key_point(img):
    boxes , name_list = detect_face_position(img)
    if type(boxes) is not numpy.ndarray:
        return None,None
    width = img.size[0]
    height = img.size[1]
    result = []
    for i, box in enumerate(boxes):
        print("boxes is ", box)
        crop_width = box[2] - box[0]
        crop_height = box[3] - box[1]
        face = img.crop((box[0], box[1], box[2], box[3]))
        face = cv2.cvtColor(numpy.asarray(face), cv2.COLOR_RGB2BGR)
        face_resize = cv2.resize(face, (224, 224))

        images_input = torch.from_numpy(face_resize)
        images_input = images_input.unsqueeze(0)
        images_input = np.transpose(images_input, (0, 3, 1, 2))
        images_input = images_input.float()
        outputs = net(images_input.to(device))

        outputs_val = torch.squeeze(outputs)

        #convert the proportion to the whole image
        for i in range(68):
            outputs_val[i*2] = (outputs_val[i*2]*crop_width + box[0])/width
            outputs_val[i*2+1] = (outputs_val[i*2+1]*crop_height + box[1])/height

        result.append(outputs_val.tolist())

    return result

if __name__=="__main__":
    img = Image.open(r".\test_img\test10.jpg")
    width = img.size[0]
    height = img.size[1]
    face_key_point_list = detect_face_key_point(img)
    print(face_key_point_list)

    img = cv2.cvtColor(numpy.asarray(img), cv2.COLOR_RGB2BGR)
    for face_key_point in face_key_point_list:
        for p in range(68):
            cv2.circle(img, (int(face_key_point[p * 2] * width), int(face_key_point[p * 2 + 1] * height)),
                       2, (0, 255, 0), 2)

    cv2.imshow("11", img)
    cv2.waitKey(0)



