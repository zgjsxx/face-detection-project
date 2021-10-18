from flask import  Flask, request ,render_template
import cv2
import numpy as np
from detect_face_position import detect_face_position
from detect_face_key_point import detect_face_key_point
from detect_face_property import detect_face_property
from PIL import Image, ImageDraw, ImageFont
import json
import base64
from io import BytesIO
import re

fontStyle = ImageFont.truetype("SIMYOU.TTF", 40,encoding="utf-8")
app = Flask(__name__)

@app.route('/')
def index():
    return '<h1>Hello Face Detection!</h1>'

@app.route('/home')
def home():
    return render_template('face.html')

@app.route('/detect',methods=['POST'])
def detect():
    # print(request)
    # print(request.values)
    print("Image recieved")
    data_url = request.values['imageBase64']
    #print(data_url)
    tag = "base64,"
    index = data_url.find(tag)
    data_url = data_url[index+len(tag):]
    #print(data_url)
    # Decoding base64 string to bytes object
    image_data = base64.b64decode(data_url)
    file = open('0.png', 'wb')  # 保存为0.png的图片

    file.write(image_data)

    file.close()

    img = Image.open('0.png')  # 获取图片
    img_raw = img.copy()
    width = img_raw.size[0]
    height = img_raw.size[1]
    boxes, name_list = detect_face_position(img)
    final_result = {}
    if type(boxes) is not np.ndarray:
        final_result["bounding_box"] = None
        final_result["key_point"] = None
        final_result["face_property"] = None
        return json.dumps(final_result)

    face_key_point_list = detect_face_key_point(img_raw)
    face_property = detect_face_property(img)

    #print(face_key_point_list)
    draw = ImageDraw.Draw(img)
    for i, box in enumerate(boxes):
        draw.rectangle(box.tolist(), outline=(255, 0, 0))  # 绘制框
        draw.text( (int(box[0]),int(box[1])), str(name_list[i]), fill=(255,0,0),font=fontStyle)
    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

    for face_key_point in face_key_point_list:
        for p in range(68):
            cv2.circle(img, (int(face_key_point[p * 2] * width), int(face_key_point[p * 2 + 1] * height)),
                       2, (0, 255, 0), 2)

    print(face_property)
    # cv2.imshow("capture", img)
    # cv2.waitKey(0)

    print(boxes)
    print(face_key_point_list)
    print(face_property)
    final_result["bounding_box"] = boxes.tolist()
    final_result["key_point"] = face_key_point_list
    final_result["face_property"] = face_property
    final_result["name_list"] = name_list
    return json.dumps(final_result)
if __name__ == '__main__':
    app.run(debug=True)