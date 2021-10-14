
# face project

---
[![Python](https://img.shields.io/static/v1?label=build&message=passing&color=green)](https://www.python.org/)
[![Python](https://img.shields.io/static/v1?label=python&message=3.8.12&color=blue)](https://www.python.org/)
[![torch](https://img.shields.io/static/v1?label=torch&message=1.8.1&color=blue)](https://pytorch.org/)

# 1.introduction with the whole project
This project is a pytorch implementation of face detection, face recognition ,face key point detection 
and face property detection. The project includes the a simple UI and a flask-based backend system.

# 1.1 face detection and recognize
This will show the position of an face in the picture and mark it with a rectangle.
This part is using the MTCNN and facenet model provided by facenet_pytorch package.
We do not train this model by myself, just use it.

# 1.2. face key point detection
This part can detect 68 key points  of the face area. The face key point distrubution is as follows:
![img.png](document/pic/img1.png).

We use the [300W-LP](http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3ddfa/main.htm) database to train the model.
The train code can be found on my another github repo.

# 1.3. face property recognize
For this part,we use the model to recognize the property of the face. In this project, we use model to recognize
the man or woman and open mouth or not open mouth.
We train the model with [celeba1](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) database

# 2. how to run
First, you need to install the third-party packages
```
pip install -r requirements.txt
```
then simply run the below command
```
python3 main.py
```

input http://127.0.0.1/home
and then you can get the below result:
![img.png](document/pic/img2.png)




