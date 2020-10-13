#!/usr/bin/env python
# coding: utf-8

# In[1]:


import dlib # 영상처리 알고리즘
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np


# In[2]:


detector = dlib.get_frontal_face_detector() # detector 얼굴을 영역을 찾아주는 변수
sp = dlib.shape_predictor(
    './models/shape_predictor_5_face_landmarks.dat') # 얼굴의 눈코입을 찾아주는 모델


# In[3]:


img = dlib.load_rgb_image('./imgs/12.jpg')
plt.figure(figsize=(16,10))
plt.imshow(img)
plt.show()


# In[4]:


# Patch 얼굴 영역 찾기
img_result = img.copy()
dets = detector(img,1)
if len(dets) == 0:
    print('cannot find faces!')
else:
    fig, ax = plt.subplots(1, figsize=(16,10)) # fig, ax를 subplot들로 받는다
    for det in dets:
        x, y, w, h = det.left(), det.top(), det.width(), det.height()
        rect = patches.Rectangle((x,y),w,h,linewidth =2, edgecolor ='r', facecolor = 'none')
        ax.add_patch(rect) # 상자를 띄워주는 것
    ax.imshow(img_result)
    plt.show()


# In[5]:


# Landmark 눈, 코, 입을 찾아주는 것
fig, ax = plt.subplots(1, figsize=(16,10))
objs = dlib.full_object_detections() # 삐뚫어진 얼굴을 돌려주눈 기능
for detection in dets:
    s = sp(img, detection)
    objs.append(s)
    for point in s.parts():
        circle = patches.Circle((point.x, point.y),radius=3, edgecolor='r', facecolor='r')
        ax.add_patch(circle)
ax.imshow(img_result)
        


# In[6]:


faces = dlib.get_face_chips(img, objs, size=256, padding=0.3) # padding 이미지 사이의 간격을 띄어주기 위해서
fig, axes = plt.subplots(1, len(faces)+1, figsize=(20,16))
axes[0].imshow(img)
for i, face in enumerate(faces):
    axes[i+1].imshow(face)


# In[7]:


# 변수 생성 
def align_faces(img): # 얼굴 이미지들을 return 해주는 함수
    dets = detector(img, 1) # 이미지 속 얼굴 정보 저장되어 있는 곳
    objs = dlib.full_object_detections() # object에 대한 정보가 들어있는 객체 (위치 정보)
    for detection in dets: # dets - 얼굴 영역 정보
        s = sp(img, detection) # s = landmark 정보를 갖고있음
        objs.append(s)
    faces = dlib.get_face_chips(img, objs, size=256, padding=0.35) # 이미지에서 얼굴 영역만 추출해주는 것  / 얼굴 이미지들
    return faces


test_img = dlib.load_rgb_image('./imgs/04.jpg') # 이미지 불러오기
test_faces = align_faces(test_img) # align_face 함수에 이미지 적용
fig, axes = plt.subplots(1, len(test_faces)+1, figsize=(20,16)) # figure 안에 원본 + 1 를 해서 이미지를 생성
axes[0].imshow(test_img) # figure의 [0] (원본)을 test_img로
for i, face in enumerate(test_faces):
    axes[i+1].imshow(face)


# In[16]:


# Beauty GAN 모델 불러오기 (Beauty GAN 사용법대로 코드)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.import_meta_graph('./models/model.meta')
saver.restore(sess, tf.train.latest_checkpoint('./models'))
graph = tf.get_default_graph()
X = graph.get_tensor_by_name('X:0')
Y = graph.get_tensor_by_name('Y:0')
Xs = graph.get_tensor_by_name('generator/xs:0')


# In[18]:


def preprocess(img): # 이미지 전처리
    return (img / 255. - 0.5) * 2

def deprocess(img):
    return (img + 1) / 2


# In[57]:


img1 = dlib.load_rgb_image('./imgs/no_makeup/2456983E57891F6A13.jpg') # 노메이크업 이미지
img1_faces = align_faces(img1)

img2 = dlib.load_rgb_image('./imgs/makeup/43944_27928_3623.png') # 메이크업 레퍼런스 이미지
img2_faces = align_faces(img2)

fig, axes = plt.subplots(1,2, figsize=(16,10))
axes[0].imshow(img1_faces[0])
axes[1].imshow(img2_faces[0])
plt.show()


# In[58]:


src_img = img1_faces[0]
ref_img = img2_faces[0]

X_img = preprocess(src_img)
X_img = np.expand_dims(X_img, axis=0)

Y_img = preprocess(ref_img)
Y_img = np.expand_dims(Y_img, axis=0)

output = sess.run(Xs, feed_dict={X:X_img, Y:Y_img}) # 결과물
output_img = deprocess(output[0])

fig, axes = plt.subplots(1, 3, figsize=(20,10))
axes[0].set_title('Source')
axes[0].imshow(src_img)
axes[1].set_title('Reference')
axes[1].imshow(ref_img)
axes[2].set_title('Result')
axes[2].imshow(output_img)
plt.show()


# In[ ]:




