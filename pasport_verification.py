import os
from flask import Flask, flash, request, redirect, url_for, send_from_directory, render_template, session
from flask_session import Session
from werkzeug.utils import secure_filename
from werkzeug.middleware.profiler import ProfilerMiddleware

import dlib
import numpy as np
import cv2
import torch
from torch.autograd import Variable
import net_sphere
import sys
import glob
import pickle
from pathlib import Path
import time
from matlab_cp2tform import get_similarity_transform_for_cv2
from scipy.spatial.distance import pdist
import logging
from random import choice
from string import ascii_letters
#from flask_ngrok import run_with_ngrok
from flask_images import resized_img_src, Images

from flask_dropzone import Dropzone

logging.basicConfig(filename='server-side.log', format='%(asctime)s :: %(levelname)s :: %(message)s', level=logging.DEBUG)


def logger(return_res = True, show_args = True):

    def decorator(func):
        import time

        def wrapper(*args, **kwargs):
            start = time.time()
            if show_args:
              logging.info(f'Function {func.__name__} started with arguments {args} and {kwargs}')
            else:
              logging.info(f'Function {func.__name__} started')

            try:
                res = func(*args, **kwargs)
                finish = time.time() 
                if return_res:
                  logging.info(f'Function {func.__name__} finished in {finish-start} sec with result {res}')
                else:
                  logging.info(f'Function {func.__name__} finished in {finish-start} sec')
                
            except Exception as exc:
                finish = time.time()
                logging.error(f'Function {func.__name__} crashed with Exception {exc} in {finish-start} sec') 

            return res    

        return wrapper   

    return decorator      


@logger(return_res=False, show_args = False)
def alignment(src_img,src_pts):
    ref_pts = [ [30.2946, 51.6963],[65.5318, 51.5014],
        [48.0252, 71.7366],[33.5493, 92.3655],[62.7299, 92.2041] ]

            
    crop_size = (96, 112)
    src_pts = np.array(src_pts).reshape(5,2)

    s = np.array(src_pts).astype(np.float32)
    r = np.array(ref_pts).astype(np.float32)

    tfm = get_similarity_transform_for_cv2(s, r)
    face_img = cv2.warpAffine(src_img, tfm, crop_size)
    return face_img



face_template = np.load('face_template.npy')
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
INNER_EYES_AND_BOTTOM_LIP = [38, 44, 30, 49, 54]

args = {"--net": 'sphere20a',        
        "--model": 'sphere20a.pth'}
net = getattr(net_sphere,args['--net'])()
net.load_state_dict(torch.load(args['--model']))
#net.cuda()
net.eval()
net.feature = True

PROFILE_DIR = 'profile_dir'
UPLOAD_FOLDER = 'static'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
dropzone = Dropzone(app)
#images = Images(app)
#run_with_ngrok(app)

app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.secret_key = 'super secret key'
app.config['SESSION_TYPE'] = 'filesystem'

app.config.update(
    UPLOAD_FOLDER=UPLOAD_FOLDER,
    # Flask-Dropzone config:
    DROPZONE_ALLOWED_FILE_TYPE='image',
    DROPZONE_MAX_FILE_SIZE=3,
    DROPZONE_MAX_FILES=2,
    DROPZONE_IN_FORM=True,
    DROPZONE_UPLOAD_ON_CLICK=True,
    DROPZONE_UPLOAD_ACTION='handle_upload',  # URL or endpoint
    DROPZONE_UPLOAD_BTN_ID='submit',
    )

Session(app)


@logger(show_args = False)
def face_detection(img):
    face_rects = detector(img, 1)
    first_face = face_rects[0]
    faces_amount = len(face_rects)

    return faces_amount, first_face

@logger(return_res=False, show_args = False)
def rect_maker(img, face_rect):
  cv2.rectangle(img, (face_rect.left(), face_rect.top()), (face_rect.right(), face_rect.bottom()), (255, 0, 0), 6)
  return None

@logger(return_res=False, show_args = False)
def recolor(img): 
  img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  return img
  
@logger(return_res=False, show_args = False)
def preview_maker(img, need_rect = False, face_rect = None):
    
    k = img.shape[1]/img.shape[0] #width/height
    w_max = 250
    h_max = 450

    if w_max/k > h_max:
        heigth = int(h_max)
        width = int(h_max*k)
    else: 
        height = int(w_max/k)
        width = int(w_max)    
    
    preview = cv2.resize(img, (width, height))

    if need_rect: 
      rect_maker(preview, face_rect)

    return preview, width

@logger(return_res=False)
def img_read(path):
  return cv2.imread(path)


@logger(show_args = False)
def prediction(img, face_rect):
  return predictor(img, face_rect)  


@logger(return_res=False, show_args = False)
def cnn_processing(img):
    imglist = [img,cv2.flip(img,1)]
    for j in range(len(imglist)):
        imglist[j] = imglist[j].transpose(2, 0, 1).reshape((1,3,112,96))
        imglist[j] = (imglist[j]-127.5)/128.0
    img_part = np.vstack(imglist)
    with torch.no_grad():
        img_part = Variable(torch.from_numpy(img_part).float())

    output = net(img_part)

    return output.data.numpy()

@logger()
def verification(dist):
    if dist > 0.5 and dist < 0.6:
        text = 'Скорее всего, это разные люди. Однако, есть небольшие сомнения - лучше попробовать еще другое фото, чтобы убедиться наверняка.'   
    if dist > 0.6:
        text = 'Скорее всего, это разные люди.'   
    if dist < 0.5 and dist > 0.3:
        text = 'Возможно, что это один и тот же человек. Но лучше проверить еще на другом фото, чтобы точно быть уверенным.'  
    if dist < 0.3:
        text = 'Скорее всего это один и тот же человек.'
    return text       

@app.route('/again', methods=['POST'])
def again():
    return render_template('index-test.html', message = 'Загрузите новые фотографии, на каждой из которых изображено только 1 лицо. После этого нажмите кнопку "Отправить".')

@app.route('/')
def index():
    return render_template('index-test.html', message = 'Выберите 2 фотографии, на каждой из которых изображено только 1 лицо. После этого нажмите кнопку "Отправить".')


@app.route('/first_upload', methods=['POST'])
def handle_upload():
  session["count"] = 0
  session["file_names"] = {}
  for key, file in request.files.items():
        if key.startswith('file'):
            session["count"] += 1
            session["file_names"][f'file{session["count"]}'] = file.filename
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))

  return ''

  


@app.route('/form', methods=['POST'])
def handle_form(): 

  session["face_rectangles"] = {}

  preview_names = {}
  w = {}

  if session["count"] != 2:
    return render_template('index-test.html', message = 'Вы загрузили меньше 2 фотографий, попробуйте еще раз.')

  else: 
    for key, name in session["file_names"].items(): 
      img1 = img_read('static/' + name)
      img1, _ = preview_maker(img1)
      img_to_detection = recolor(img1)

      try:
        face_amount, face_rect = face_detection(img_to_detection)
      except IndexError: 
        return render_template('index-test.html', message = 'Похоже, на одной из фотографий не найдено лицо. Попробуйте загрузить другие фото.')   

      if face_amount > 1: 
        return render_template('index-test.html', message = 'На фотографиях изображено слишком много лиц. Загрузите фотографию, где изображено только одно лицо.')

      session["face_rectangles"][name] = face_rect      
      
      preview_photo, w[key] = preview_maker(img1, need_rect=True, face_rect=face_rect) 
      cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], f'{name}_preview.jpg'), preview_photo)
      preview_names[key] = f'{name}_preview.jpg' #поменять                     

    return render_template('preview-test.html', img1 = preview_names['file1'], w1 = w['file1'],  img2 = preview_names['file2'], w2 = w['file2'])


@app.route('/upload', methods=['POST'])
def upload_file():

    embs = []
    names = {}
    w = {}

    try:

      for key, name in session["file_names"].items(): 
        img1 = img_read('static/' + name)   
        face_rect = session["face_rectangles"][name]

        points = prediction(img1, face_rect)
        landmarks = np.array([*map(lambda p: [p.x, p.y], points.parts())])

        img_part = alignment(img1, landmarks[INNER_EYES_AND_BOTTOM_LIP])        
        f = cnn_processing(img_part)
        emb_face = f[0]
        embs.append(emb_face)
        preview, w[key] = preview_maker(img1, need_rect=True, face_rect=face_rect)
        cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], f'{name}_final.jpg'), preview)
        names[key] = f'{name}_final.jpg'

      dist = round(pdist([embs[0], embs[1]], 'cosine')[0], 3) 

      text = verification(dist)

      return render_template('uploaded-test.html', dist = dist, text = text, img1 = names['file1'], w1 = w['file1'], img2 = names['file2'], w2 = w['file2'])

    except IndexError:
        logging.error('No face detected!') 
        return render_template('index-test.html', message = 'Похоже, на одной из фотографий не найдено лицо. Попробуйте загрузить другие фото.')   

    
app.config['PROFILE'] = True
app.wsgi_app = ProfilerMiddleware(app.wsgi_app, restrictions=[5], profile_dir=PROFILE_DIR)


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)