import os
from flask import Flask, flash, request, redirect, url_for, send_from_directory, render_template
from werkzeug.utils import secure_filename

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
from matlab_cp2tform import *
from scipy.spatial.distance import pdist
import logging
from random import choice
from string import ascii_letters

logging.basicConfig(filename='server-side.log', format='%(asctime)s :: %(levelname)s :: %(message)s', level=logging.DEBUG)


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



UPLOAD_FOLDER = 'static'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.secret_key = 'super secret key'
app.config['SESSION_TYPE'] = 'filesystem'


def face_detection(img):
    face_rects = detector(img, 1)
    first_face = face_rects[0]
    faces_amount = len(face_rects)

    return faces_amount, first_face

def preview_maker(img, face_rect):
    cv2.rectangle(img, (face_rect.left(), face_rect.top()), (face_rect.right(), face_rect.bottom()), (255, 0, 0), 10)
    width = 250
    height = int(img.shape[0]*(250/img.shape[1]))
    logging.info(f'Old shape: {(img.shape[0], img.shape[1])}; New shape: {(height, width)}: (h,w).')
    preview = cv2.resize(img, (width, height))

    return preview


def photo_upload():
    uploaded_files_1 = request.files.getlist('file1')
    logging.info(uploaded_files_1)
    uploaded_files_2 = request.files.getlist('file2')
    logging.info(uploaded_files_2) 
    if len(uploaded_files_1) > 0 and len(uploaded_files_2) > 0:
        logging.info(uploaded_files_1[0].filename) 
        logging.info(uploaded_files_2[0].filename)    
        uploaded_files = uploaded_files_1 + uploaded_files_2
        logging.info('Combined files')
        logging.info(f'Length: {len(uploaded_files_1[0].filename)} and {len(uploaded_files_2[0].filename)}')
        if len(uploaded_files_2[0].filename) == 0 or len(uploaded_files_1[0].filename) == 0:
            photo_amount = 1
        elif len(uploaded_files_2) > 1 or len(uploaded_files_1) > 1:
            photo_amount = 3
        else:
            photo_amount = 2   
    else:
        photo_amount = 1
        uploaded_files = []         
    logging.info(f'Photo amount: {photo_amount}')
    

    return uploaded_files, photo_amount

def photo_save(file, name):
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], name))
    name_to_save = ''.join(choice(ascii_letters) for i in range(20))
    file.save(os.path.join('dataset', f'{name_to_save}.jpg'))
    logging.info('File saved')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/retry_less')
def retry_less():
    return render_template('retry_less.html')

@app.route('/retry_more')
def retry_more():
    return render_template('retry_more.html')

@app.route('/retry_no_face')
def retry_no_face():
    return render_template('retry_no_face.html') 

@app.route('/retry_more_faces')
def retry_more_faces():
    return render_template('retry_more_faces.html')          

@app.route('/preview', methods=['POST'])
def preview():

    uploaded_files, photo_amount = photo_upload()


    logging.info(f'Photo amount: {photo_amount}')

    try:
        if photo_amount < 2:
            logging.error('Less then 2 files')
            return redirect(url_for('retry_less'))
            
            
        elif photo_amount > 2:
            logging.error('More then 2 files')
            return redirect(url_for('retry_more'))
            
        else:    
            for i, file in enumerate(uploaded_files):
        
                if file.filename != '':

                    photo_save(file, name = f'pic{i+1}.jpg')

                    img1 = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], f'pic{i+1}.jpg'))

                    logging.info('File read')

                    face_amount, face_rect = face_detection(img1)

                   # if face_amount < 1: 
                        #logging.error('No face found')
                        #return redirect(url_for('retry_no_face'))
                        #break

                    if face_amount > 1: 
                        logging.error('Too much faces found')
                        return redirect(url_for('retry_more_faces'))                                           
                        break                
                                        
                    preview_photo = preview_maker(img1, face_rect)

                    cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], f'pic{i+1}_preview.jpg'), preview_photo)

            return render_template('preview.html')

    except IndexError:
        logging.error('No face detected!') 
        return redirect(url_for('retry_no_face'))  



@app.route('/upload', methods=['POST'])
def upload_file():
    embs = []
    try:
 
            for i in range(2):
        

                    img1 = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], f'pic{i+1}.jpg'))
                    logging.info('File read')
                #os.remove(os.path.join(app.config['UPLOAD_FOLDER'], f'pic{i+1}.jpg'))
                    #user_image[f'{i+1}'] = '../' + str(UPLOAD_FOLDER) + '/' + str(file.filename)
                #img1 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                    face_rect =  detector(img1, 1)
                    if len(face_rect) < 1: 
                        return render_template('retry.html', error = f'На фотографии {i+1} не найдено лицо! Попробуйте еще раз.')
                        logging.error('No face found')
                        break

                    if len(face_rect) > 1: 
                        return render_template('retry.html', error = f'На фотографии {i+1} найдено несколько лиц! Попробуйте еще раз.')
                        logging.error('Too much faces found')
                        break

                    face_rect = face_rect[0]

                    logging.info('Detection completed')
                    points = predictor(img1, face_rect)
                    logging.info('Point prediction completed')
                    landmarks = np.array([*map(lambda p: [p.x, p.y], points.parts())])
                    img_part = alignment(img1, landmarks[INNER_EYES_AND_BOTTOM_LIP])
                    logging.info('Alignment completed')
                    imglist = [img_part,cv2.flip(img_part,1)]
                    for j in range(len(imglist)):
                        imglist[j] = imglist[j].transpose(2, 0, 1).reshape((1,3,112,96))
                        imglist[j] = (imglist[j]-127.5)/128.0
                    img_part = np.vstack(imglist)
                    logging.info('Vstack completed')
                    start = time.time()
                    with torch.no_grad():
                    # Почему-то если убрать строчку про тест, сервис не отрабатывает, как нужно... Просто зависает. Смотри логи за 19.01.2021. Хотя и с ним не всегда работает. 
                        test = torch.from_numpy(img_part).float()
                        logging.info('Test completed')
                        img_part = Variable(torch.from_numpy(img_part).float())
                        logging.info('Variable completed')
                    output = net(img_part)
                    finish = time.time() - start
                    logging.info('Photo processed with CNN')
                    logging.info(f'CNN processinf time: {finish}')
                    f = output.data.numpy()
                    emb_face = f[0]
                    embs.append(emb_face)
                    cv2.rectangle(img1, (face_rect.left(), face_rect.top()), (face_rect.right(), face_rect.bottom()), (255, 0, 0), 10)
                    width = 250
                    height = int(img1.shape[0]*(250/img1.shape[1]))
                    logging.info(f'Old shape: {(img1.shape[0], img1.shape[1])}; New shape: {(height, width)}: (h,w).')
                    img1 = cv2.resize(img1, (width, height))
                    cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], f'pic{i+1}.jpg'), img1)
                

            dist = round(pdist([embs[0], embs[1]], 'cosine')[0], 3) 

            if dist > 0.5 and dist < 0.6:
                text = 'Скорее всего, это разные люди. Однако, есть небольшие сомнения - лучше попробовать еще другое фото, чтобы убедиться наверняка.'   
            if dist > 0.6:
                text = 'Скорее всего, это разные люди.'   
            if dist < 0.5 and dist > 0.3:
                text = 'Возможно, что это один и тот же человек. Но лучше проверить еще на другом фото, чтобы точно быть уверенным.'  
            if dist < 0.3:
                text = 'Скорее всего это один и тот же человек.'       
            logging.info('Dist found')    

            return render_template('uploaded.html', dist = dist, text = text)

    except IndexError:
        logging.error('No face detected!') 
        return render_template('retry.html')   



if __name__ == '__main__':
    app.run(host='0.0.0.0')    
