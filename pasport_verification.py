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


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    uploaded_files_1 = request.files.getlist('file1')
    logging.info('Uploaded file 1')
    uploaded_files_2 = request.files.getlist('file2')
    logging.info('Uploaded file 2')
    uploaded_files = uploaded_files_1 + uploaded_files_2
    logging.info('Combined files')
    user_image = {}
    embs = []
    logging.info(f'Length: {len(uploaded_files_1)} and {len(uploaded_files_2)}')
    try:
        if len(uploaded_files_1) + len(uploaded_files_2) != 2:
            return render_template('index.html')
            logging.error('Less then 2 files')
        else:    
            for i, file in enumerate(uploaded_files):
        
                if file.filename != '':
                    file.save(os.path.join(app.config['UPLOAD_FOLDER'], f'pic{i+1}.jpg'))
                    name_to_save = ''.join(choice(ascii_letters) for i in range(20))
                    file.save(os.path.join('dataset', f'{name_to_save}.jpg'))
                    logging.info('File saved')
                #time.sleep(2)
                #print(len(file))
                #image_bytes = file.read()
                #print(len(image_bytes))
                #nparr = np.frombuffer(image_bytes, np.uint8)
                    img1 = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], f'pic{i+1}.jpg'))
                    logging.info('File read')
                #os.remove(os.path.join(app.config['UPLOAD_FOLDER'], f'pic{i+1}.jpg'))
                    user_image[f'{i+1}'] = '../' + str(UPLOAD_FOLDER) + '/' + str(file.filename)
                #img1 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                    face_rect =  detector(img1, 1)[0] 
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
                    cv2.rectangle(img1, (face_rect.left(), face_rect.top()), (face_rect.right(), face_rect.bottom()), (255, 125, 255), 5)
                    cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], f'pic{i+1}.jpg'), img1)
                

            dist = round(pdist([embs[0], embs[1]], 'cosine')[0], 3)    
            logging.info('Dist found')    

        return render_template('uploaded.html', dist = dist)

    except IndexError:
        logging.error('No face detected!') 
        return render_template('retry.html')   


if __name__ == '__main__':
    app.run(host='0.0.0.0')    
