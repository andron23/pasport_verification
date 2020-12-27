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


app.secret_key = 'super secret key'
app.config['SESSION_TYPE'] = 'filesystem'


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    uploaded_files = request.files.getlist('file[]')
    user_image = {}
    embs = []
    print(len(uploaded_files))
    if len(uploaded_files) != 2:
        return render_template('index.html')
    else:    
        for i, file in enumerate(uploaded_files):
        
            if file.filename != '':
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], f'pic{i+1}.jpg'))
                image_bytes = file.read()
                nparr = np.frombuffer(image_bytes, np.uint8)
                user_image[f'{i+1}'] = '../' + str(UPLOAD_FOLDER) + '/' + str(file.filename)
                img1 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                face_rect1 =  detector(img1, 1)[0] 
                points1 = predictor(img1, face_rect1)
                landmarks1 = np.array([*map(lambda p: [p.x, p.y], points1.parts())])
                img_part1 = alignment(img1, landmarks1[INNER_EYES_AND_BOTTOM_LIP])
                imglist1 = [img_part1,cv2.flip(img_part1,1)]
                for i in range(len(imglist1)):
                    imglist1[i] = imglist1[i].transpose(2, 0, 1).reshape((1,3,112,96))
                    imglist1[i] = (imglist1[i]-127.5)/128.0
                img_part1 = np.vstack(imglist1)
                with torch.no_grad():
                    img_part1 = Variable(torch.from_numpy(img_part1).float())
                output1 = net(img_part1)
                f1 = output1.data.numpy()
                emb_face1 = f1[0]
                embs.append(emb_face1)

        dist = pdist([embs[0], embs[1]], 'cosine')[0]        

    return render_template('uploaded.html', dist = dist)

app.run(debug=True)    
