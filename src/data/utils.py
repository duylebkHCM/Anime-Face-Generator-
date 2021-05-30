import numpy as np
import sys
import os
import csv
from scipy import misc
import scipy.stats as stats
import random
import pickle

from torch.utils.data import Dataset

hair_color = ['orange hair', 'white hair', 'aqua hair', 'gray hair',
    'green hair', 'red hair', 'purple hair', 'pink hair',
    'blue hair', 'black hair', 'brown hair', 'blonde hair']

eye_color = ['gray eyes', 'black eyes', 'orange eyes',
    'pink eyes', 'yellow eyes', 'aqua eyes', 'purple eyes',
    'green eyes', 'brown eyes', 'red eyes', 'blue eyes']

def crop_center(img,cropx,cropy):
    y,x,z = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx, :]


def make_one_hot(hair, eye):

    eyes_hot = np.zeros([len(eye_color)])
    eyes_hot[eye] = 1
    hair_hot = np.zeros([len(hair_color)])
    hair_hot[hair] = 1
    tag_vec = np.concatenate((eyes_hot, hair_hot))

    return tag_vec


def load_test(test_path, hair_map, eye_map):

    test = []
    with open(test_path, 'r') as f:

        for line in f.readlines():
            hair = 0
            eye = 0
            if line == '\n':
                break
            line = line.strip().split(',')[1]
            p = line.split(' ')
            p1 = ' '.join(p[:2]).strip()
            p2 = ' '.join(p[-2:]).strip()
        
            if p1 in hair_map:
                hair = hair_map[p1]
            elif p2 in hair_map:
                hair = hair_map[p2]
            
            if p1 in eye_map:
                eye = eye_map[p1]
            elif p2 in eye_map:
                eye = eye_map[p2]

            test.append(make_one_hot(hair, eye))
    
    return test

 
def dump_img(img_dir, img_feats, test):

    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    img_feats = (img_feats + 1.)/2 * 255.
    img_feats = np.array(img_feats, dtype=np.uint8)

    for idx, img_feat in enumerate(img_feats):
        path = os.path.join(img_dir, 'sample_{}_{}.jpg'.format(test, idx+1))
        misc.imsave(path, img_feat)



def preprocessing(preproc_dir, img_dir, tag_path, eye_map, hair_map):

    attrib_tags = [] 
    img_feat = []
    img_size = 96
    resieze = int(96*1.15)

    with open(tag_path, 'r') as f:
        for idx, row in enumerate(csv.reader(f)):

            tags = row[1].split('\t')
            hair = 'unk'
            eyes = 'unk'
            has_hair = False
            has_eye = False
            skip_hair = False
            skip_eye = False
            skip = False

            for t in tags:
                if t != '':
                    tag = t.split(':')[0].strip()

                    if tag == 'bicolored eyes':
                        print(tag)
                        skip = True
                        break

                    if tag in eye_map:

                        if has_eye:
                            skip_hair = True
                        
                        eyes = tag
                        has_eye = True

                    elif tag in hair_map:
                        if has_hair:
                            skip_eye = True

                        hair = tag
                        has_hair = True

            if skip_hair:
                hair = 'unk'

            if skip_eye:
                eyes = 'unk'


            if eyes == 'unk' or hair == 'unk':
                skip = True

            if skip:
                continue

            hair_idx = hair_map[hair]
            eyes_idx = eye_map[eyes]

            img_path = os.path.join(img_dir, '{}.jpg'.format(idx))
            feat = misc.imread(img_path)
            feat = misc.imresize(feat, [img_size, img_size, 3])
            attrib_tags.append([hair_idx, eyes_idx])
            img_feat.append(feat)

            m_feat = np.fliplr(feat)
            attrib_tags.append([hair_idx, eyes_idx])
            img_feat.append(m_feat)

            feat_p5 = misc.imrotate(feat, 5)
            feat_p5 = misc.imresize(feat_p5, [resieze , resieze, 3])
            feat_p5 = crop_center(feat_p5, img_size,img_size)

            attrib_tags.append([hair_idx, eyes_idx])
            img_feat.append(feat_p5)

            feat_m5 = misc.imrotate(feat, -5)
            feat_m5 = misc.imresize(feat_m5, [resieze, resieze, 3])
            feat_m5 = crop_center(feat_m5, img_size,img_size)

            attrib_tags.append([hair_idx, eyes_idx])
            img_feat.append(feat_m5)

    img_feat = np.array(img_feat)

    pickle.dump(img_feat, open(os.path.join(preproc_dir, "img_feat_96.dat"), 'wb'))
    pickle.dump(attrib_tags, open(os.path.join(preproc_dir, "tags.dat"), 'wb'))

    return img_feat, attrib_tags

