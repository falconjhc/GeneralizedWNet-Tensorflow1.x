# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import

import os
import glob

import imageio
import scipy.misc as misc
import numpy as np
import copy as cp

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as img

matplotlib.use('agg')
import pylab
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
def FindKeys(dict, val): return list(value for key, value in dict.items() if key in val)


def PrintNetworkVars(networkName, exceptions=[]):
    temp_layers=[]
    num_parameter_total = 0
    var_list = [ii for ii in tf.trainable_variables() if networkName in ii.name]
    for ii in var_list:
        for jj in exceptions:
            if jj in ii.name:
                continue
        
        if 'batch_normalization' not in ii.name:
            multiplier=1
            for jj in range(len(ii.shape)):
                    curt = np.int64(ii.shape[jj])
                    multiplier = multiplier * curt
                    num_parameter_total = num_parameter_total + multiplier

        if 'W:0' in ii.name and 'downsample' not in ii.name:
            temp_layers.append(ii)
    num_parameter_total_m = num_parameter_total / 1000000
    print("Total Parameters: %.2fM, " % num_parameter_total_m, end='')
    print("in total %d layers." % len(temp_layers))


def SplitName(inputStr):
    result = []
    temp = ""
    for i in inputStr:
        if i.isupper() :
            if temp != "":
                result.append(temp)
            temp = i
        else:
            temp += i

    if temp != "":
        result.append(temp)
    return result



def get_uninitialized_variables(sess,var_list):
    unitialized=[]
    for var in var_list:
        try:
            sess.run(var)
        except tf.errors.FailedPreconditionError:
            unitialized.append(var)
    return unitialized


def pad_seq(seq, batch_size):
    # pad the sequence to be the multiples of batch_size
    seq_len = len(seq)
    if seq_len % batch_size == 0:
        return seq
    padded = batch_size - (seq_len % batch_size)
    seq.extend(seq[:padded])
    return seq




def normalize_image(img):
    """
    Make image zero centered and in between (-1, 1)
    """
    normalized = (img / 127.5) - 1.
    return normalized


def read_split_image(img):
    mat = misc.imread(img).astype(np.float)
    side = int(mat.shape[1] / 2)
    assert side * 2 == mat.shape[1]
    img_A = mat[:, :side]  # target [0,1,2]
    img_B = mat[:, side:]  # source

    return img_A, img_B


def shift_and_resize_image(img, shift_x, shift_y, nw, nh):
    w=img.shape[0]
    h=img.shape[1]
    enlarged = misc.imresize(img, [nw, nh])
    return enlarged[shift_x:shift_x + w, shift_y:shift_y + h]


# def scale_back(images):
#     return (images + 1.) / 2.

def scale_with_probability(images,channels,probability):


    if np.sum(probability) < 0:
        probability = - probability
    batchsize = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]
    probability = np.reshape(probability, [batchsize, 1])
    if channels==1:
        probability=np.reshape(probability,[batchsize,1,1])
        probability=np.tile(probability,[1,h,w])
        probability=np.reshape(probability,[probability.shape[0],probability.shape[1],probability.shape[2],1])
    elif channels==3:
        probability = np.reshape(probability, [batchsize, 1, 1, 1])
        probability = np.tile(probability, [1, h, w,channels])
    images = np.multiply(probability,images)
    return images

def scale_back_for_img(images):

    if np.min(images) == np.max(images):
        return images
    min_v=np.min(images)
    images=images-min_v
    max_v=np.max(images)
    images=np.float32(images)/np.float32(max_v)
    return images
def scale_back_for_dif(images):
    images = np.abs(images)
    if np.min(images) == np.max(images):
        return images

    min_v=np.min(images)
    images=images-min_v
    max_v=np.max(images)
    images=np.float32(images)/np.float32(max_v)
    return images


def image_show(img):
    img_out = cp.deepcopy(img)
    img_out = np.squeeze(img_out)
    img_shapes=img_out.shape
    if len(img_shapes)==2:
        curt_channel_img = img_out
        min_v = np.min(curt_channel_img)
        curt_channel_img = curt_channel_img - min_v
        max_v = np.max(curt_channel_img)
        curt_channel_img = curt_channel_img/ np.float32(max_v)
        img_out = curt_channel_img*255
    elif img_shapes[2] == 3:
        channel_num = img_shapes[2]
        for ii in range(channel_num):
            curt_channel_img = img[:,:,ii]
            min_v = np.min(curt_channel_img)
            curt_channel_img = curt_channel_img - min_v
            max_v = np.max(curt_channel_img)
            curt_channel_img = curt_channel_img / np.float32(max_v)
            img_out[:,:,ii] = curt_channel_img*255
    else:
        print("Channel Number is INCORRECT:%d" % img_shapes[2])
    plt.imshow(np.float32(img_out)/255)
    pylab.show()

def image_revalue(img,tah_mark):
    img_out = cp.deepcopy(img)
    img_out = np.squeeze(img_out)
    if tah_mark:
        img_out = np.tanh(img_out)
    return img_out


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j * h:j * h + h, i * w:i * w + w, :] = image

    return img


def save_concat_images(imgs, img_path):
    concated = np.concatenate(imgs, axis=1)
    misc.imsave(img_path, concated)


def compile_frames_to_gif(frame_dir, gif_file):
    frames = sorted(glob.glob(os.path.join(frame_dir, "*.png")))
    print(frames)
    images = [misc.imresize(imageio.imread(f), interp='nearest', size=0.33) for f in frames]
    imageio.mimsave(gif_file, images, duration=0.1)
    return gif_file



def calculate_weighted_loss(weight,loss_org,penalty=1):


    loss_org_shape_dim = loss_org.shape.ndims
    weight_matrix_reshape = [int(weight.shape[0])]
    weight_matrix_tile = [1]
    for ii in range(loss_org_shape_dim-1):
        weight_matrix_reshape.append(1)
        weight_matrix_tile.append(int(loss_org.shape[ii+1]))


    weight_matrix = tf.tile(input=tf.reshape(tensor=weight,
                                             shape=weight_matrix_reshape),
                            multiples=weight_matrix_tile)
    loss_matrix = tf.multiply(weight_matrix,loss_org)
    loss_value = tf.reduce_mean(loss_matrix)*penalty
    return loss_value


def correct_ckpt_path(real_dir,maybe_path):
    maybe_path_dir = str(os.path.split(os.path.realpath(maybe_path))[0])
    if not maybe_path_dir == real_dir:
        return os.path.join(real_dir,str(os.path.split(os.path.realpath(maybe_path))[1]))
    else:
        return maybe_path



def softmax(x):
    x -= np.max(x, axis= 1, keepdims=True)
    f_x = np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
    return f_x