# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import cv2
import time
import sys
import os
import glob

import numpy as np
import tensorflow as tf


# TOM: hack to import from submodule without __init.py__ (can we fix it?)
import sys
ROOT_PATH = '/Users/frappuccino_o/dev/FalsifyNN/neural_nets/squeezeDet/'
cfg_folder = os.path.realpath(ROOT_PATH + 'src')
sys.path.append(cfg_folder)

from config import *
from train import _draw_box
from nets import *

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('mode', 'image', """'image' or 'video'.""")
tf.app.flags.DEFINE_string('checkpoint', ROOT_PATH + 'data/model_checkpoints/squeezeDet/model.ckpt-87000',"""Path to the model parameter file.""")

def init():

  with tf.Graph().as_default():
    # Load model
    mc = kitti_squeezeDet_config()
    mc.BATCH_SIZE = 1
    # model parameters will be restored from checkpoint
    mc.LOAD_PRETRAINED_MODEL = False
    model = SqueezeDet(mc, FLAGS.gpu)

    # Start tensorflow session
    saver = tf.train.Saver(model.model_params)
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    saver.restore(sess, FLAGS.checkpoint)

    return (sess,mc,model)



def classify(im_path,conf):
    (sess,mc,model) = conf;
    im = cv2.imread(im_path)
    im = im.astype(np.float32, copy=False)
    im = cv2.resize(im, (mc.IMAGE_WIDTH, mc.IMAGE_HEIGHT))
    input_image = im - mc.BGR_MEANS

    # Detect
    det_boxes, det_probs, det_class = sess.run(
     [model.det_boxes, model.det_probs, model.det_class],
     feed_dict={model.image_input:[input_image], model.keep_prob: 1.0})

    # Filter
    final_boxes, final_probs, final_class = model.filter_prediction(
     det_boxes[0], det_probs[0], det_class[0])

    #keep_idx    = [idx for idx in range(len(final_probs)) \
    #                   if final_probs[idx] > mc.PLOT_PROB_THRESH]

    keep_idx = [idx for idx in range(len(final_probs)) \
                if final_probs[idx] > 0.1]

    final_boxes = [final_boxes[idx] for idx in keep_idx]
    final_probs = [final_probs[idx] for idx in keep_idx]
    final_class = [final_class[idx] for idx in keep_idx]

    # # Extract labels + confidence values
    # res = []
    # for label, confidence, box in zip(final_class, final_probs, final_boxes):
    #     res.append((label,confidence,box))
    # return res

    return (final_boxes,final_probs,final_class)


# def main(argv=None):
#   if not tf.gfile.Exists(FLAGS.out_dir):
#     tf.gfile.MakeDirs(FLAGS.out_dir)
#   if FLAGS.mode == 'image':
#     conf = startSqueezedet()
#     #squeezeDet('./squeezeDet/data/sample.png',conf)
#     squeezeDet('./squeezeDet/data/sample_2.png',conf)
#   else:
#     video_demo()
#
# if __name__ == '__main__':
#     tf.app.run()
