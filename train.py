
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"]="1"
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

from ops import *
import cv2
import os
from vgg import *
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time
from natsort import natsorted
import imageio

path = 'data'

np.random.seed(0)

global params
params = {'path' : path,
          'batch_size' : 8,
          'output_size': 256,
          'gf_dim': 32,
          'df_dim': 32,
          'model_path' : './model',
          'L1_lambda': 100,
          'lr': 0.0001,
          'beta_1': 0.5,
          'epochs': 50,
          'Img_saved_path' : 'SavedImgs',
          'Img_saved_path_for_real_data' : 'SavedReal',
          'Stage_epochs' : [10000,20000]}

if not os.path.isdir(params['Img_saved_path']):
    os.mkdir(params['Img_saved_path'])
    
if not os.path.isdir(params['Img_saved_path_for_real_data']):
    os.mkdir(params['Img_saved_path_for_real_data'])
    
def get_file_paths(path):
    img_paths = [os.path.join(root, file)  for root, dirs, files in os.walk(path) for file in files if '_gt' not in file]
    gt_path = [os.path.join(os.path.dirname(file), os.path.basename(file).split('.')[0] + '_gt.png') for file in img_paths]
    return np.array(img_paths[:int(len(img_paths)*.9)]), np.array(gt_path[:int(len(gt_path)*.9)]), np.array(img_paths[int(len(img_paths)*.9):])        , np.array(gt_path[int(len(gt_path)*.9):])


def load_data_CONTENT(path):
    im = ~cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (256, 256))
    return np.expand_dims(im, -1)/127.5 - 1.

def load_data(path):
    im = cv2.resize(cv2.imread(path, 0), (256, 256))
    return  np.expand_dims(im, -1)/127.5 - 1.

# Functions to load and save weights
def load_weights(saver, model_dir):
    ckpt = tf.train.get_checkpoint_state(model_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver.restore(sess, os.path.join(model_dir, ckpt_name))
        print("MODEL LOADED SUCCESSFULLY")
    else:
        print("LOADING MODEL FAILED")

def save(saver, checkpoint_dir, step):
    dir = os.path.join(checkpoint_dir, "model")
    saver.save(sess, dir, step)


global g_bn_d1, g_bn_d2, g_bn_d3, g_bn_d4, g_bn_d5, g_bn_d6, g_bn_d7

d_bn1 = batch_norm(name='d_bn1')
d_bn2 = batch_norm(name='d_bn2')
d_bn3 = batch_norm(name='d_bn3')

g_bn_e2 = batch_norm(name='g_bn_e2')
g_bn_e3 = batch_norm(name='g_bn_e3')
g_bn_e4 = batch_norm(name='g_bn_e4')
g_bn_e5 = batch_norm(name='g_bn_e5')
g_bn_e6 = batch_norm(name='g_bn_e6')
g_bn_e7 = batch_norm(name='g_bn_e7')
g_bn_e8 = batch_norm(name='g_bn_e8')

g_bn_d1 = batch_norm(name='g_bn_d1')
g_bn_d2 = batch_norm(name='g_bn_d2')
g_bn_d3 = batch_norm(name='g_bn_d3')
g_bn_d4 = batch_norm(name='g_bn_d4')
g_bn_d5 = batch_norm(name='g_bn_d5')
g_bn_d6 = batch_norm(name='g_bn_d6')
g_bn_d7 = batch_norm(name='g_bn_d7')


global g_bn_d1_, g_bn_d2_, g_bn_d3_, g_bn_d4_, g_bn_d5_, g_bn_d6_, g_bn_d7_


global d_bn1_, d_bn2_, d_bn3_
d_bn1_ = batch_norm(name='d_bn1_')
d_bn2_ = batch_norm(name='d_bn2_')
d_bn3_ = batch_norm(name='d_bn3_')

g_bn_e2_ = batch_norm(name='g_bn_e2_')
g_bn_e3_ = batch_norm(name='g_bn_e3_')
g_bn_e4_ = batch_norm(name='g_bn_e4_')
g_bn_e5_ = batch_norm(name='g_bn_e5_')
g_bn_e6_ = batch_norm(name='g_bn_e6_')
g_bn_e7_ = batch_norm(name='g_bn_e7_')
g_bn_e8_ = batch_norm(name='g_bn_e8_')

g_bn_d1_ = batch_norm(name='g_bn_d1_')
g_bn_d2_ = batch_norm(name='g_bn_d2_')
g_bn_d3_ = batch_norm(name='g_bn_d3_')
g_bn_d4_ = batch_norm(name='g_bn_d4_')
g_bn_d5_ = batch_norm(name='g_bn_d5_')
g_bn_d6_ = batch_norm(name='g_bn_d6_')
g_bn_d7_ = batch_norm(name='g_bn_d7_')


def Noise_transfer_network(content, style, y=None):
    s = params['output_size']
    output_c_dim = 1
    s2, s4, s8, s16, s32, s64, s128 = int(s/2), int(s/4), int(s/8), int(s/16), int(s/32), int(s/64), int(s/128)
    gf_dim = params['gf_dim']
    
    
    with tf.variable_scope("generator1") as globscope:
    
        with tf.variable_scope("content_encoder") as scope:

            # image is (256 x 256 x input_c_dim)
            c_e1 = conv2d(content, gf_dim, name='g_e1_conv')
            # e1 is (128 x 128 x gf_dim)
            c_e2 = g_bn_e2(conv2d(lrelu(c_e1), gf_dim*2, name='g_e2_conv'))
            # e2 is (64 x 64 x gf_dim*2)
            c_e3 = g_bn_e3(conv2d(lrelu(c_e2), gf_dim*4, name='g_e3_conv'))
            # e3 is (32 x 32 x gf_dim*4)
            c_e4 = g_bn_e4(conv2d(lrelu(c_e3), gf_dim*8, name='g_e4_conv'))
            # e4 is (16 x 16 x gf_dim*8)
            c_e5 = g_bn_e5(conv2d(lrelu(c_e4), gf_dim*8, name='g_e5_conv'))
            # e5 is (8 x 8 x gf_dim*8)
            c_e6 = g_bn_e6(conv2d(lrelu(c_e5), gf_dim*8, name='g_e6_conv'))
            # e6 is (4 x 4 x gf_dim*8)
            c_e7 = g_bn_e7(conv2d(lrelu(c_e6), gf_dim*8, name='g_e7_conv'))
            # e7 is (2 x 2 x gf_dim*8)
            c_e8 = g_bn_e8(conv2d(lrelu(c_e7), gf_dim*8, name='g_e8_conv'))
            # e8 is (1 x 1 x gf_dim*8)

        with tf.variable_scope("style_encoder") as scope:

            # image is (256 x 256 x input_c_dim)
            s_e1 = conv2d(style, gf_dim, name='g_e1_conv')
            # e1 is (128 x 128 x gf_dim)
            s_e2 = g_bn_e2(conv2d(lrelu(s_e1), gf_dim*2, name='g_e2_conv'))
            # e2 is (64 x 64 x gf_dim*2)
            s_e3 = g_bn_e3(conv2d(lrelu(s_e2), gf_dim*4, name='g_e3_conv'))
            # e3 is (32 x 32 x gf_dim*4)
            s_e4 = g_bn_e4(conv2d(lrelu(s_e3), gf_dim*8, name='g_e4_conv'))
            # e4 is (16 x 16 x gf_dim*8)
            s_e5 = g_bn_e5(conv2d(lrelu(s_e4), gf_dim*8, name='g_e5_conv'))
            # e5 is (8 x 8 x gf_dim*8)
            s_e6 = g_bn_e6(conv2d(lrelu(s_e5), gf_dim*8, name='g_e6_conv'))
            # e6 is (4 x 4 x gf_dim*8)
            s_e7 = g_bn_e7(conv2d(lrelu(s_e6), gf_dim*8, name='g_e7_conv'))
            # e7 is (2 x 2 x gf_dim*8)
            s_e8 = g_bn_e8(conv2d(lrelu(s_e7), gf_dim*8, name='g_e8_conv'))

        m_e8 = tf.concat([c_e8, s_e8],-1)

        with tf.variable_scope("decoder") as scope:

            batch_size = params['batch_size']
            d1, d1_w, d1_b = deconv2d(tf.nn.relu(m_e8),
                [batch_size, s128, s128, gf_dim*8], name='g_d1', with_w=True)
            d1 = tf.nn.dropout(g_bn_d1(d1), 0.5)
            d1 = tf.concat([d1, c_e7], 3)
            # d1 is (2 x 2 x gf_dim*8*2)

            d2, d2_w, d2_b = deconv2d(tf.nn.relu(d1),
                [batch_size, s64, s64, gf_dim*8], name='g_d2', with_w=True)
            d2 = tf.nn.dropout(g_bn_d2(d2), 0.5)
            d2 = tf.concat([d2, c_e6], 3)
            # d2 is (4 x 4 x gf_dim*8*2)

            d3, d3_w, d3_b = deconv2d(tf.nn.relu(d2),
                [batch_size, s32, s32, gf_dim*8], name='g_d3', with_w=True)
            d3 = tf.nn.dropout(g_bn_d3(d3), 0.5)
            d3 = tf.concat([d3, c_e5], 3)
            # d3 is (8 x 8 x gf_dim*8*2)

            d4, d4_w, d4_b = deconv2d(tf.nn.relu(d3),
                [batch_size, s16, s16, gf_dim*8], name='g_d4', with_w=True)
            d4 = g_bn_d4(d4)
            d4 = tf.concat([d4, c_e4], 3)
            # d4 is (16 x 16 x gf_dim*8*2)

            d5, d5_w, d5_b = deconv2d(tf.nn.relu(d4),
                [batch_size, s8, s8, gf_dim*4], name='g_d5', with_w=True)
            d5 = g_bn_d5(d5)
            d5 = tf.concat([d5, c_e3], 3)
            # d5 is (32 x 32 x gf_dim*4*2)

            d6, d6_w, sd6_b = deconv2d(tf.nn.relu(d5),
                [batch_size, s4, s4, gf_dim*2], name='g_d6', with_w=True)
            d6 = g_bn_d6(d6)
            d6 = tf.concat([d6, c_e2], 3)
            # d6 is (64 x 64 x gf_dim*2*2)

            d7, d7_w, d7_b = deconv2d(tf.nn.relu(d6),
                [batch_size, s2, s2, gf_dim], name='g_d7', with_w=True)
            d7 = g_bn_d7(d7)
            d7 = tf.concat([d7, c_e1], 3)
            # d7 is (128 x 128 x gf_dim*1*2)

            d8, d8_w, d8_b = deconv2d(tf.nn.relu(d7),
                [batch_size, s, s, output_c_dim], name='g_d8', with_w=True)
            # d8 is (256 x 256 x output_c_dim)

        return tf.nn.tanh(d8)



def Noise_remover_network(image, y=None):
    
    s = params['output_size']
    output_c_dim = 1
    s2, s4, s8, s16, s32, s64, s128 = int(s/2), int(s/4), int(s/8), int(s/16), int(s/32), int(s/64), int(s/128)
    gf_dim = params['gf_dim']
        
    with tf.variable_scope("generator2", reuse = tf.AUTO_REUSE) as scope:

        # image is (256 x 256 x input_c_dim)
        e1 = conv2d(image, gf_dim, name='g_e1_conv')
        # e1 is (128 x 128 x gf_dim)
        e2 = g_bn_e2_(conv2d(lrelu(e1), gf_dim*2, name='g_e2_conv'))
        # e2 is (64 x 64 x gf_dim*2)
        e3 = g_bn_e3_(conv2d(lrelu(e2), gf_dim*4, name='g_e3_conv'))
        # e3 is (32 x 32 x gf_dim*4)
        e4 = g_bn_e4_(conv2d(lrelu(e3), gf_dim*8, name='g_e4_conv'))
        # e4 is (16 x 16 x gf_dim*8)
        e5 = g_bn_e5_(conv2d(lrelu(e4), gf_dim*8, name='g_e5_conv'))
        # e5 is (8 x 8 x gf_dim*8)
        e6 = g_bn_e6_(conv2d(lrelu(e5), gf_dim*8, name='g_e6_conv'))
        # e6 is (4 x 4 x gf_dim*8)
        e7 = g_bn_e7_(conv2d(lrelu(e6), gf_dim*8, name='g_e7_conv'))
        # e7 is (2 x 2 x gf_dim*8)
        e8 = g_bn_e8_(conv2d(lrelu(e7), gf_dim*8, name='g_e8_conv'))
        # e8 is (1 x 1 x gf_dim*8)
        
        batch_size = params['batch_size']
        d1, d1_w, d1_b = deconv2d(tf.nn.relu(e8),
            [batch_size, s128, s128, gf_dim*8], name='g_d1', with_w=True)
        d1 = tf.nn.dropout(g_bn_d1_(d1), 0.5)
        d1 = tf.concat([d1, e7], 3)
        # d1 is (2 x 2 x gf_dim*8*2)

        d2, d2_w, d2_b = deconv2d(tf.nn.relu(d1),
            [batch_size, s64, s64, gf_dim*8], name='g_d2', with_w=True)
        d2 = tf.nn.dropout(g_bn_d2_(d2), 0.5)
        d2 = tf.concat([d2, e6], 3)
        # d2 is (4 x 4 x gf_dim*8*2)

        d3, d3_w, d3_b = deconv2d(tf.nn.relu(d2),
            [batch_size, s32, s32, gf_dim*8], name='g_d3', with_w=True)
        d3 = tf.nn.dropout(g_bn_d3_(d3), 0.5)
        d3 = tf.concat([d3, e5], 3)
        # d3 is (8 x 8 x gf_dim*8*2)

        d4, d4_w, d4_b = deconv2d(tf.nn.relu(d3),
            [batch_size, s16, s16, gf_dim*8], name='g_d4', with_w=True)
        d4 = g_bn_d4_(d4)
        d4 = tf.concat([d4, e4], 3)
        # d4 is (16 x 16 x gf_dim*8*2)

        d5, d5_w, d5_b = deconv2d(tf.nn.relu(d4),
            [batch_size, s8, s8, gf_dim*4], name='g_d5', with_w=True)
        d5 = g_bn_d5_(d5)
        d5 = tf.concat([d5, e3], 3)
        # d5 is (32 x 32 x gf_dim*4*2)

        d6, d6_w, sd6_b = deconv2d(tf.nn.relu(d5),
            [batch_size, s4, s4, gf_dim*2], name='g_d6', with_w=True)
        d6 = g_bn_d6_(d6)
        d6 = tf.concat([d6, e2], 3)
        # d6 is (64 x 64 x gf_dim*2*2)

        d7, d7_w, d7_b = deconv2d(tf.nn.relu(d6),
            [batch_size, s2, s2, gf_dim], name='g_d7', with_w=True)
        d7 = g_bn_d7_(d7)
        d7 = tf.concat([d7, e1], 3)
        # d7 is (128 x 128 x gf_dim*1*2)

        d8, d8_w, d8_b = deconv2d(tf.nn.relu(d7),
            [batch_size, s, s, output_c_dim], name='g_d8', with_w=True)
        # d8 is (256 x 256 x output_c_dim)

        #return tf.nn.tanh(d8[:,:,:,:3]), tf.nn.tanh(d8[:,:,:,3:4])  #(w/o text , bin text)
        return tf.nn.tanh(d8)



def discriminator1(image, y=None, reuse=False):
    
    df_dim = params['df_dim']
    batch_size = params['batch_size']
    with tf.variable_scope("discriminator1") as scope:

        # image is 256 x 256 x (input_c_dim + output_c_dim)
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse == False

        h0 = lrelu(conv2d(image, df_dim, name='d_h0_conv'))
        # h0 is (128 x 128 x df_dim)
        h1 = lrelu(d_bn1(conv2d(h0, df_dim*2, name='d_h1_conv')))
        # h1 is (64 x 64 x df_dim*2)
        h2 = lrelu(d_bn2(conv2d(h1, df_dim*4, name='d_h2_conv')))
        # h2 is (32x 32 x df_dim*4)
        h3 = lrelu(d_bn3(conv2d(h2, df_dim*8, d_h=1, d_w=1, name='d_h3_conv')))
        # h3 is (16 x 16 x df_dim*8)
        h4 = linear(tf.reshape(h3, [batch_size, -1]), 1, 'd_h3_lin')

        return tf.nn.sigmoid(h4), h4
    

def discriminator2(image, y=None, reuse=False):
    df_dim = params['df_dim']
    batch_size = params['batch_size']
    with tf.variable_scope("discriminator2") as scope:

        # image is 256 x 256 x (input_c_dim + output_c_dim)
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse == False

        h0 = lrelu(conv2d(image, df_dim, name='d_h0_conv'))
        # h0 is (128 x 128 x df_dim)
        h1 = lrelu(d_bn1(conv2d(h0, df_dim * 2, name='d_h1_conv')))
        # h1 is (64 x 64 x df_dim*2)
        h2 = lrelu(d_bn2(conv2d(h1, df_dim * 4, name='d_h2_conv')))
        # h2 is (32x 32 x df_dim*4)
        h3 = lrelu(d_bn3(conv2d(h2, df_dim * 8, d_h=1, d_w=1, name='d_h3_conv')))
        # h3 is (16 x 16 x df_dim*8)
        h4 = linear(tf.reshape(h3, [batch_size, -1]), 1, 'd_h3_lin')

        return tf.nn.sigmoid(h4), h4

		
def get_content_loss(output, content):
    
    #tf.summary.image('mask', (1 - content),  max_outputs=1)
    
    #output = (output+1)/2
    
    masked_ = (1 - content) * tf.square(output - content)
    
    #masked_ = (1 - content) * output
    #tf.summary.image('masked', masked_, max_outputs=1)
    
    return tf.reduce_mean(masked_)
 



def get_style_loss(output, style):
    
    style = (tf.image.grayscale_to_rgb(style)+1)/2

    output = (tf.image.grayscale_to_rgb(output)+1)/2
    
    data_dict = loadWeightsData('./vgg16.npy')
    
    def gram_matrix(x):
        assert isinstance(x, tf.Tensor)
        b, h, w, ch = x.get_shape().as_list()
        features = tf.reshape(x, [b, h*w, ch])
        # gram = tf.batch_matmul(features, features, adj_x=True)/tf.constant(ch*w*h, tf.float32)
        gram = tf.matmul(features, features, adjoint_a=True)/tf.constant(ch*w*h, tf.float32)
        return gram
    
    vgg_o = custom_Vgg16(output, data_dict=data_dict)
    feature_o = [vgg_o.conv1_2, vgg_o.conv2_2, vgg_o.conv3_3, vgg_o.conv4_3, vgg_o.conv5_3]
    gram_o = [gram_matrix(l) for l in feature_o]
    vgg_s = custom_Vgg16(style, data_dict=data_dict)
    feature_s = [vgg_s.conv1_2, vgg_s.conv2_2, vgg_s.conv3_3, vgg_s.conv4_3, vgg_s.conv5_3]
    gram_s = [gram_matrix(l) for l in feature_s]
    loss_s = tf.zeros(params['batch_size'], tf.float32)
    for g, g_ in zip(gram_o, gram_s):
        loss_s += tf.reduce_mean(tf.subtract(g, g_) ** 2, [1, 2])
        
    return tf.reduce_mean(loss_s)


from tensorflow.python.framework import ops
ops.reset_default_graph()

global sess

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

sess = tf.Session()
graph = tf.get_default_graph()

content = tf.placeholder(dtype=tf.float32, shape=[params['batch_size'],256,256,1], name = 'content')
style = tf.placeholder(dtype=tf.float32, shape=[params['batch_size'],256,256,1], name = 'style')
Real_input = tf.placeholder(dtype=tf.float32, shape=[params['batch_size'],256,256,1], name = 'Real_input')

output = Noise_transfer_network(content, style)

Cleaned = Noise_remover_network(output)

# For testing
Real_cleaned = Noise_remover_network(Real_input)

tf.summary.image("Content_Image", content, max_outputs=1)
tf.summary.image("Style_Image", style, max_outputs=1)
tf.summary.image("Generated_Noise", output, max_outputs=1)
tf.summary.image("Cleaned_output", Cleaned, max_outputs=1)

## stage1 losses

D1_real,D1_real_logits = discriminator1(style, reuse=False)
D1_fake,D1_fake_logits = discriminator1(output, reuse=True)

d1_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D1_real_logits, labels=tf.ones_like(D1_real)))
d1_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D1_fake_logits, labels=tf.zeros_like(D1_fake)))
d1_loss = d1_loss_real + d1_loss_fake
g1_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D1_fake_logits, labels=tf.ones_like(D1_fake)))

content_loss = get_content_loss(output, content)
style_loss = get_style_loss(output, style)

g1_loss = g1_loss + 10*content_loss + 0.5*style_loss 

## stage2 losses

D2_real,D2_real_logits = discriminator2(content, reuse=False)
D2_fake,D2_fake_logits = discriminator2(Cleaned, reuse=True)

d2_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D2_real_logits, labels=tf.ones_like(D2_real)))
d2_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D2_fake_logits, labels=tf.zeros_like(D2_fake)))
d2_loss = d2_loss_real + d2_loss_fake
g2_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D2_fake_logits, labels=tf.ones_like(D2_fake)))

g2_loss = g2_loss + 100 * tf.reduce_mean(tf.abs(content - Cleaned))

## Combined stage1 & stage2 loss

g1_loss_e2e = g1_loss 
d1_loss_e2e = d1_loss

g2_loss_e2e = g2_loss
d2_loss_e2e = d2_loss

t_vars = tf.trainable_variables()

g1_vars = [var for var in t_vars if 'generator1' in var.name]
g2_vars = [var for var in t_vars if 'generator2' in var.name]

d1_vars = [var for var in t_vars if 'discriminator1' in var.name]
d2_vars = [var for var in t_vars if 'discriminator2' in var.name]


d1_optim = tf.train.AdamOptimizer(params['lr'], beta1=params['beta_1'])
g1_optim = tf.train.AdamOptimizer(params['lr'], beta1=params['beta_1'])
d2_optim = tf.train.AdamOptimizer(params['lr'], beta1=params['beta_1'])
g2_optim = tf.train.AdamOptimizer(params['lr'], beta1=params['beta_1'])


# stage1 training

d1_grads = tf.gradients(d1_loss, d1_vars)
d1_train = d1_optim.apply_gradients(zip(d1_grads, d1_vars))

g1_grads = tf.gradients(g1_loss, g1_vars)
g1_train = g1_optim.apply_gradients(zip(g1_grads, g1_vars))

# stage2 training

d2_grads = tf.gradients(d2_loss, d2_vars)
d2_train = d2_optim.apply_gradients(zip(d2_grads, d2_vars))

g2_grads = tf.gradients(g2_loss, g2_vars)
g2_train = g2_optim.apply_gradients(zip(g2_grads, g2_vars))

# Combined training

d1_grads_e2e = tf.gradients(d1_loss_e2e, d1_vars)
d1_train_e2e = d1_optim.apply_gradients(zip(d1_grads_e2e, d1_vars))

g1_grads_e2e = tf.gradients(g1_loss_e2e, g1_vars)
g1_train_e2e = g1_optim.apply_gradients(zip(g1_grads_e2e, g1_vars))

d2_grads_e2e = tf.gradients(d2_loss_e2e, d2_vars)
d2_train_e2e = d2_optim.apply_gradients(zip(d2_grads_e2e, d2_vars))

g2_grads_e2e = tf.gradients(g2_loss_e2e, g2_vars)
g2_train_e2e = g2_optim.apply_gradients(zip(g2_grads_e2e, g2_vars))



init_op = tf.global_variables_initializer()
sess.run(init_op)

saver = tf.train.Saver()
load_weights(saver, params['model_path'])

summary = tf.summary.merge_all()
trainwriter = tf.summary.FileWriter("logs", sess.graph)

counter = 1
start_time = time.time()
train_STYLEpath, train_CONTENTpath, test_STYLEpath, test_CONTENTpath = get_file_paths(params['path'])

np.random.shuffle(train_STYLEpath)
np.random.shuffle(train_CONTENTpath)

data_set_size = len(train_STYLEpath)

print (data_set_size//params['batch_size'])

rndm_test_style = test_STYLEpath[np.random.choice(len(test_STYLEpath),params['batch_size']*10, replace = False)]
rndm_test_content = test_CONTENTpath[np.random.choice(len(test_CONTENTpath),params['batch_size']*10, replace = False)]

Fixd_rndm_indx = np.random.choice(len(test_STYLEpath),params['batch_size']*10, replace = False)

paired_smaple_input = test_STYLEpath[Fixd_rndm_indx]
paired_smaple_output = test_CONTENTpath[Fixd_rndm_indx]

for epoch in range(params['epochs']):

    #print("Epoch:{}".format(epoch))

    for idx in range(data_set_size//params['batch_size']):

        batch_STYLEpath = train_STYLEpath[idx * params['batch_size'] : (idx + 1) * params['batch_size']]
        batch_STYLEdata = np.array([load_data(path) for path in batch_STYLEpath])
        batch_CONTENTpath = train_CONTENTpath[idx * params['batch_size']: (idx + 1) * params['batch_size']]
        batch_CONTENTdata = np.array([load_data_CONTENT(path) for path in batch_CONTENTpath])

        feed_dict = {content : batch_CONTENTdata , style : batch_STYLEdata}


        if counter<=params['Stage_epochs'][0]:

            ## stage1 training
            _, d1_loss_ = sess.run([d1_train, d1_loss], feed_dict) # update D1 network
            _, g1_loss_ = sess.run([g1_train, g1_loss], feed_dict) # update G1 network
            _,summary_, g1_loss_ = sess.run([g1_train,summary, g1_loss], feed_dict) # update G1 network

            if counter%10 == 0:

                print ('## stage1 training ## : idx : ' +str(counter) + ' Dis1 loss : '+str(d1_loss_) + ' Gen1 loss : '+str(g1_loss_))

        elif counter>params['Stage_epochs'][0] and counter<=params['Stage_epochs'][1]:


            ## stage2 training
            _, d2_loss_ = sess.run([d2_train, d2_loss], feed_dict) # update D2 network
            _, g2_loss_ = sess.run([g2_train, g2_loss], feed_dict) # update G2 network
            _,summary_, g2_loss_ = sess.run([g2_train,summary, g2_loss], feed_dict) # update G2 network

            if counter%10 == 0:

                print ('## stage2 training ## : idx : ' +str(counter) + ' Dis2 loss : '+str(d2_loss_) + ' Gen2 loss : '+str(g2_loss_))

        else:
  
            ## combined training
            _, d1_loss_e2e_ = sess.run([d1_train_e2e, d1_loss], feed_dict) # update D1 network
            _, g1_loss_e2e_ = sess.run([g1_train_e2e, g1_loss], feed_dict) # update G1 network
            _,summary_, g1_loss_e2e_ = sess.run([g1_train_e2e, summary, g1_loss], feed_dict) # update G1 network

            _, d2_loss_e2e_ = sess.run([d2_train_e2e, d2_loss], feed_dict) # update D1 network
            _, g2_loss_e2e_ = sess.run([g2_train_e2e, g2_loss], feed_dict) # update G1 network
            _,summary_, g2_loss_e2e_ = sess.run([g2_train_e2e, summary, g2_loss], feed_dict) # update G1 network

            if counter%10 == 0:

                print ('## Combined training ## : idx : ' +str(counter) + ' Dis1 loss : '+str(d1_loss_e2e_) + ' Gen1 loss : '+str(g1_loss_e2e_)+ ' Dis2 loss : '+str(d2_loss_e2e_) + ' Gen2 loss : '+str(g2_loss_e2e_))

        trainwriter.add_summary(summary_, counter)
        
        
        
        if counter%500==0:
            
            
            
            if not os.path.isdir(os.path.join(params['Img_saved_path'], str(counter))):
                os.mkdir(os.path.join(params['Img_saved_path'], str(counter)))
                
            if not os.path.isdir(os.path.join(params['Img_saved_path_for_real_data'], str(counter))):
                os.mkdir(os.path.join(params['Img_saved_path_for_real_data'], str(counter)))
            
            cc = 1
            
            for k in range(10):
                
                batchx = paired_smaple_input[k * params['batch_size'] : (k + 1) * params['batch_size']]
                batchx_data = np.array([load_data(path) for path in batchx])
                batchy = paired_smaple_output[k * params['batch_size'] : (k + 1) * params['batch_size']]
                batchy_data = np.array([load_data_CONTENT(path) for path in batchy])

                feed_dict = {Real_input : batchx_data}
                
                Real_cleaned_ = sess.run(Real_cleaned, feed_dict)
                
                all_imgs = np.concatenate((batchx_data, batchy_data, Real_cleaned_),2)
                
                for no_i in range(params['batch_size']):
                    
                    imageio.imwrite(os.path.join(params['Img_saved_path_for_real_data'], str(counter), str(cc) + ".jpg"), all_imgs[no_i,:,:,:])
                
                    cc = cc + 1

            cc = 1     
                    
            for k in range(10):
                
                batch_STYLEpath = rndm_test_style[k * params['batch_size'] : (k + 1) * params['batch_size']]
                batch_STYLEdata = np.array([load_data(path) for path in batch_STYLEpath])
                batch_CONTENTpath = rndm_test_content[k * params['batch_size']: (k + 1) * params['batch_size']]
                batch_CONTENTdata = np.array([load_data_CONTENT(path) for path in batch_CONTENTpath])

                feed_dict = {content : batch_CONTENTdata , style : batch_STYLEdata}
                
                _cleaned, _output = sess.run([Cleaned, output], feed_dict)
                
                all_imgs = np.concatenate((batch_CONTENTdata, batch_STYLEdata, _output, _cleaned),2)
                
                for no_i in range(params['batch_size']):
                    
                    imageio.imwrite(os.path.join(params['Img_saved_path'], str(counter), str(cc) + ".jpg"), all_imgs[no_i,:,:,:])
                
                    cc = cc + 1

        counter = counter + 1
        if counter % 1000 == 0:
            save(saver, params['model_path'], counter)
            print('Model Saved!!')

