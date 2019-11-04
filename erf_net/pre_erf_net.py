# Author Lingge Li from XJTU(446049454@qq.com)
# pre_erf_net


import tensorflow as tf
from erf_net_slim import *
from config import cfg
from imageNet_config import imageNet_config

mc = imageNet_config()
SAVE_VARIABLES = 'save_variables'

def _variable_on_device(name, shape, initializer, regularizer = None, trainable=True):
    dtype = 'float'
    collections = [tf.GraphKeys.GLOBAL_VARIABLES, SAVE_VARIABLES]
    with tf.device('/cpu:0'):
        var = tf.get_variable(
            name, shape, initializer=initializer, dtype=dtype, regularizer = regularizer, collections = collections, trainable=trainable)
    return var

def _variable_with_weight_decay(name, shape, initializer, regularizer = None,  trainable=True):
    dtype = 'float'
    collections = [tf.GraphKeys.GLOBAL_VARIABLES, SAVE_VARIABLES]
    with tf.device('/cpu:0'):
        var = tf.get_variable(
            name, shape, initializer=initializer, dtype=dtype, regularizer = regularizer, collections = collections, trainable=trainable)
    return var


def global_average_pooling(layer_name, inputs, stride, padding = 'VALID'):
    batch_num, height, width, channels = inputs.get_shape().as_list() 
    with tf.variable_scope(layer_name) as scope:
        out = tf.nn.avg_pool(inputs, 
                          ksize = [1, height, width, 1],
                          strides = [1,stride, stride,1],
                          padding = padding)
        out = tf.reduce_mean(out, [1,2])
    return out


def conv_layer(layer_name, inputs, filters, size, stride, padding='SAME',freeze=False, xavier=False, relu=True, stddev=0.001):
    wd = mc.WEIGHT_DECAY

    with tf.variable_scope(layer_name) as scope:
        channels = inputs.get_shape()[3]
        kernel_init = tf.truncated_normal_initializer(
            stddev=stddev, dtype=tf.float32)
        bias_init = tf.constant_initializer(0.0)

        kernel = _variable_with_weight_decay(
             'kernels', shape=[size, size, int(channels), filters], initializer=kernel_init, regularizer=tf.contrib.layers.l2_regularizer(wd),trainable=(not freeze))

        biases = _variable_on_device('biases', [filters], bias_init,
                               trainable=(not freeze))
        conv = tf.nn.conv2d(
            inputs, kernel, [1, stride, stride, 1], padding=padding,
            name='convolution')
        conv_bias = tf.nn.bias_add(conv, biases, name='bias_add')

        if relu:
            out = tf.nn.relu(conv_bias, 'relu')
        else:
            out = conv_bias

        return out

def build_encoder(input_data):
    input_shape = input_data.get_shape().as_list()
    print('encoder input shape: [' + ','.join([str(x) for x in input_shape]) + ']')
    is_training = True
    loss_collection = 'loss'
    # downsampler 1
    output = downsampler(input_data, 16, initializer=msra_initializer(), wd=cfg.WEIGHT_DECAY,
                        use_bn=cfg.USE_BN, is_training=is_training, bn_decay=0.99,
                        loss_collection=loss_collection, name='l1_downsampler')

    # downsampler 2
    output = downsampler(output, 64, initializer=msra_initializer(), wd=cfg.WEIGHT_DECAY,
                        use_bn=cfg.USE_BN, is_training=is_training, bn_decay=0.99,
                        loss_collection=loss_collection, name='l2_dowansampler')

    # nonbt1d_3
    output = res_nonbt1d(output, 3, initializer=msra_initializer(), wd=cfg.WEIGHT_DECAY, use_bn=cfg.USE_BN,
                        is_training=is_training, bn_decay=0.99, use_dropout=cfg.USE_DROPOUT,
                        drop_rate=cfg.DROP_RATE, loss_collection=loss_collection, name='l3_nonbt1d')
        
    # nonbt1d_4
    output = res_nonbt1d(output, 3, initializer=msra_initializer(), wd=cfg.WEIGHT_DECAY, use_bn=cfg.USE_BN,
                        is_training=is_training, bn_decay=0.99, use_dropout=cfg.USE_DROPOUT,
                        drop_rate=cfg.DROP_RATE, loss_collection=loss_collection, name='l4_nonbt1d')
        
    # nonbt1d_5
    output = res_nonbt1d(output, 3, initializer=msra_initializer(), wd=cfg.WEIGHT_DECAY, use_bn=cfg.USE_BN,
                        is_training=is_training, bn_decay=0.99, use_dropout=cfg.USE_DROPOUT,
                        drop_rate=cfg.DROP_RATE, loss_collection=loss_collection, name='l5_nonbt1d')
        
    # nonbt1d_6
    output = res_nonbt1d(output, 3, initializer=msra_initializer(), wd=cfg.WEIGHT_DECAY, use_bn=cfg.USE_BN,
                        is_training=is_training, bn_decay=0.99, use_dropout=cfg.USE_DROPOUT,
                        drop_rate=cfg.DROP_RATE, loss_collection=loss_collection, name='l6_nonbt1d')
        
    # nonbt1d_7
    output = res_nonbt1d(output, 3, initializer=msra_initializer(), wd=cfg.WEIGHT_DECAY, use_bn=cfg.USE_BN,
                        is_training=is_training, bn_decay=0.99, use_dropout=cfg.USE_DROPOUT,
                        drop_rate=cfg.DROP_RATE, loss_collection=loss_collection, name='l7_nonbt1d')
        
    # downsampler 8
    output = downsampler(output, 128, initializer=msra_initializer(), wd=cfg.WEIGHT_DECAY,
                        use_bn=cfg.USE_BN, is_training=is_training, bn_decay=0.99,
                        loss_collection=loss_collection, name='l8_dowansampler')

    # nonbt1d_9(dilated 2)
    output = res_nonbt1d(output, 3, True, 2, initializer=msra_initializer(), wd=cfg.WEIGHT_DECAY, use_bn=cfg.USE_BN,
                        is_training=is_training, bn_decay=0.99, use_dropout=cfg.USE_DROPOUT,
                        drop_rate=cfg.DROP_RATE, loss_collection=loss_collection, name='l9_nonbt1d')
        
    # nonbt1d_10(dilated 4)
    output = res_nonbt1d(output, 3, True, 4, initializer=msra_initializer(), wd=cfg.WEIGHT_DECAY, use_bn=cfg.USE_BN,
                        is_training=is_training, bn_decay=0.99, use_dropout=cfg.USE_DROPOUT,
                        drop_rate=cfg.DROP_RATE, loss_collection=loss_collection, name='l10_nonbt1d')
        
    # nonbt1d_11(dilated 8)
    output = res_nonbt1d(output, 3, True, 8, initializer=msra_initializer(), wd=cfg.WEIGHT_DECAY, use_bn=cfg.USE_BN,
                        is_training=is_training, bn_decay=0.99, use_dropout=cfg.USE_DROPOUT,
                        drop_rate=cfg.DROP_RATE, loss_collection=loss_collection, name='l11_nonbt1d')

    # nonbt1d_12(dilated 16)
    output = res_nonbt1d(output, 3, True, 16, initializer=msra_initializer(), wd=cfg.WEIGHT_DECAY, use_bn=cfg.USE_BN,
                        is_training=is_training, bn_decay=0.99, use_dropout=cfg.USE_DROPOUT,
                        drop_rate=cfg.DROP_RATE, loss_collection=loss_collection, name='l2_nonbt1d')
        
    # nonbt1d_13(dilated 2)
    output = res_nonbt1d(output, 3, True, 2, initializer=msra_initializer(), wd=cfg.WEIGHT_DECAY, use_bn=cfg.USE_BN,
                        is_training=is_training, bn_decay=0.99, use_dropout=cfg.USE_DROPOUT,
                        drop_rate=cfg.DROP_RATE, loss_collection=loss_collection, name='l13_nonbt1d')
        
    # nonbt1d_14(dialted 4)
    output = res_nonbt1d(output, 3, True, 4, initializer=msra_initializer(), wd=cfg.WEIGHT_DECAY, use_bn=cfg.USE_BN,
                        is_training=is_training, bn_decay=0.99, use_dropout=cfg.USE_DROPOUT,
                        drop_rate=cfg.DROP_RATE, loss_collection=loss_collection, name='l14_nonbt1d')
        
    # nonbt1d_15(dilated 8)
    output = res_nonbt1d(output, 3, True, 8, initializer=msra_initializer(), wd=cfg.WEIGHT_DECAY, use_bn=cfg.USE_BN,
                        is_training=is_training, bn_decay=0.99, use_dropout=cfg.USE_DROPOUT,
                        drop_rate=cfg.DROP_RATE, loss_collection=loss_collection, name='l15_nonbt1d')
        
    # nonbt1d_16(dilated 16)
    output = res_nonbt1d(output, 3, True, 16, initializer=msra_initializer(), wd=cfg.WEIGHT_DECAY, use_bn=cfg.USE_BN,
                        is_training=is_training, bn_decay=0.99, use_dropout=cfg.USE_DROPOUT,
                        drop_rate=cfg.DROP_RATE, loss_collection=loss_collection, name='l16_nonbt1d')

    output = conv_layer('fc_conv_last', output , filters=1000, size=3, stride=1, padding='SAME', xavier=False, relu=False, stddev=0.0001)
    net = global_average_pooling('global_average_pooling', output, stride = 1)
    
    return net






