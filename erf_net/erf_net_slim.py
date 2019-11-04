# Author Lingge Li from XJTU(446049454@qq.com)
# erf_net functions

import tensorflow as tf
import numpy as np
import sys
import math



######### Initializers ###########

def msra_initializer():
    """ [ref] K. He et.al 2015: arxiv:1502.01852 
    """
    def _initializer(shape, dtype=tf.float32, partition_info=None):
        k = shape[0] * shape[1]        # kernel size
        d = shape[3]                   # filter number

        stddev = math.sqrt(2. / (k**2 * d))
        return tf.truncated_normal(shape, stddev=stddev, dtype=dtype)

    return _initializer

def orthogonal_initializer(scale = 1.1):
    """ From Lasagne and Keras. Reference: Saxe et al., http://arxiv.org/abs/1312.6120
    """
    def _initializer(shape, dtype=tf.float32, partition_info=None):
        flat_shape = (np.prod(shape[:-1]), shape[-1])
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        # pick the one with the correct shape
        q = u if u.shape == flat_shape else v
        q = q.reshape(shape) #this needs to be corrected to float32
        return tf.constant(scale * q[:shape[0], :shape[1]], dtype=dtype)

    return _initializer

def bilinear_initializer():

    def _initializer(shape, dtype=tf.float32, partition_info=None):
        width = shape[0]
        heigh = shape[1]
        fw = width/2.0
        fh = heigh/2.0
        cw = (width - 1) / (2.0 * fw)
        ch = (heigh - 1) / (2.0 * fh)

        bilinear = np.zeros([f_shape[0], f_shape[1]])
        for x in range(width):
            for y in range(heigh):
                value = (1 - abs(x / fw - cw)) * (1 - abs(y / fh - ch))
                bilinear[x, y] = value

        weights = np.zeros(shape)
        for i in range(shape[2]):
            for j in range(shape[3]):
                weights[:, :, i, j] = bilinear

        return tf.constant(weights, dtype=dtype)

    return _initializer


##################################


####### Varible Helpers ##########

def variable_on_cpu(name, shape, initializer):
    # to create a Variable stored on CPU memory

    with tf.device('/cpu:0'):
        value = tf.get_variable(name, shape, initializer=initializer)
    return value

def variable_with_weight_decay(name, shape, initializer, wd=None, use_cpu=False, loss_collection='losses'):
    # to create an initialized Variable with weight decay
    if use_cpu:
        value = variable_on_cpu(name, shape, initializer)
    else:
        value = tf.get_variable(name, shape, initializer=initializer)
    
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(value), wd, name='weight_decay_loss')
        tf.add_to_collection(loss_collection, weight_decay)
    return value
##################################


######### erf_net functions ###########

def get_deconv_output_feature_map_shape(ori_img_h, ori_img_w, downsample_pow):
    fmap_h = ori_img_h
    fmap_w = ori_img_w
    for i in range(downsample_pow-1):
        # return min int[x >= math.ceil(...)] 
        fmap_h = math.ceil(fmap_h / 2.0)
        fmap_w = math.ceil(fmap_w / 2.0)
    return (int(fmap_h), int(fmap_w))


def bacth_norm_layer(input_data, is_training, decay, name):
    
    return tf.contrib.layers.batch_norm(input_data, decay=decay, center=True, scale=True, epsilon=1e-5,
                                        activation_fn=tf.nn.relu, is_training=is_training, scope=name+'_bn')


def conv_bn_layer(input_data, shape, stride = 1, initializer=orthogonal_initializer(), wd=2e-4,
                  use_bias=False, use_bn=False, is_training=None, bn_decay=0.99, use_cpu=False,
                  loss_collection='losses', name=None):
    # bias is meaningless when using bacth normalization
    if use_bn:
        use_bias = False
    #print('conv_bn_layer use_bn', use_bn)
    #print('conv_bn_layer use_bias: ', use_bias)
     # kernel_shape = [kernelsize, kernelsize, inputchannels, outputchannels]
    out_channels = shape[3]
    strides = [1, stride, stride, 1]
    with tf.variable_scope(name) as scope:
        kernel = variable_with_weight_decay('weights', shape=shape, initializer=initializer, wd=wd,
                                            use_cpu=use_cpu, loss_collection=loss_collection)
        conv = tf.nn.conv2d(input_data, kernel, strides, padding='SAME')

        if use_bias:
            print('wo kao')
            if use_cpu:
                biases = variable_on_cpu('biases', [out_channels], tf.constant_initializer(0.0))
            else:
                biases = tf.get_variable('biases', [out_channels], initializer=tf.constant_initializer(0.0))
            conv = tf.nn.bias_add(conv, biases)
        if use_bn:
            if is_training is None:
                print('is_training is None when use bn layer!')
                sys.exit(-1)
            conv = bacth_norm_layer(conv, is_training, bn_decay, scope.name)
        
        conv = tf.nn.relu(conv)
    
    return conv

def dilated_conv_bn_layer(input_data, shape, rate, initializer=orthogonal_initializer(), wd=2e-4, use_bias=False,
                          use_bn=False, is_training=None, bn_decay=0.99, use_cpu=False, loss_collection='losses', name=None):
    # bias is meaningless when using bacth normalization
    use_bias = True
    if use_bn:
        use_bias = False
    
    out_channels = shape[3]
    with tf.variable_scope(name) as scope:
        kernel = variable_with_weight_decay('weights', shape=shape, initializer=initializer, wd=wd,
                                            use_cpu=use_cpu, loss_collection=loss_collection)
        dilated = tf.nn.atrous_conv2d(input_data, kernel, rate, padding='SAME')

        if use_bias:
            if use_cpu:
                biases = variable_on_cpu('biases', [out_channels], tf.constant_initializer(0.0))
            else:
                biases = tf.get_variable('biases', [out_channels], initializer=tf.constant_initializer(0.0))
            dilated = tf.nn.bias_add(dilated, biases)
        
        if use_bn:
            if is_training is None:
                print('is_training is None when use bn layer!')
                sys.exit(-1)
            dilated = bacth_norm_layer(dilated, is_training, bn_decay, scope.name)
        
        dilated = tf.nn.relu(dilated)
    
    return dilated


def downsampler(input_data, out_feature_maps, initializer=orthogonal_initializer(), wd=2e-4, use_bn=False, 
                is_training = None, bn_decay=0.99, loss_collection='losses', name='downsampler'):
    # bias is meaningless when using bacth normalization
    use_bias = True
    if use_bn:
        use_bias = False
    #print('downsampler use_bn', use_bn)
    #print('downsampler use_bias', use_bias)
    input_data_shape = input_data.get_shape().as_list()
    input_feature_maps = input_data_shape[3]
    conv_feature_maps = out_feature_maps - input_feature_maps

    with tf.variable_scope(name) as scope:
        # maxpooling part
        pool = tf.nn.max_pool(input_data, [1, 2, 2, 1], [1, 2, 2, 1], padding = 'SAME', name = 'pool')

        # convolution part
        conv = conv_bn_layer(input_data, [3, 3, input_feature_maps, conv_feature_maps], 2, initializer, wd=wd, 
                             use_bias=use_bias, loss_collection=loss_collection, name='conv')
        
        # concat
        tensor = tf.concat([pool, conv], 3)

        if use_bn:
            if is_training is None:
                print('is_training is None when use bn layer!')
                sys.exit(-1)
            tensor = bacth_norm_layer(tensor, is_training, bn_decay, scope.name)
        else:
            tensor = tf.nn.relu(tensor)
    
    return tensor

def res_nonbt1d(input_data, kernel_size, dilate=False, dilate_rate=None, initializer=orthogonal_initializer(),
                wd=2e-4, use_bn=False, is_training=None, bn_decay=0.99, use_dropout=False, drop_rate=0.5,
                use_cpu=False, loss_collection='losses', name='nonbt1d'):
    input_data_shape = input_data.get_shape().as_list()
    channels = input_data_shape[3]
    kernel_shape1 = [kernel_size, 1, channels, channels]
    kernel_shape2 = [1, kernel_size, channels, channels]

    with tf.variable_scope(name) as scope:
        output = conv_bn_layer(input_data, kernel_shape1, initializer=initializer, wd=wd, use_cpu=use_cpu,
                               loss_collection=loss_collection, name='1dconv1')
        
        output = conv_bn_layer(output, kernel_shape2, initializer=initializer, wd=wd, use_bn=use_bn,
                               is_training=is_training, bn_decay=bn_decay, use_cpu=use_cpu,
                               loss_collection=loss_collection, name='1dconv2')
        
        if dilate:
            if dilate_rate is None:
                print('dilate_rate is None when use dilate convolution!')
                sys.exit(-1)
            
            output = dilated_conv_bn_layer(output, kernel_shape1, dilate_rate, initializer=initializer, wd=wd,
                                          use_cpu=use_cpu, loss_collection=loss_collection, name='dilate_1dconv3')
            
            output = dilated_conv_bn_layer(output, kernel_shape2, dilate_rate, initializer=initializer, wd=wd,
                                           use_bn=use_bn, is_training=is_training, bn_decay=bn_decay, use_cpu=use_cpu,
                                           loss_collection=loss_collection, name='dilate_1dconv4')
        else:
            output = conv_bn_layer(output, kernel_shape1, initializer=initializer, wd=wd, use_cpu=use_cpu,
                               loss_collection=loss_collection, name='1dconv3')
            
            output = conv_bn_layer(output, kernel_shape2, initializer=initializer, wd=wd, use_bn=use_bn,
                               is_training=is_training, bn_decay=bn_decay, use_cpu=use_cpu,
                               loss_collection=loss_collection, name='1dconv4')
        
        if use_dropout:
            output = tf.contrib.layers.dropout(output, 1-drop_rate, 
                                               is_training=tf.placeholder_with_default(False, (), name='is_training'),
                                               scope='drop')
        
        # short cut: identity map
        tensor = tf.add(output, input_data, name='merge_map')
        tensor = tf.nn.relu(tensor)

        return tensor

def deconv_bn_layer(input_data, shape, output_shape, stride=2, initializer=bilinear_initializer(),
                    wd=None, use_bias=False, use_bn=False, is_training=None, bn_decay=0.99,
                    use_cpu=False, loss_collection='losses', name=None):
    # bias is meaningless when using batch normalization
    if use_bn:
        use_bias = False
    
    # output_shape = [batchsize, h, w, c]
    # pay attention to tf.nn.conv2d_transpose()!
    # kernel_shape = [kernelsize, kernelsize, outputchannels, inputchannels]
    out_channels = shape[2]
    strides = [1, stride, stride, 1]

    with tf.variable_scope(name) as scope:
        kernel = variable_with_weight_decay('weights', shape, initializer=initializer, wd=wd,
                                            use_cpu=use_cpu, loss_collection=loss_collection)
        deconv = tf.nn.conv2d_transpose(input_data, kernel, output_shape, strides=strides, padding='SAME')

        if use_bias:
            if use_cpu:
                biases = variable_on_cpu('biases', [out_channels], tf.constant_initializer(0.0))
            else:
                biases = tf.get_variable('biases', [out_channels], initializer=tf.constant_initializer(0.0))
        
            deconv =tf.nn.bias_add(deconv, biases)
    
        if use_bn:
            if is_training is None:
                print('is_training is None when use bn layer!')
                sys.exit(-1)
            deconv = bacth_norm_layer(deconv, is_training, bn_decay, scope.name)
    
        deconv = tf.nn.relu(deconv)

    return deconv



##################################
