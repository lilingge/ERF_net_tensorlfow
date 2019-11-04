# Author Lingge Li from XJTU(446049454@qq.com)
# train fuctions

import tensorflow as tf

def get_train_op(cfg, total_loss, global_step):
    init_lr = cfg.INIT_LEARNING_RATE
    
    # learning rate exponential decay
    lr = tf.train.exponential_decay(init_lr, global_step, cfg.LR_DECAY_STEP,
                                    cfg.LR_DECAY_FACTOR, staircase=True)

    # compute gradients and crop gradients
    optimizer = tf.train.AdamOptimizer(lr)
    gradients = optimizer.compute_gradients(total_loss)
    crop_gradients = optimizer.apply_gradients(gradients, global_step=global_step)

    # add histogram summaries for gradients
    for grad, var in gradients:
        if grad is not None:
            tf.summary.histogram(var.op.name + '/gradients', grad)
    
    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(0.99, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    # first run crop_gradients, variables_averages_op, then do nothing
    with tf.control_dependencies([crop_gradients, variables_averages_op]):
        train_op = tf.no_op(name='train')

    return train_op

def add_summaries(var_list, mode, cfg=None):
    summaries_op = []
    for var in var_list:
        if mode == 'scala':
            summaries_op.append(tf.summary.scalar(var.op.name, var))
        elif mode == 'hist':
            summaries_op.append(tf.summary.histogram(var.op.name, var))
        elif mode == 'image':
            max_show = cfg.BATCH_SIZE
            summaries_op.append(tf.summary.image(var.op.name, var, max_show))
    
    return summaries_op





