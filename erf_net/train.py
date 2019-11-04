# Author Lingge Li from XJTU(446049454@qq.com)
# train 

import tensorflow as tf
import os
import sys
import time
import numpy as np

from data_load import TFLoadingPipeline
from erf_net import ERF_Net
from erf_net_slim import *
from train_slim import *
from IOU_metrics import *
from config import cfg


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('model', 'erf', """ which model to use """)
tf.app.flags.DEFINE_string('ckpt_path', './log', """ dir to store ckpt """)
tf.app.flags.DEFINE_string('summary_path', './summary', """ dir to store summary """)
tf.app.flags.DEFINE_string('images_path', '', """ path to train images """)
tf.app.flags.DEFINE_string('labels_path', '', """ path to label images """)
tf.app.flags.DEFINE_string('val_images_path', '', """ path to validate train images """)
tf.app.flags.DEFINE_string('val_labels_path', '', """ path to validate label images """)

def checkArgs():
    print('Training model: {}'.format(FLAGS.model))
    print("Train images path: {}".format(FLAGS.images_path))
    print("Label images path: {}".format(FLAGS.labels_path))
    print("Validation train path: {}".format(FLAGS.val_images_path))
    print("Validation label path: {}".format(FLAGS.val_labels_path))
    print("Model checkpoint path: {}".format(FLAGS.ckpt_path))
    print("Model summary path {}".format(FLAGS.summary_path))

def main(self):
    
    checkArgs()
    ckpt_path = FLAGS.ckpt_path
    if not os.path.isdir(ckpt_path):
        os.mkdir(ckpt_path)
    
    if FLAGS.model == 'erf':
        model = ERF_Net()
    else:
        print('please input a model!')
        sys.exit(-1)
    net_var = tf.global_variables()
    for idx,var in enumerate(net_var):
	    print(idx,var.name)
    '''
    restore_variables = []
    for net in net_var:
        try:
            if 'Exponential' in net.name or 'Adam' in net.name:
                continue
            elif 'nonbt1d' in net.name:
                if net.name[2]=='_':
                    restore_variables.append(net)
                elif float(net.name[1:3])<=16:
                    restore_variables.append(net)
        except:
            continue
    for idx,var in enumerate(restore_variables):
        print(idx,var.name)
    print('restore length:',len(restore_variables))
    loader = tf.train.Saver(var_list= restore_variables)
    '''

    data_pipeline = TFLoadingPipeline()
    data_pipeline.setup(FLAGS.images_path, FLAGS.labels_path, cfg.BATCH_SIZE, cfg.TRAIN_QUEUE_CAPACITY)
    data_pipeline.setup_validate(FLAGS.val_images_path, FLAGS.val_labels_path, cfg.VAL_BATCH_SIZE, cfg.EVAL_QUEUE_CAPACITY)

    global_step = tf.Variable(0, trainable=False)
    
    # Compute gradients and track the moving averages of all trainable variables.
    train_op = get_train_op(cfg, model.total_loss, global_step)
    
    saver = tf.train.Saver(max_to_keep=10)

    # add summaries
    print(type(model.trainables()))
    add_summaries([model.input_image], 'image', cfg)
    add_summaries(model.trainables(), 'hist')
    #add_summaries(model.bn_variables(), 'hist')
    #add_summaries(model.bn_mean_variance(), 'hist')
    add_summaries(model.losses, 'scala')
    
    merged = tf.summary.merge_all()

    # add evaluation metric summaries


    with tf.Session() as sess:
        total_start_time = time.time()

        summary_writer = tf.summary.FileWriter(FLAGS.summary_path, sess.graph)

        # initialize
        print('initializing model params...')
        sess.run(tf.global_variables_initializer())

        if cfg.reload:
            print('reloading the pretrained parameters...')
            ckpt = tf.train.get_checkpoint_state(ckpt_path)
            if ckpt and ckpt.model_checkpoint_path:
                #loader.restore(sess, ckpt.model_checkpoint_path)
                # step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
                step = 1
            else:
                step = 1
        else:
            tf.gfile.DeleteRecursively(FLAGS.ckpt_path)
            tf.gfile.MakeDirs(FLAGS.ckpt_path)
            step = 1

        data_pipeline.start(sess)
        

        for step in range(cfg.MAX_STEP):
            start_time = time.time()
            image_batch, label_batch = data_pipeline.load_batch()
            #print('label_batch: ', type(label_batch), label_batch.shape, label_batch[0].flatten().shape)
            
            feed_dict = {
                model.input_image: image_batch,
                model.is_training: True,
                model.label_image: label_batch
            }
            model.is_train = True

            # run training
            _, pred, loss, step, summay_str = sess.run([train_op, model.segment_output, model.total_loss, global_step, merged],
                                                       feed_dict=feed_dict)
            #print('predictions: ', type(pred), pred[0].shape, pred[0].argmax(2), pred.argmax(2).shape)

            # every 10 step, output metrics
            if step % 10 == 0:
                duration = time.time() - start_time
                print('step {}: {} sec elapsed, loss = {}'.format(step, duration, loss))
                mean_iu, iu = per_class_iu(pred, label_batch)
                print('mean IU = {}'.format(mean_iu))
                for i in range(cfg.NUM_CLASSES):
                    print('class #{} iu = {}'.format(i, iu[i]))
                
            # every 100 step, do validation & write summary
            if step % 100 == 0:
                # write summary
                # summary_str = sess.run(merged, feed_dict=feed_dict)
                summary_writer.add_summary(summay_str, step)

                print(' start validating...... ')
                VAL_STEP = int(cfg.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL / cfg.VAL_BATCH_SIZE)
                print(VAL_STEP, type(VAL_STEP))
                total_val_loss = 0.0
                class_hist = np.zeros((cfg.NUM_CLASSES, cfg.NUM_CLASSES))
                for val_step in range(VAL_STEP):
                    val_image_batch, val_label_batch = data_pipeline.load_validate_batch()
                    model.is_train = False
                    val_feed_dict = {
                        model.input_image: val_image_batch,
                        model.is_training: False,
                        model.label_image: val_label_batch
                    }

                    val_loss, val_pred = sess.run([model.total_loss, model.segment_output], feed_dict=feed_dict)
                    total_val_loss = total_val_loss + val_loss
                    class_hist = class_hist + get_class_hist(val_pred, val_label_batch)
                
                avg_loss = total_val_loss / VAL_STEP
                print('val loss: {}'.format(avg_loss))
                mean_iu, iu = per_class_iu_from_hist(class_hist)
                print('val mean iu = {}'.format(mean_iu))
                for i in range(cfg.NUM_CLASSES):
                    print('val class #{} iu = {}'.format(i, iu[i]))
                
                print(' end validating.... ')
            
            # every 1000 steps, save the model checkpoint
            if step % 1000 == 0 or (step + 1) == cfg.MAX_STEP:
                checkpoint_path = os.path.join(ckpt_path, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step = step)

        # training over
        print('training complete')
        data_pipeline.shutdown()
        total_end_time = time.time()
        print('total time elapsed: {} h'.format((total_end_time - total_start_time)/3600.0))   

if __name__ == '__main__':
    tf.app.run()
