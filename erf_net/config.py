from easydict import EasyDict as edict
import numpy as np

cfg = edict()

cfg.MAX_STEP = 100000

cfg.IMAGE_HEIGHT = 1024
cfg.IMAGE_WIDTH = 2048
cfg.IMAGE_DEPTH = 3


cfg.BATCH_SIZE = 1
cfg.VAL_BATCH_SIZE = cfg.BATCH_SIZE
cfg.NUM_CLASSES = 19
cfg.LABEL_WEIGHTED = True
cfg.CLASS_WEIGHT = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]#,1,1,1,1,1,1,1,1,1,1,1]

cfg.INIT_LEARNING_RATE = 0.0005
cfg.LR_DECAY_STEP = 30000
cfg.LR_DECAY_FACTOR = 0.5
cfg.SUBTRACT_CHANNEL_MEAN = False
cfg.WEIGHT_DECAY = 2e-4
cfg.USE_BN = True
cfg.USE_DROPOUT = True
cfg.DROP_RATE = 0.5

cfg.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 500

cfg.TRAIN_QUEUE_CAPACITY = 100
cfg.EVAL_QUEUE_CAPACITY = 64

cfg.reload = True
