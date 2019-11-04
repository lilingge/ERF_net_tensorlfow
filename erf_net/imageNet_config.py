import numpy as np

from easydict import EasyDict as edict

def imageNet_config():
  mc = edict()

  mc.IMAGE_WIDTH           = 224
  mc.IMAGE_HEIGHT          = 224
  mc.BATCH_SIZE            = 64
  mc.KEEP_PROB             = 0.5
  mc.WEIGHT_DECAY          = 0.0005
  mc.LEARNING_RATE         = 0.01
  mc.DECAY_STEPS           = 10000

  mc.MOMENTUM              = 0.9
  mc.LR_DECAY_FACTOR       = 0.5

  return mc
