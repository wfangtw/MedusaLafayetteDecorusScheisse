#########################################################
#   FileName:	    [ macros.py ]			#
#   PackageName:    []					#
#   Sypnosis:	    [ Define macros ]			#
#   Author:	    [ MedusaLafayetteDecorusSchiesse]   #
#########################################################

#macro definitions
INPUT_DIM = 39
NEURONS_PER_LAYER = 128
OUTPUT_DIM = 48
BATCH_SIZE = 128
LEARNING_RATE = 0.0001
LEARNING_RATE_DECAY = 0.9999
MOMENTUM = 0.9
MAX_EPOCH = 20
TRAIN_SIZE = 1024
TRAIN_TIMES = TRAIN_SIZE*MAX_EPOCH/BATCH_SIZE