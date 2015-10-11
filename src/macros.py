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
BATCH_SIZE = 12
LEARNING_RATE = 0.0001
LEARNING_RATE_DECAY = 0.9999
MAX_EPOCH = 5
TRAIN_SIZE = 1024
TRAIN_TIMES = TRAIN_SIZE*MAX_EPOCH/BATCH_SIZE
