#########################################################
#   FileName:	    [ activation.py ]			#
#   PackageName:    [ DNN]					#
#   Synopsis:	    [ Define activation functions ]		#
#   Author:	    [ MedusaLafayetteDecorusSchiesse]   #
#########################################################

def relu(x):
        return T.switch(x < 0, 0.01*x, x)

def softmax(vec):
        vec = T.exp(vec)
        return vec / vec.sum(axis=0)
