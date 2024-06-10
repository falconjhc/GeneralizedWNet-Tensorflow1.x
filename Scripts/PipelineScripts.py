# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import

import os
os.system('cls' if os.name == 'nt' else 'clear')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow.compat.v1 as tf # type: ignore
tf.disable_v2_behavior()



import argparse
import sys
sys.path.append('../')
os.chdir(sys.path[0])
 
from Pipelines.Trainer import Trainer
from Configurations.ConfigurationOrganization import ParameterSetting  # noqa: E402
eps = 1e-9

import importlib
from easydict import EasyDict

# OPTIONS SPECIFICATION
# resumeTrain   =   0: training from stratch
#                   1: training from a based model
parser = argparse.ArgumentParser(description='Train')
parser.add_argument('--config', dest='config',type=str,required=True)
parser.add_argument('--resumeTrain', dest='resumeTrain', type=int,required=True)
parser.add_argument('--initLr', dest='initLr',type=float,required=True)
parser.add_argument('--batchSize', dest='batchSize', type=int,required=True)
parser.add_argument('--epochs', dest='epochs', type=int, required=True)
parser.add_argument('--skipTest', dest='skipTest', type=bool, default=False)
parser.add_argument('--encoder', dest='encoder', type=str, required=True)
parser.add_argument('--mixer', dest='mixer', type=str, required=True)
parser.add_argument('--decoder', dest='decoder', type=str, required=False, default=None)



def main(_):
    args = parser.parse_args()
    hyperParams = \
    ParameterSetting(EasyDict(importlib.import_module('.'+args.config, 
                                                      package='Configurations').hyperParams), args).config
    penalties = EasyDict(importlib.import_module('.'+args.config, 
                                                 package='Configurations').penalties)

    print("#####################################################")
    model = Trainer(hyperParams=hyperParams, penalties=penalties)
    model.Pipelines()

if __name__ == '__main__':
    tf.app.run()
