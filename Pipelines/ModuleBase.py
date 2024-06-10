# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function
from easydict import EasyDict


class EvaluationIO(object):
    def __init__(self):
        self.train= EasyDict({})
        self.testOnTrain=EasyDict({})
        self.testOnValidate=EasyDict({})




class ModuleBase(object):
    def __init__(self):
        self.inputs=EasyDict({})
        self.outputs=EasyDict({})
        self.groundtruths=EasyDict({})

class ModuelIO(object):
    def __init__(self):
       self.train= ModuleBase()
       self.testOnTrain=ModuleBase()
       self.testOnValidate=ModuleBase()
       
class ModuleBlock(object):
    def __init__(self):
        self.savers=None
        self.trainVars=None
        self.path=None
        self.io=ModuelIO()