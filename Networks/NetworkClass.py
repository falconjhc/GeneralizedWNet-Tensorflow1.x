# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function
from easydict import EasyDict


class NetworkBase(object):
    def __init__(self):
        self.inputs=EasyDict({})
        self.outputs=EasyDict({})
        self.trainVars=EasyDict({})
        self.savers=EasyDict({})
        self.groundtruths=EasyDict({})
        
class EncoderBase(NetworkBase):
    def __init__(self):
        super().__init__()
        
    def InitOutputLists(self):
        self.outputs.update({'fullFeatureList': list()})
        self.outputs.update({'encodedFinalOutputList': list()})
        self.outputs.update({'category': list()})


    def ReorganizeOutputList(self, repeats):

        def _reorganize(inputList):
            _modNum = len(inputList)//repeats
            if repeats==1:
                return inputList
            fullList = list()
            for ii in range(len(inputList)):
                if ii % _modNum ==0:
                    _thisList = list()
                _thisList.append(inputList[ii])
                if (ii+1)%_modNum==0:
                    fullList.append(_thisList)
            return fullList
        
        # self.outputs.shortcutOutputList =_reorganize(self.outputs.shortcutOutputList)
        # self.outputs.residualOutputList =_reorganize(self.outputs.residualOutputList)
        self.outputs.fullFeatureList =_reorganize(self.outputs.fullFeatureList)
        self.outputs.encodedFinalOutputList =_reorganize(self.outputs.encodedFinalOutputList)
        self.outputs.category = _reorganize(self.outputs.category)
        
        return self
        
class NetworkIO(object):
    def __init__(self):
        self.train = None
        self.testOnTrain = None
        self.testOnValidate = None