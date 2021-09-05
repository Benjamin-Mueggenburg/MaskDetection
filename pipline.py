from module import Module
import numpy as np
import torch

class Pipeline():
    def __init__(self, modules: list[Module], use_tensorRt=False, device=torch.device('cuda')):
        self.using_tensorRt = use_tensorRt
        self.modules = modules if isinstance(modules, list) else [modules]
        self.kept_intermediate_outputs = []
        self.device = device

    def init(self):
        print(f"Initing modules with Torch {self.device.type} as device")
        for module in self.modules:
            module.init(self.device))

    def run(self, inputs):
        '''Image input'''
        #original_input = np.copy(inputs)
        for module in self.modules:
            inputs = module.preprocess(inputs)
            outputs = module.inference(inputs)
            outputs = module.postprocess(outputs)

            if module.keep_output_in_memory:
                self.kept_intermediate_outputs = outputs

            inputs = outputs #For the next iteration with the next module

        return outputs
           
        

def build_pipeline(modules, use_tensorRt=False):
    return Pipeline(modules, use_tensorRt=use_tensorRt)