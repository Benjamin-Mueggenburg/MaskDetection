from general import device
'''Filter base class. In a pipeline and filter architecture each filter will be a model/module '''
class Module():
    def __init__(self):
        self.model = None
        self.keep_output_in_memory = False

    def init(self, device=device):
        self.device = device
        
        if self.model == None:
            self.model = self.initialise_weights()

    def initialise_weights(self):
        '''Return model with weights 
        @Override this'''
        return None
    
    def preprocess(self, inputs):
        '''Given a list of inputs (images) preprocess for inference
            @Override'''
        return inputs

    def postprocess(self, outputs):
        '''Give list of outputs arr with shape [batch_size, num_boxes, x, y, w, h] postprocess'''
        return outputs

    def inference(self, inputs):
        outputs = inputs
        return outputs


    