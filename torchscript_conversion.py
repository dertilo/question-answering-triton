# stolen from: https://blog.einstein.ai/benchmarking-tensorrt-inference-server/
# see: https://gist.githubusercontent.com/keskarnitish/1061cbd101ab186e2d80c7877517e7ee/raw/887a8a64ea6e77787bb0b4fbf2db542b282d5c07/saved_pytorch_model.py

import torch
from transformers import *


class WrappedModel(torch.nn.Module):
    def __init__(self):
        super(WrappedModel, self).__init__()
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased').cuda()
    def forward(self, data):
        return self.model(data.cuda())

example = torch.zeros((4,128), dtype=torch.long) # bsz , seqlen
pt_model = WrappedModel().eval()
traced_script_module = torch.jit.trace(pt_model, example)
traced_script_module.save("model.pt")