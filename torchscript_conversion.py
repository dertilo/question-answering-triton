# stolen from: https://blog.einstein.ai/benchmarking-tensorrt-inference-server/
# see: https://gist.githubusercontent.com/keskarnitish/1061cbd101ab186e2d80c7877517e7ee/raw/887a8a64ea6e77787bb0b4fbf2db542b282d5c07/saved_pytorch_model.py
from typing import List, Tuple

import torch
from transformers import *

from huggingface_simple_qa import get_sample_context_questions


class WrappedModel(torch.nn.Module):
    def __init__(
        self, model, arg_names: List[str], out_names: List[str], use_gpu=False
    ):
        super().__init__()
        self.arg_names = arg_names
        self.out_names = out_names
        self.use_gpu = use_gpu
        self.model = self.__to_cuda(model)

    def __to_cuda(self, x):
        return x.cuda() if self.use_gpu else x

    def forward(self, *inputs):
        out = self.model(
            **{k: self.__to_cuda(v) for k, v in zip(self.arg_names, inputs)}
        )
        return tuple(
            out[k] for k in self.out_names
        )  # WTF! why does torchscript-trace only accept tuples!?


if __name__ == "__main__":

    model_name = "deepset/bert-base-cased-squad2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)

    text, questions = get_sample_context_questions()

    inputs = tokenizer(questions[0], text, add_special_tokens=True, return_tensors="pt")
    output = model(**inputs)
    out_names = output.keys()
    named_args = list(inputs.items())
    in_names = [n for n, _ in named_args]
    pt_model = WrappedModel(model, in_names, out_names).eval()
    traced_script_module = torch.jit.trace(
        pt_model, tuple(t for _, t in named_args)
    )  # WTF! why does torchscript-trace only accept tuples!?
    traced_script_module.save("model.pt")
