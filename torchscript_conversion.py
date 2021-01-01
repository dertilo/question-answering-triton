# stolen from: https://blog.einstein.ai/benchmarking-tensorrt-inference-server/
# see: https://gist.githubusercontent.com/keskarnitish/1061cbd101ab186e2d80c7877517e7ee/raw/887a8a64ea6e77787bb0b4fbf2db542b282d5c07/saved_pytorch_model.py
import os

import json

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

    def __to_cuda(self, x): #TODO(tilo): why is this necessary?
        return x.cuda() if self.use_gpu else x

    def forward(self, *inputs):
        out = self.model(
            **{k: self.__to_cuda(v) for k, v in zip(self.arg_names, inputs)}
        )
        return tuple(
            out[k] for k in self.out_names
        )  # WTF! why does torchscript-trace only accept tuples!?


DATA_TYPES = {"torch.FloatTensor": "TYPE_FP32", "torch.LongTensor": "TYPE_INT64"}


def build_variables(variables: List[torch.Tensor], is_input=True):
    in_out = "input" if is_input else "output"
    return [
        {
            "name": f"'{in_out}__{k}'",
            "data_type": DATA_TYPES[x.type()],
            "dims": list(x.size()),
        }
        for k, x in enumerate(variables)
    ]


def remove_double_quotes(s):
    return s.replace('"', "").replace("'", '"')


def generate_config_pbtxt(inputs, outputs, dir: str, platform="pytorch_libtorch"):
    # to stuff meta-data in config.pbtxt see https://github.com/triton-inference-server/server/issues/439
    model_name = dir.split("/")[-1]
    inputs_s = build_variables(inputs, is_input=True)
    outputs_s = build_variables(outputs, is_input=False)

    text = f"""name: '{model_name}'
    platform: '{platform}'
    input {json.dumps(inputs_s, indent=4)}
    output {json.dumps(outputs_s, indent=4)}
        """

    with open(f"{dir}/config.pbtxt", "w") as f:
        f.write(remove_double_quotes(text))


def repack(tuples: List[Tuple]) -> List[List]:
    return [list(x) for x in zip(*tuples)]


if __name__ == "__main__":

    model_name = "deepset/bert-base-cased-squad2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)

    text, questions = get_sample_context_questions()

    inputs = tokenizer(questions[0], text, add_special_tokens=True, return_tensors="pt")
    output = model(**inputs)

    input_names, input_vars = repack(inputs.items())
    output_names, output_vars = repack(output.items())

    model_repo = f"/home/tilo/code/ML/triton-server/docs/examples/model_repository"
    model_folder = model_name.replace("/", "_").replace("-", "_")
    model_dir = f"{model_repo}/{model_folder}"
    os.makedirs(model_dir, exist_ok=True)
    generate_config_pbtxt(input_vars, output_vars, model_dir)
    model_save_dir = f"{model_dir}/1"
    os.makedirs(model_save_dir, exist_ok=True)

    pt_model = WrappedModel(model, input_names, output_names).eval()
    traced_script_module = torch.jit.trace(
        pt_model, tuple(input_vars)
    )  # WTF! why does torchscript-trace only accept tuples!?
    traced_script_module.save(f"{model_save_dir}/model.pt")
