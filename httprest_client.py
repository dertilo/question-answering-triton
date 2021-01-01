import uuid

from datetime import datetime

import hashlib

from pprint import pprint
from tritonclient.utils import np_to_triton_dtype

from typing import Dict, List
import tritonclient.http as httpclient
import numpy as np

# TODO(tilo): this is very dirty hack!
INPUT_NAMES = ["input_ids", "token_type_ids", "attention_mask"]
OUTPUT_NAMES = ["start_logits", "end_logits"]


class HttpTritonClient:
    def __init__(self, model_name, url="localhost:8000") -> None:
        super().__init__()
        self.model_name = model_name
        self.url = url
        self.client = None

    def __enter__(self):
        self.client = httpclient.InferenceServerClient(self.url)
        self.metadata = self.client.get_model_metadata(model_name=self.model_name)
        self.model_config = self.client.get_model_config(model_name=self.model_name)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.client.close()

    def __call__(self, inputs: Dict[str, np.ndarray]):
        self._check_input(inputs)

        def build_infer_input(name:str, array:np.ndarray):
            ii = httpclient.InferInput(
                name, list(array.shape), np_to_triton_dtype(array.dtype)
            )
            ii.set_data_from_numpy(array, binary_data=True)
            return ii

        infer_inputs = [
            build_infer_input(cfg["name"],inputs[input_name])
            for input_name, cfg in zip(INPUT_NAMES, self.model_config["input"])
        ]

        outputs = [
            httpclient.InferRequestedOutput(cfg["name"])
            for cfg in self.model_config["output"]
        ]
        request_id = str(uuid.uuid4()) # TODO(tilo): really collision-free ?
        response = self.client.infer(
            self.model_name, infer_inputs, request_id=request_id, outputs=outputs
        )
        # result = response.get_response()
        return {k:response.as_numpy(k) for k in OUTPUT_NAMES}

    def _check_input(self, inputs):
        input_shapes_are_valid = all(
            [
                list(inputs[input_name].shape) == cfg["dims"]
                for input_name, cfg in zip(INPUT_NAMES, self.model_config["input"])
            ]
        )
        assert input_shapes_are_valid
        types_are_valid = all(
            [
                np_to_triton_dtype(inputs[input_name].dtype) == cfg[
                    "data_type"].replace("TYPE_", "")
                for input_name, cfg in zip(INPUT_NAMES, self.model_config["input"])
            ]
        )
        assert types_are_valid


if __name__ == "__main__":
    # model_name = "densenet_onnx"
    model_name = "deepset_bert_base_cased_squad2"
    with HttpTritonClient(model_name) as client:
        client
