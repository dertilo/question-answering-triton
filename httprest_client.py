import uuid
from tritonclient.utils import np_to_triton_dtype

from typing import Dict, List
import tritonclient.http as httpclient
import numpy as np


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

        def build_infer_input(name: str, array: np.ndarray):
            ii = httpclient.InferInput(
                name, list(array.shape), np_to_triton_dtype(array.dtype)
            )
            ii.set_data_from_numpy(array, binary_data=True)
            return ii

        infer_inputs = [
            build_infer_input(cfg["name"], inputs[cfg["name"]])
            for cfg in self.model_config["input"]
        ]

        outputs = [
            httpclient.InferRequestedOutput(cfg["name"])
            for cfg in self.model_config["output"]
        ]
        request_id = str(uuid.uuid4())  # TODO(tilo): really collision-free ?
        response = self.client.infer(
            self.model_name, infer_inputs, request_id=request_id, outputs=outputs
        )
        # result = response.get_response()
        outputs_dict = {
            cfg["name"]: response.as_numpy(cfg["name"])
            for cfg in self.model_config["output"]
        }
        return outputs_dict

    def _check_input(self, inputs):
        input_shapes_are_valid = all(
            [
                (list(inputs[cfg["name"]].shape) == cfg["dims"]) or cfg["dims"] == [-1]
                for cfg in self.model_config["input"]
            ]
        )
        assert input_shapes_are_valid
        types_are_valid = all(
            [
                np_to_triton_dtype(inputs[cfg["name"]].dtype)
                == cfg["data_type"].replace("TYPE_", "")
                for cfg in self.model_config["input"]
            ]
        )
        assert types_are_valid
