# question-answering-triton
### Questions Answering
* SQUAD-like extractive question answering should be referred to as __reading comprehension__!
* "real" QA is __open__! no context given, but to be retrieved -> [deepset-haystack](https://github.com/deepset-ai/haystack)
* google's QA-dataset [natural-question-answering](https://www.kaggle.com/c/tensorflow2-question-answering/overview) might be less boring than SQUAD
    * there's been a kaggle-challenge; a successful participants git-repo can be found [here](https://github.com/see--/natural-question-answering)


### deploying models to [Triton](https://developer.nvidia.com/nvidia-triton-inference-server)
* triton supports multiple model-formats (implements corresponding [backends](https://github.com/triton-inference-server/backend/blob/main/README.md#backends)) like: TensorFlow, TensorRT, PyTorch, ONNX, ...
* models can be converted/compiled to more compute-optimized formats (trade-off between flexibility and performance) in order to make use of "faster" [backends](https://github.com/triton-inference-server/backend/blob/main/README.md#backends)
  
1. clean + pytorchic way:
    1. use [torch-script](https://pytorch.org/docs/stable/jit.html) to convert pytorch-model 
    2. use [TRTorch](https://github.com/NVIDIA/TRTorch) to "compile" torchscript-module to specific GPU-hardware
2. huggingface's way is to use [ONNX](https://huggingface.co/transformers/serialization.html); they even provide a [conversion-script](https://github.com/huggingface/transformers/blob/master/src/transformers/convert_graph_to_onnx.py)

### quick start
0. create environment + install requirements: `pip install -r requirements.txt`
1. convert pretrained model to ONNX (substitue `MODEL_NAME` with any model found on https://huggingface.co/models)
```shell
FULL_PATH_TO_MODEL_REPO=some_path
MODEL_NAME='deepset/bert-base-cased-squad2'
MODEL_FOLDER=$(echo $MODEL_NAME | tr '/' '-')
python convert_graph_to_onnx.py --framework pt --pipeline question-answering --model "$MODEL_NAME" $FULL_PATH_TO_MODEL_REPO/model_repository_new/$MODEL_FOLDER/1/model.onnx
```
2. setup triton-server
```shell
MODEL_PATH=$FULL_PATH_TO_MODEL_REPO/model_repository
docker run --rm -p8000:8000 -p8001:8001 -p8002:8002 -v $MODEL_PATH:/models nvcr.io/nvidia/tritonserver:20.12-py3 tritonserver --model-repository=/models # change version "20.12" to whichever wanted
```
* if successful triton prints something like this 
```shell
+-----------------------------------------+---------+--------+
| Model                                   | Version | Status |
+-----------------------------------------+---------+--------+
| bert-base-cased-squad2                  | 1       | READY  |
```
3. run exemplary python script: [huggingface_simple_qa.py](huggingface_simple_qa.py)
```shell
python huggingface_simple_qa.py
# should output the following
Question: How many pretrained models are available in 🤗 Transformers?
Answer: over 32 +
Question: What does 🤗 Transformers provide?
Answer: general - purpose architectures
Question: 🤗 Transformers provides interoperability between which frameworks?
Answer: TensorFlow 2. 0 and PyTorch

```

# more details notes (no need to read)
##1. SQUAD-type QA Inferencer based on huggingface transformers
* pretrained [bert-base-cased-squad2](https://huggingface.co/deepset/bert-base-cased-squad2) by huggingface + deepset
##2. Converting to some optimized format
* ONNX vs. TorchScript+TRTorch? 
* [torch2trt](https://github.com/NVIDIA-AI-IOT/torch2trt) directly converts pytorch to TensorRT

### [ONNX](https://huggingface.co/transformers/serialization.html)
```shell
python convert_graph_to_onnx.py --framework pt --pipeline question-answering --model "deepset/bert-base-cased-squad2" ~/code/ML/triton-server/docs/examples/model_repository_new/bert-base-cased-squad2/1/model.onnx
```
* convert ONNX to TensorRT [see](https://github.com/onnx/onnx-tensorrt)

### [torch-script](https://pytorch.org/docs/stable/jit.html)
* `torch.jit.trace(model, some_example_input)` neglects any logic within the `model`
* `torch.jit.script(model)` to capture logic
### [TRTorch](https://github.com/NVIDIA/TRTorch)
* `TRTorch is a compiler that uses TensorRT (NVIDIA's Deep Learning Optimization SDK and Runtime) to optimize TorchScript code`
* `From a TRTorch prespective, there is better support (i.e your module is more likely to compile) for traced modules because it doesn’t include all the complexities of a complete programming language, though both paths supported. `
* TRTorch has 3 main interfaces for using the compiler.
1. CLI: [trtorchc](https://nvidia.github.io/TRTorch/tutorials/trtorchc.html)
    * `supports almost all features of the compiler from the command line`
    * `compiler can output two formats, either a TorchScript program with the TensorRT engine embedded or the TensorRT engine itself as a PLAN file.`    
2. [Compiling with TRTorch in Python](https://nvidia.github.io/TRTorch/py_api/trtorch.html)
   * returns: `Compiled TorchScript Module, when run it will execute via TensorRT`
3. [TRTorch Directly From PyTorch](https://nvidia.github.io/TRTorch/tutorials/use_from_pytorch.html)
    * more low level than `2.` ?
        

##3. Triton Serving
* compared to TorchServe ? 
* [quick-start](https://github.com/triton-inference-server/server/blob/r20.12/docs/quickstart.md)
    * [docker-images](https://ngc.nvidia.com/catalog/containers/nvidia:tritonserver)
        * `can also work on CPU-only systems. In both cases you can use the same Triton Docker image`
        * client ? 
```shell
docker pull nvcr.io/nvidia/tritonserver:20.12-py3
MODEL_PATH=$HOME/code/ML/triton-server/docs/examples/model_repository
docker run --rm -p8000:8000 -p8001:8001 -p8002:8002 -v $MODEL_PATH:/models nvcr.io/nvidia/tritonserver:20.12-py3 tritonserver --model-repository=/models
# Verify Triton Is Running Correctly
curl -v localhost:8000/v2/health/ready
# Client
docker run -it --rm --net=host nvcr.io/nvidia/tritonserver:20.12-py3-sdk
./install/bin/image_client -m densenet_onnx -c 3 -s INCEPTION /workspace/images/mug.jpg
```
### Questions / TODO
* `HTTP/REST` vs `GRPC` ? 
* `GRPC generated library (generated stub)` vs `client library`
* why/when would one want batching in the client?
    * [salesforce](https://blog.einstein.ai/benchmarking-tensorrt-inference-server/) says: 
        `Static batching, i.e., client-side batching, does not significantly improve throughput or latency when dynamic batching is employed. This obviates the need for explicit batching outside of the inference server` -> no need for client-side batching!
    * see [dynamic batcher](https://github.com/triton-inference-server/server/blob/r20.12/docs/model_configuration.md#dynamic-batcher)
* why/when async / streaming in client?