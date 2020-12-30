# question-answering-triton

##1. SQUAD-type QA Inferencer based on huggingface transformers

##2. Pytorch to TorchScript to TensorRT
* why not ONNX?
### [torch-script](https://pytorch.org/docs/stable/jit.html)

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
```
* [dynamic batcher](https://github.com/triton-inference-server/server/blob/r20.12/docs/model_configuration.md#dynamic-batcher)