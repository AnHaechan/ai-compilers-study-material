## AI compilers study materials 

### Description
This repository is designed as a focused resource for individuals beginning work in the field of AI compilers and systems.

​By AI compilers, I specifically refer to the software stack responsible for translating and optimizing DNN models into efficient assembly code for various accelerators, including GPUs and NPUs.

​I have curated the most valuable resources I've encountered to date to provide a solid starting point.

### Prerequisites
- Basic understanding of DNN and Transformer architectures.
- Some experience on parallel programming and GPUs (e.g., implementing matmul on CUDA).

### Introductions
- [A friendly introduction to machine learning compilers and optimizers](https://huyenchip.com/2021/09/07/a-friendly-introduction-to-machine-learning-compilers-and-optimizers.html)
- [Democratizing AI compute: Part 6](https://www.modular.com/blog/democratizing-ai-compute-part-6-what-about-ai-compilers)
  - Strongly recommend to read the full series.
- [Introduction to AI compilers](https://docs.google.com/presentation/d/1RZdV3Z-Q1NEbpU1-qk9C97yE1QvwNLJ9Gc7JLaFLZCw/edit?slide=id.p#slide=id.p)

### State of AI compilers
- In the pre-LLM era, AI compilers consist of **graph compilers + kernel libraries / compilers**. 
  - Graph compilers perform various computational graph-level optimizations.
  - Kernels for individual operators can be crafted by direct programming (*kernel libraries*) or automatic generation (*kernel compilers*).
  - [TVM](https://github.com/apache/tvm) and [XLA](https://github.com/openxla/xla) were the most popular compiler frameworks.
- After the domination of LLMs, the focus has shifted to **LLM runtimes + attention kernels**.
  - LLMs, or autoregressive Transformers with self-attention, has become the core workload.
  - Hence, it is crucial to optimize its performance to the last bit via runtime strategies and custom kernels.
  - [vLLM](https://github.com/vllm-project/vllm) runtime with kernels written in template-based languages, such as [CUTLASS](https://docs.nvidia.com/cutlass/index.html) or [Triton](https://github.com/triton-lang/triton), is a popular choice.
  - [FlashInfer](https://github.com/flashinfer-ai/flashinfer) and [FlexAttention](https://pytorch.org/blog/flexattention/) even provides template programming specifically for attentions. They generate kernels aware of KV cache access patterns and hardware features.
- The details have been changing, but AI compilers remain to be an important topic. 
  - Graph compilers can accelerate the rest of the model[[example](https://blog.vllm.ai/2025/08/20/torch-compile.html)], used together with custom attention kernels.
  - Kernel programming, especially for recent GPU architectures, remains to be complex and ad-hoc[[paper](https://arxiv.org/html/2504.07004v1)].
  - AI compilers can support faster prototyping for different attention algorithms, or even non-Transformer models [[mamba](https://arxiv.org/pdf/2312.00752)].

### Papers
- [List of papers](https://github.com/merrymercy/awesome-tensor-compilers)
- There is [a survey paper](https://arxiv.org/abs/2002.03794), but keep in mind that it is a bit outdated (2020).
- I'd recommend skimming through recent ML/DNN/AI compiler papers on ASPLOS/MLSys/PLDI/ISCA, such as [Relax](https://arxiv.org/pdf/2311.02103), [Cypress](https://arxiv.org/pdf/2504.07004), [Exo2](https://arxiv.org/pdf/2411.07211).
- FlashAttention series and PagedAttention are also recommended to understand the gist of kernel optimization for LLMs[[1](https://arxiv.org/pdf/2205.14135)][[2](https://arxiv.org/pdf/2307.08691)][[3](https://arxiv.org/pdf/2407.08608)][[decoding](https://arxiv.org/pdf/2311.01282)][[paged](https://arxiv.org/pdf/2309.06180)].

### Frameworks
#### Graph compilers + Kernel libraries
- NVIDIA [TensorRT](https://github.com/NVIDIA/TensorRT) + [cuDNN](https://developer.nvidia.com/cudnn)
  - Core logics are closed-sourced.
- Intel [OpenVino](https://github.com/openvinotoolkit/openvino) + [oneDNN](https://github.com/uxlfoundation/oneDNN)

#### Graph & Kernel compilers
- [TorchInductor](https://github.com/pytorch/pytorch/tree/main/torch/_inductor)
- [TVM](https://github.com/apache/tvm)
- [XLA](https://github.com/openxla/xla)

#### LLM runtimes
- [vLLM](https://github.com/vllm-project/vllm)
- NVIDIA [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)

#### Template-based kernel languages
- NVIDIA [CUTLASS](https://docs.nvidia.com/cutlass/index.html)
- [Triton](https://github.com/triton-lang/triton)
  - [Gluon](https://github.com/triton-lang/triton/blob/main/python/tutorials/gluon/01-intro.py)
    - For Hopper/Blackwell GPUs, Triton struggles to achieve optimal performance due to the increased complexity of GPU.
    - To mitigate this, Gluon is being developed within the Triton ecosystem, which exposes more lower-level controls akin to CUTLASS.
  - [Helion](https://pytorch.org/blog/helion/)
    - Helion provides the middle ground between TorchInductor and Triton, offering flexbilty to program kernels with templates, but hiding details of Triton
 
#### Attention template languages
- [FlashInfer](https://github.com/flashinfer-ai/flashinfer)
- [FlexAttention](https://pytorch.org/blog/flexattention/)

### Hands-on tutorials
- TVM
  - https://mlc.ai/
  - https://tvm.d2l.ai/
- Triton
  - https://github.com/srush/Triton-Puzzles
  - https://triton-lang.org/main/getting-started/tutorials/index.html
- TorchInductor
  - https://github.com/meta-pytorch/workshops/tree/master/ASPLOS_2024
  - https://docs.google.com/document/d/1zY9Nlmh5jT39Q92aDYf3dHOAXh2xCMdeS6pnbX0Dqpw/edit?usp=drivesdk
 
### Other useful materials
- [GPU MODE Youtube channel](https://www.youtube.com/@GPUMODE)
- [AI compilers study](https://carpedm30.notion.site/AI-Compiler-Study-aaf4cff2c8734e50ad95ac6230dbd80b)
- [Matrix Multiplication on Blackwell](https://www.modular.com/blog/matrix-multiplication-on-nvidias-blackwell-part-1-introduction)
- [How to Scale Your Model: A Systems View of LLMs on TPUs](https://jax-ml.github.io/scaling-book/)
- [Domain-specific architectures for AI inference](https://fleetwood.dev/posts/domain-specific-architectures)
