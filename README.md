## AI compilers study materials 

### Description
This repository is designed as a focused resource for individuals beginning work in the field of AI compilers and systems.

​By AI compilers, I specifically refer to the software stack responsible for translating and optimizing DNN models into efficient assembly code for various accelerators, including GPUs and NPUs.

​I have curated valuable resources I've encountered to date to provide a solid starting point.

### Prerequisites
- Basic understanding of DNN and Transformer architectures.
- Some experience on parallel programming and GPUs (e.g., implementing matmul on CUDA).

### Introductions
- [A friendly introduction to machine learning compilers and optimizers](https://huyenchip.com/2021/09/07/a-friendly-introduction-to-machine-learning-compilers-and-optimizers.html)
- [Democratizing AI compute: Part 6](https://www.modular.com/blog/democratizing-ai-compute-part-6-what-about-ai-compilers)
  - Strongly recommend to read the full series.
- [Introduction to AI compilers](https://docs.google.com/presentation/d/1RZdV3Z-Q1NEbpU1-qk9C97yE1QvwNLJ9Gc7JLaFLZCw/edit?slide=id.p#slide=id.p)

### State of AI compilers
- In the pre-LLM era, AI compilers consist of **graph optimizers + operator libraries OR compilers**. 
  - *Graph optimizers* perform various computational graph-level rewrites such as operator fusion, redundancy elimination.
  - Each tensor operator in the graph is implemented as an accelerator kernel, e.g., CUDA kernels.
     - *Operator libraries*: programming directly for peak performance 
     - *Operator compilers*: automatically generating for wide coverage with moderate performance 
  - [TVM](https://github.com/apache/tvm) and [XLA](https://github.com/openxla/xla) were the popular end-to-end compiler frameworks.
- After the domination of LLMs, the focus has shifted to **LLM runtimes + template compilers**.
  - For LLMs, AI compilers should maximize the performance of matmul/attention under runtime strategies.
    - DNN architectures have converged into Transformers, of which the core workloads are dense matmul and attention.
    - LLMs inferences are not merely a single forward pass; rather multi-step decoding, which induced runtimes optimizations such as KV caching and iterative batching.
  - Traditional AI compilers, targetted for covering different architectures under simple runtime, fall short on delivering the performance; While direct kernel programming requires too much effort.
  - *Template-based languages* are the middle ground: Users write templates for an operator kernel to give hints and the rest are dealt with their compiler.
    - [CUTLASS](https://docs.nvidia.com/cutlass/index.html) and [Triton](https://github.com/triton-lang/triton) are the popular choices.
    - [FlashInfer](https://github.com/flashinfer-ai/flashinfer) and [FlexAttention](https://pytorch.org/blog/flexattention/) even provide templates specifically for attention.
- There are many unresovled challenges in the world of AI compilers.
  - Programming kernels even with template languages is becoming [increasingly complex with new hardware features](https://arxiv.org/html/2504.07004v1).
  - [Post-Transformer models](https://arxiv.org/pdf/2312.00752) demands different kinds of graph optimizations and operator kernels.
  - Mixed-precision and sparse computation are tightly coupled with AI compilers.
  - There is a discussion on [writing LLM kernels using LLM agents](https://arxiv.org/pdf/2502.10517) (can a full circle be made?).

### Papers
- [List of papers](https://github.com/merrymercy/awesome-tensor-compilers)
- There is [a survey paper](https://arxiv.org/abs/2002.03794), but keep in mind that it is a bit outdated (2020).
- I'd recommend skimming through recent ML/DNN/AI compiler papers on ASPLOS/MLSys/PLDI/ISCA, such as [Relax](https://arxiv.org/pdf/2311.02103), [Cypress](https://arxiv.org/pdf/2504.07004), [Exo2](https://arxiv.org/pdf/2411.07211).
- FlashAttention series and PagedAttention are also recommended to understand the gist of kernel optimization for LLMs[[1](https://arxiv.org/pdf/2205.14135)][[2](https://arxiv.org/pdf/2307.08691)][[3](https://arxiv.org/pdf/2407.08608)][[decoding](https://arxiv.org/pdf/2311.01282)][[paged](https://arxiv.org/pdf/2309.06180)].

### Frameworks
#### Graph compilers + Operator libraries
- NVIDIA [TensorRT](https://github.com/NVIDIA/TensorRT) + [cuDNN](https://developer.nvidia.com/cudnn)
  - Core logics are closed-sourced.
- Intel [OpenVino](https://github.com/openvinotoolkit/openvino) + [oneDNN](https://github.com/uxlfoundation/oneDNN)

#### Graph + Operator compilers
- [TVM](https://github.com/apache/tvm)
- [XLA](https://github.com/openxla/xla)
- [TorchInductor](https://github.com/pytorch/pytorch/tree/main/torch/_inductor)
  - TorchInductor feels like a "modern" version of traditional AI compilers, in the sense that it utilizes Triton for the part of operator compilation and has been getting some boost by the PyTorch team.
  - For example, [vLLMs use TorchInductor to compile non-attention part of the LLMs](https://blog.vllm.ai/2025/08/20/torch-compile.html).

#### LLM runtimes
- [vLLM](https://github.com/vllm-project/vllm)
- [SGLang](https://github.com/sgl-project/sglang)
- NVIDIA [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)

#### Template compilers
- NVIDIA [CUTLASS](https://docs.nvidia.com/cutlass/index.html)
- [Triton](https://github.com/triton-lang/triton)
  - [Gluon](https://github.com/triton-lang/triton/blob/main/python/tutorials/gluon/01-intro.py)
    - For Hopper/Blackwell GPUs Triton struggles to achieve optimal performance due to the increased complexity of GPU.
    - Gluon is being developed within the Triton ecosystem which exposes more lower-level controls akin to CUTLASS.
  - [Helion](https://pytorch.org/blog/helion/)
    - Helion, developed by the PyTorch team, provides the middle ground between TorchInductor and Triton, offering flexbilty to program kernels with templates but hiding details of Triton.
- [TileLang](https://github.com/tile-ai/tilelang) is another template-based language 
 
#### Attention template compilers
- [FlashInfer](https://github.com/flashinfer-ai/flashinfer)
- [FlexAttention](https://pytorch.org/blog/flexattention/)
  - [FlashLight](https://arxiv.org/pdf/2511.02043) has been released very recently (Nov 07, 2025), which surpasses FlexAttention's performance with automatic compilation, being part of PyTorch workflow as the TorchInductor extension.

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
