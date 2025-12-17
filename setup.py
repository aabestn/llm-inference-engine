import os
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="llm_engine_cuda",
    version="0.1.0",
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    ext_modules=[
        CUDAExtension(
            name="llm_engine_cuda",
            sources=[
                "csrc/src/bindings.cpp",
                "csrc/src/paged_attention.cu",
                "csrc/src/quant_matmul.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "-gencode=arch=compute_80,code=sm_80",  # NVIDIA Ampere
                    "-gencode=arch=compute_89,code=sm_89",  # NVIDIA Ada Lovelace
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    install_requires=[
        "torch>=2.1.0",
        "fastapi>=0.100.0",
        "uvicorn>=0.22.0",
        "pydantic>=2.0.0",
    ],
)