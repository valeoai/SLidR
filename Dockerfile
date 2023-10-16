ARG PYTORCH="1.12.0"
ARG CUDA="11.3"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

# --------------------------------------------------------------------

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub

RUN DEBIAN_FRONTEND=noninteractive apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y openssh-server sudo

# -------------------------------------------------------------------

##############################################
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 6.2 7.0 7.2 7.5 8.0 8.6+PTX"
##############################################

ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"

# Install dependencies
RUN apt-get update
RUN apt-get install -y git ninja-build cmake build-essential libopenblas-dev \
    xterm xauth openssh-server tmux wget mate-desktop-environment-core

RUN apt-get clean
RUN rm -rf /var/lib/apt/lists/*

# For faster build, use more jobs.
ENV MAX_JOBS=4
RUN git clone --recursive "https://github.com/NVIDIA/MinkowskiEngine"
RUN cd MinkowskiEngine; python setup.py install --force_cuda --blas=openblas

RUN apt-get install libgl1-mesa-glx -y
RUN pip install spconv-cu113

RUN pip install nuscenes-devkit==1.1.9
RUN pip install tqdm pyquaternion

RUN pip install pytorch_lightning==1.6.0
RUN pip install pillow==8.3.1
RUN pip install scikit-image==0.18.2 
