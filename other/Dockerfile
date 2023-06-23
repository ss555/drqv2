# Use the official image as a parent image
#!/bin/bash

CPU_PARENT=ubuntu:20.04
GPU_PARENT=nvidia/cuda:11.3-cudnn8-runtime-ubuntu20.04

TAG=stablebaselines/stable-baselines3
VERSION=$(cat ./stable_baselines3/version.txt)

if [[ ${USE_GPU} == "True" ]]; then
  PARENT=${GPU_PARENT}
  PYTORCH_DEPS="cudatoolkit=11.3"
else
  PARENT=${CPU_PARENT}
  PYTORCH_DEPS="cpuonly"
  TAG="${TAG}-cpu"
fi

echo "docker build --build-arg PARENT_IMAGE=${PARENT} --build-arg PYTORCH_DEPS=${PYTORCH_DEPS} -t ${TAG}:${VERSION} ."
docker build --build-arg PARENT_IMAGE=${PARENT} --build-arg PYTORCH_DEPS=${PYTORCH_DEPS} -t ${TAG}:${VERSION} .
docker tag ${TAG}:${VERSION} ${TAG}:latest

#COPY ./custom_Mujoco_tasks/.mujoco/ .mujoco
ENV MJLIB_PATH=/root/.mujoco/bin/libmujoco210.so
ENV MUJOCO_GL=egl
ENV LD_LIBRARY_PATH /.mujoco/mujoco210/bin:${LD_LIBRARY_PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib64:${LD_LIBRARY_PATH}
ENV PYOPENGL_PLATFORM=egl


# i need this for ffmpeg
#ENV PATH="/tf/.local/bin/:${PATH}"

RUN apt-cache search mesa
RUN apt-get update && apt-get install -y libgl1-mesa-glx libosmesa6
