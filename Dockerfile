FROM ubuntu:16.04
# Based on Keras Face classification project
MAINTAINER Daniele Giunchi <dannox1975@gmail.com>

# Supress warnings about missing front-end. As recommended at:
# http://stackoverflow.com/questions/22466255/is-it-possibe-to-answer-dialog-questions-when-installing-under-docker
ARG DEBIAN_FRONTEND=noninteractive

# Essentials: developer tools, build tools, OpenBLAS
RUN apt-get update && apt-get install -y --no-install-recommends \
    apt-utils git curl vim unzip openssh-client wget \
    build-essential cmake \
    libopenblas-dev

# Python 3.5
RUN apt-get install -y --no-install-recommends python3.5 python3.5-dev python3-pip python3-tk && \
    pip3 install --no-cache-dir --upgrade pip setuptools && \
    echo "alias python='python3'" >> /root/.bash_aliases && \
    echo "alias pip='pip3'" >> /root/.bash_aliases
RUN apt-get install -y --no-install-recommends libjpeg-dev zlib1g-dev && \
    pip3 --no-cache-dir install Pillow
RUN pip3 --no-cache-dir install \
    numpy scipy  scikit-image  matplotlib Cython

# Tensorflow 1.6.0 - CPU
RUN pip3 install --no-cache-dir --upgrade tensorflow

#EXPOSE 8008

# Dependencies
RUN apt-get install -y --no-install-recommends \
    libjpeg8-dev libtiff5-dev libjasper-dev libpng12-dev \
    libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libgtk2.0-dev \
    liblapacke-dev checkinstall

#project download
RUN git clone -b 3.4.1 --depth 1 https://github.com/opencv/opencv.git /usr/local/src/opencv
# Compile
RUN cd /usr/local/src/opencv && mkdir build && cd build && \
    cmake -D CMAKE_INSTALL_PREFIX=/usr/local \
          -D BUILD_TESTS=OFF \
          -D BUILD_PERF_TESTS=OFF \
          -D PYTHON_DEFAULT_EXECUTABLE=$(which python3) \
          .. && \
    make -j"$(nproc)" && \
    make install

# Keras
RUN pip3 install --no-cache-dir --upgrade h5py pydot_ng keras

# Install 
RUN  git clone --depth 1  https://github.com/dgiunchi/keras_face.git /usr/local/src/keras_face
RUN cd /usr/local/src/keras_face && $(which python3) setup.py install

RUN wget --quiet https://github.com/dgiunchi/keras_face/models/siamese-face-net-architecture.h5 -O /usr/local/src/keras_face/siamese-face-net-architecture.h5
RUN wget --quiet https://github.com/dgiunchi/keras_face/models/siamese-face-net-config.npy -O /usr/local/src/keras_face/siamese-face-net-config.npy
RUN wget --quiet https://github.com/dgiunchi/keras_face/models/siamese-face-net-weights.h5 -O /usr/local/src/keras_face/siamese-face-net-weights.h5

# CMD ["/bin/bash"]
ENTRYPOINT ["/bin/bash", "-c",  "/usr/local/src/keras_face/siamese_demo_predict.py \"$@\"", "--"]
