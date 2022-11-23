# syntax=docker/dockerfile:1.3-labs
FROM ubuntu:18.04 as base-os

ENV DEBIAN_FRONTEND="noninteractive"

RUN set -x \
    && apt-get update \
    && apt-get install --yes \
        git \
    && rm -rf /var/lib/apt/lists/*

FROM base-os as git
WORKDIR /src

FROM git as git-cmake
RUN git clone https://gitlab.kitware.com/cmake/cmake.git . \
    && git checkout tags/v3.22.1

FROM git as git-ORB_SLAM3
RUN git clone -b docker_opencv3.2_fix https://github.com/jahaniam/ORB_SLAM3 .

FROM git as git-Pangolin
RUN git clone https://github.com/stevenlovegrove/Pangolin . \
    && git checkout v0.6

FROM base-os as ros-melodic

RUN set -x \
    && apt-get update \
    && apt-get install --yes \
        locales \
        lsb-release \
        mesa-utils \
        wget \
        curl \
        libssl-dev \
        build-essential \
        dbus-x11 \
        software-properties-common \
        gdb \
        valgrind \
    && rm -rf /var/lib/apt/lists/*

# Locale
RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

# Install ROS
RUN set -x \
    && echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list \
    && apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654 \
    && apt-get update \
    && apt-get install --yes --allow-downgrades --allow-remove-essential --allow-change-held-packages \
        libpcap-dev \
        libopenblas-dev \
        gstreamer1.0-tools libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libgstreamer-plugins-good1.0-dev \
        ros-melodic-desktop-full python-rosinstall python-rosinstall-generator python-wstool build-essential python-rosdep \
        ros-melodic-socketcan-bridge \
        ros-melodic-geodesy \
    && rm -rf /var/lib/apt/lists/*

# Configure ROS
RUN set -x \
    && rosdep init \
    && rosdep update \
    && echo "source /opt/ros/melodic/setup.bash" >> /root/.bashrc \
    && echo "export ROSLAUNCH_SSH_UNKNOWN=1" >> /root/.bashrc

FROM ros-melodic as orbslam3

RUN set -x \
    && apt-get update \
    && apt-get install --yes \
        gnupg2 \
        lsb-core \
        python-pip \
        libpng16-16 \
        libjpeg-turbo8 \
        libtiff5 \
        # Base tools
        # cmake \
        unzip \
        pkg-config \
        python-dev \
        # OpenCV dependencies
        python-numpy \
        # Pangolin dependencies
        libgl1-mesa-dev \
        libglew-dev \
        libpython2.7-dev \
        libeigen3-dev \
        apt-transport-https \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Build Pangolin
COPY --from=git-Pangolin /src/ /tmp/Pangolin/
RUN set -x \
    && mkdir /tmp/Pangolin/build \
    && cd /tmp/Pangolin/build \
    && cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS=-std=c++11 .. \
    && make -j$nproc \
    && make install \
    && cd / \
    && rm -rf /tmp/Pangolin/

# Build ORB_SLAM3
COPY --from=git-ORB_SLAM3 /src/ /ORB_SLAM3/
RUN set -x \
    && cd /ORB_SLAM3 \
    && chmod +x build.sh \
    && ./build.sh

FROM orbslam3 as base

RUN set -x \
    && apt-get update \
    && apt install --yes software-properties-common \
    && add-apt-repository --yes ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install --yes \
        python3.10

RUN set -x \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && update-alternatives --set python3 /usr/bin/python3.10 \
    # python3-pip apt package includes old pip version
    # do manual install instead to get latest version
    && wget --output-document=get-pip.py https://bootstrap.pypa.io/get-pip.py \
    && python3.10 get-pip.py

FROM base as python-dependencies

RUN python3.10 -m pip install pipenv

WORKDIR /app

COPY Pipfile* /app/

# the created requirements.txt includes every dependency from the Pipfile (except PyQt6)
RUN set -x \
    && pipenv requirements > requirements.txt \
    # pip install PyQt6 fails:
    # module 'sipbuild.api' has no attribute 'prepare_metadata_for_build_wheel'
    # https://stackoverflow.com/questions/72275959/pip-install-pyqt6-fails-module-sipbuild-api-has-no-attribute-prepare-metadat
    && sed -i 's/^pyqt6/#pyqt6/' requirements.txt

RUN python3.10 -m pip install \
            --disable-pip-version-check \
            --no-cache-dir \
            --no-warn-script-location \
            --prefix=/install \
            --requirement=requirements.txt

FROM base

COPY --from=python-dependencies /install/ /usr/
COPY <<EOF /entrypoint.sh
#!/bin/sh
exec \$@
EOF

RUN set -x \
    && ln -s /ORB_SLAM3 /home/ORB_SLAM2 \
    && chmod 755 /entrypoint.sh

WORKDIR /data
ENTRYPOINT ["/entrypoint.sh"]
