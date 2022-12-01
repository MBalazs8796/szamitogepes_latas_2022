# syntax=docker/dockerfile:1.3-labs
FROM celinachild/orbslam2:latest as base

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

FROM base as orbslam2

COPY orbslam_driver/cpp/ /home/ORB_SLAM2/Examples/Monocular/
COPY <<EOF /tmp/CMakeLists.txt.extend
add_executable(mono_tum_local
Examples/Monocular/mono_tum_local.cc)
target_link_libraries(mono_tum_local \${PROJECT_NAME})
EOF

# Modify source code and rebuild ORB_SLAM2
RUN set -x \
    && cd /home/ORB_SLAM2 \
    && cat /tmp/CMakeLists.txt.extend >> CMakeLists.txt \
    # Disable viewer
    && sed -i 's/System::MONOCULAR,true/System::MONOCULAR,false/g' './Examples/Monocular/mono_tum.cc' \
    && chmod +x build.sh \
    && ./build.sh

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

FROM orbslam2

COPY --from=python-dependencies /install/ /usr/

WORKDIR /data
