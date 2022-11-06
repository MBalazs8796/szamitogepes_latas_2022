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

WORKDIR /data
