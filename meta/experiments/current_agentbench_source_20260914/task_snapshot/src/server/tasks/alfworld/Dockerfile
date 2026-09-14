FROM alpine:3.22.0 AS data-downloader

RUN sed -i 's|dl-cdn.alpinelinux.org|mirrors.tuna.tsinghua.edu.cn|g' /etc/apk/repositories && \
    apk add --no-cache unzip

ENV DATA_DIR=/data

WORKDIR $DATA_DIR

COPY data/alfworld/json_2.1.1_json.zip /tmp/json_2.1.1_json.zip
RUN unzip /tmp/json_2.1.1_json.zip -d $DATA_DIR && rm /tmp/json_2.1.1_json.zip

COPY data/alfworld/json_2.1.1_pddl.zip /tmp/json_2.1.1_pddl.zip
RUN unzip /tmp/json_2.1.1_pddl.zip -d $DATA_DIR && rm /tmp/json_2.1.1_pddl.zip

COPY data/alfworld/json_2.1.1_tw-pddl.zip /tmp/json_2.1.1_tw-pddl.zip
RUN unzip /tmp/json_2.1.1_tw-pddl.zip -d $DATA_DIR && rm /tmp/json_2.1.1_tw-pddl.zip

FROM python:3.9-bookworm AS worker

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN sed -i 's|deb.debian.org|mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list.d/debian.sources && \
    sed -i 's|security.debian.org|mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list.d/debian.sources && \
    apt-get update && \
    apt-get install -y build-essential cmake libgl1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY data/alfworld_repo /usr/src/alfworld
COPY data/textworld_src /usr/src/textworld
RUN sed -i '/numpy/d' /usr/src/alfworld/requirements.txt && \
    sed -i '/torch/d' /usr/src/alfworld/requirements.txt && \
    sed -i '/transformers/d' /usr/src/alfworld/requirements.txt && \
    sed -i 's|textworld @ .*|textworld @ file:///usr/src/textworld|' /usr/src/alfworld/requirements.txt && \
    echo "blis<=1.2.0" >> /usr/src/alfworld/requirements.txt && \
    echo "jsonlines" >> /usr/src/alfworld/requirements.txt && \
    echo "numpy<2.0" >> /usr/src/alfworld/requirements.txt && \
    echo "torch<2.0" >> /usr/src/alfworld/requirements.txt && \
    echo "torchvision<0.15" >> /usr/src/alfworld/requirements.txt && \
    echo "transformers<4.48.0" >> /usr/src/alfworld/requirements.txt && \
    echo "Werkzeug<3.0" >> /usr/src/alfworld/requirements.txt && \
    echo "Flask<2.3" >> /usr/src/alfworld/requirements.txt && \
    echo "spacy<3.8.0" >> /usr/src/alfworld/requirements.txt && \
    echo "setuptools<70.0.0" > /tmp/constraint.txt && \
    export PIP_CONSTRAINT=/tmp/constraint.txt && \
    export PIP_BUILD_CONSTRAINT=/tmp/constraint.txt && \
    python -m pip install --upgrade pip -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple && \
    python -m pip install --retries 5 --timeout 300 \
      --index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple \
      --extra-index-url https://download.pytorch.org/whl/cpu \
      agentrl-worker \
      -r /usr/src/alfworld/requirements.txt \
      /usr/src/alfworld

ENV ALFWORLD_DATA=/app/data/alfworld
COPY --from=data-downloader /data /app/data/alfworld

COPY ./src/server/tasks/alfworld /app/src/server/tasks/alfworld
COPY ./configs/tasks/alfworld.yaml /app/configs/tasks/alfworld.yaml
COPY ./data/alfworld /app/data/alfworld

COPY ./extra/worker-entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh", "-c", "configs/tasks/alfworld.yaml"]
