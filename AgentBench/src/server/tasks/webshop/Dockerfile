FROM python:3.9-bullseye AS worker

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TQDM_DISABLE=1

WORKDIR /app

COPY data/webshop_repo /usr/src/webshop
RUN cd /usr/src/webshop && \
    git checkout 64fa2a5c15c7daa698b9ac93f5bb5437b634c9bd || true
COPY --from=longinyu/agentbench-webshop:latest /root/webshop/data /usr/src/webshop/data

RUN echo 'deb http://archive.debian.org/debian bullseye main' > /etc/apt/sources.list && \
    echo 'Acquire::Check-Valid-Until "false";' > /etc/apt/apt.conf.d/99no-check-valid && \
    apt-get update && \
    apt-get install -y --no-install-recommends git openjdk-11-jdk-headless && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY ./src/server/tasks/webshop/requirements.txt /app/src/server/tasks/webshop/requirements.txt
RUN echo "setuptools<70.0.0" > /tmp/constraint.txt && \
    echo "spacy<3.8.0" >> /tmp/constraint.txt && \
    echo "thinc<8.3.0" >> /tmp/constraint.txt && \
    export PIP_CONSTRAINT=/tmp/constraint.txt && \
    export PIP_BUILD_CONSTRAINT=/tmp/constraint.txt && \
    python -m pip install --upgrade pip -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple && \
    python -m pip install --retries 5 --timeout 300 \
      --index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple \
      --extra-index-url https://download.pytorch.org/whl/cpu \
      -r /app/src/server/tasks/webshop/requirements.txt \
      agentrl-worker && \
    python -m spacy download en_core_web_sm && \
    python -m spacy download en_core_web_lg

ENV PYTHONPATH="/usr/src/webshop"

COPY ./src/server/tasks/webshop/webshop.patch /tmp/webshop.patch
RUN cd /usr/src/webshop && \
    git apply --check /tmp/webshop.patch && \
    git apply /tmp/webshop.patch && \
    rm /tmp/webshop.patch && \
    mkdir -p /usr/src/webshop/search_engine/indexes /usr/src/webshop/search_engine/resources
COPY --from=longinyu/agentbench-webshop:latest /root/webshop/search_engine/indexes /usr/src/webshop/search_engine/indexes
COPY --from=longinyu/agentbench-webshop:latest /root/webshop/search_engine/resources /usr/src/webshop/search_engine/resources

COPY ./src/server/tasks/webshop /app/src/server/tasks/webshop
COPY ./configs/tasks/webshop.yaml /app/configs/tasks/webshop.yaml

COPY ./extra/worker-entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh", "-c", "configs/tasks/webshop.yaml"]
