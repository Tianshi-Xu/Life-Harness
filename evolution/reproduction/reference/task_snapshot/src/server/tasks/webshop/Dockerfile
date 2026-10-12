FROM python:3.9-bullseye AS worker

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TQDM_DISABLE=1

WORKDIR /app

COPY data/webshop_repo /usr/src/webshop
# The gitlink pins the WebShop source revision. Its copied .git file points back
# into the host checkout and is invalid inside the image.
RUN rm -f /usr/src/webshop/.git
COPY --from=longinyu/agentbench-webshop:latest /root/webshop/data /usr/src/webshop/data
RUN rm -f /usr/src/webshop/data/items_shuffle.json

# Generate 100k product data from pre-built resources (memory-efficient subset)
COPY --from=longinyu/agentbench-webshop:latest /root/webshop/search_engine/resources_100k/documents.jsonl /tmp/documents_100k.jsonl
COPY ./src/server/tasks/webshop/convert_100k.py /tmp/convert_100k.py
RUN python3 /tmp/convert_100k.py && rm -f /tmp/documents_100k.jsonl /tmp/convert_100k.py /usr/src/webshop/data/items_ins_v2.json

RUN echo 'deb https://mirrors.tuna.tsinghua.edu.cn/debian bullseye main' > /etc/apt/sources.list && \
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
ENV _JAVA_OPTIONS="-Xmx2g"

COPY ./src/server/tasks/webshop/webshop.patch /tmp/webshop.patch
RUN cd /usr/src/webshop && \
    git apply --check /tmp/webshop.patch && \
    git apply /tmp/webshop.patch && \
    rm /tmp/webshop.patch && \
    sed -i "s|items_shuffle.json'|items_shuffle_100k.json'|" /usr/src/webshop/web_agent_site/utils.py && \
    sed -i "s|items_ins_v2.json'|items_ins_v2_100k.json'|" /usr/src/webshop/web_agent_site/utils.py && \
    mkdir -p /usr/src/webshop/search_engine/indexes /usr/src/webshop/search_engine/resources
COPY --from=longinyu/agentbench-webshop:latest /root/webshop/search_engine/indexes_100k /usr/src/webshop/search_engine/indexes_100k

COPY ./src/server/tasks/webshop /app/src/server/tasks/webshop
COPY ./configs/tasks/webshop.yaml /app/configs/tasks/webshop.yaml

COPY ./extra/worker-entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh", "-c", "configs/tasks/webshop.yaml"]
