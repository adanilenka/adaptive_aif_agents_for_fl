# Client container for Nvidia GPU devices
FROM nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3 AS builder

WORKDIR /usr/src/app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libffi-dev \
    gcc \
    g++ \
    git \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

ENV PATH="/root/.cargo/bin:${PATH}"

RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir \
    Cython==0.29.36 \
    numpy==1.24.4 \
    river \
    flwr==1.9.0 \
    inferactively_pymdp==0.0.7.1 \
    matplotlib==3.7.5 \
    network==0.1 \
    pandas==2.0.3 \
    pgmpy==0.1.25 \
    plotly==5.22.0 \
    psutil==5.9.8 \
    python_dateutil==2.9.0.post0 \
    scikit_learn==1.3.2 \
    scipy==1.10.1

FROM nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3

WORKDIR /usr/src/app

COPY --from=builder /usr/local/lib/python3.8/dist-packages /usr/local/lib/python3.8/dist-packages
COPY --from=builder /root/.cargo/bin /root/.cargo/bin

ENV PATH="/root/.cargo/bin:${PATH}"

COPY . .

# github access token
ARG GITHUB_TOKEN

RUN git clone --depth 1 https://${GITHUB_TOKEN}@github.com/repo_name /app/repo && \
    rm -rf /app/repo/.git

WORKDIR /app/repo

ENTRYPOINT ["python3", "flower_client.py"]