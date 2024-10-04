# Raspberry Pi 4 compatible server container
FROM arm32v7/python:3.8 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

RUN apt-get update && apt-get install -y \
    wget \
    libffi-dev \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    gcc \
    make \
    build-essential \
    curl \
    libncurses5-dev \
    less \
    git

WORKDIR /usr/src/app

COPY torch-1.7.0a0-cp38-cp38-linux_armv7l.whl /tmp/
RUN pip install /tmp/torch-1.7.0a0-cp38-cp38-linux_armv7l.whl

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

ENV PATH="/root/.cargo/bin:${PATH}"

RUN pip3 install Cython==0.29.36 
RUN pip3 install setuptools==68.0.0
RUN pip3 install cryptography
RUN pip3 install flwr==1.9.0 
RUN pip3 install network==0.1

RUN pip3 install numpy==1.24.4
RUN pip3 install pandas==2.0.3
RUN pip3 install psutil==5.9.8
RUN pip3 install python_dateutil==2.9.0.post0

FROM arm32v7/python:3.8

WORKDIR /usr/src/app

COPY --from=builder /usr/local /usr/local

CMD ["python3", "--version"]

ENV PATH="/usr/local/bin:${PATH}"
COPY --from=builder /root/.cargo/bin /root/.cargo/bin

COPY . .

RUN apt-get update && apt-get install -y git
RUN apt-get update && apt-get install -y libopenblas-dev

ARG GITHUB_TOKEN

RUN git clone --depth 1 https://${GITHUB_TOKEN}@github.com/repo_name /app/repo && \
    rm -rf /app/repo/.git

WORKDIR /app/repo

ENTRYPOINT ["python3", "flower_server.py"]