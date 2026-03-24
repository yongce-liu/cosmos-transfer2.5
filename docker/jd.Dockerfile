FROM nvcr.io/nvidia/pytorch:25.10-py3

#####################################################################
# INSTALL ZSH
RUN apt-get update && apt-get install zsh git tmux -y && chsh -s /bin/zsh && \
    git clone https://github.com/robbyrussell/oh-my-zsh.git /root/.oh-my-zsh \
    && cp /root/.oh-my-zsh/templates/zshrc.zsh-template /root/.zshrc && \
    git clone https://github.com/zsh-users/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-/root/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting && \
    git clone https://github.com/zsh-users/zsh-autosuggestions.git ${ZSH_CUSTOM:-/root/.oh-my-zsh/custom}/plugins/zsh-autosuggestions && \
    sed -i 's/plugins=(git)/plugins=(git zsh-syntax-highlighting zsh-autosuggestions)/g' /root/.zshrc
SHELL [ "/bin/bash", "-c" ]

# Install packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    ffmpeg \
    git \
    git-lfs \
    libx11-dev \
    tree \
    wget \
    vim && \
    rm -rf /var/lib/apt/lists/*

# Install just: https://just.systems/man/en/pre-built-binaries.html
RUN curl --proto '=https' --tlsv1.2 -sSf https://just.systems/install.sh | bash -s -- --to /usr/local/bin --tag 1.42.4

# COPY . /root/cosmos-transfer2.5
RUN git clone https://github.com/yongce-liu/cosmos-transfer2.5.git

RUN cd cosmos-transfer2.5 && \
    git lfs install && \
    git lfs pull

RUN cd cosmos-transfer2.5 && \
    uv sync --extra=cu130 -v

RUN cd cosmos-transfer2.5 && \
    uv pip install tyro

RUN cd cosmos-transfer2.5 && \
    export HF_ENDPOINT=https://hf-mirror.com && \
    export HF_TOKEN= && \
    uv run python examples/inference.py -i assets/robot_example/depth/robot_depth_spec.json -o outputs/depth

CMD ["/bin/bash"]
