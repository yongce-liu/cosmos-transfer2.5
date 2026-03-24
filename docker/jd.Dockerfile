FROM nvcr.io/nvidia/pytorch:25.10-py3

#####################################################################
# FOR UBUNTU MIRRORS AMD
RUN CODENAME=$(grep VERSION_CODENAME /etc/os-release | cut -d= -f2) && \
    echo "Detected Ubuntu codename: $CODENAME" && \
    \
    sed -i 's@//.*archive.ubuntu.com@//mirrors.ustc.edu.cn@g' /etc/apt/sources.list && \
    sed -i 's@security.ubuntu.com@mirrors.ustc.edu.cn@g' /etc/apt/sources.list && \
    \
    find /etc/apt/sources.list.d/ -name "*.list" -type f -exec sed -i \
    -e 's@//.*archive.ubuntu.com@//mirrors.ustc.edu.cn@g' \
    -e 's@security.ubuntu.com@mirrors.ustc.edu.cn@g' \
    -e 's@ppa:.*@# &@g' {} \; && \
    \
    apt-get clean && \
    apt-get update -y && \
    apt-get upgrade -y
#####################################################################
# INSTALL ZSH
RUN apt-get update && apt-get install zsh git tmux -y && chsh -s /bin/zsh && \
    git clone https://github.com/robbyrussell/oh-my-zsh.git ${HOME}/.oh-my-zsh \
    && cp ${HOME}/.oh-my-zsh/templates/zshrc.zsh-template ${HOME}/.zshrc && \
    git clone https://github.com/zsh-users/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-${HOME}/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting && \
    git clone https://github.com/zsh-users/zsh-autosuggestions.git ${ZSH_CUSTOM:-${HOME}/.oh-my-zsh/custom}/plugins/zsh-autosuggestions && \
    sed -i 's/plugins=(git)/plugins=(git zsh-syntax-highlighting zsh-autosuggestions)/g' ${HOME}/.zshrc
SHELL [ "/bin/bash", "-c" ]
#####################################################################
# FOR PYPI MIRRORS
RUN apt-get install -y python3-pip && \
    pip install -i https://mirrors.ustc.edu.cn/pypi/simple pip -U && \
    pip config set global.index-url https://mirrors.ustc.edu.cn/pypi/simple
# RUN ln -s /usr/bin/python3 /usr/bin/python
# ENV PATH=/usr/bin:/usr/local/bin:$PATH
#####################################################################
# FOR UV PIP
RUN pip install uv && \
    mkdir -p ${HOME}/.config/uv && \
    touch ${HOME}/.config/uv/uv.toml && \
    echo -e "[[index]]\nurl = \"https://mirrors.ustc.edu.cn/pypi/simple\"\ndefault = true" > ${HOME}/.config/uv/uv.toml
#####################################################################

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
    pip install ./packages/* && \
    pip install .[cu130]

CMD ["/bin/bash"]
