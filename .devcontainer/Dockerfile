FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive

# Install necessary packages
RUN apt-get update && apt-get install -y \
    curl \
    git \
    wget \
    unzip \
    fontconfig \
    xz-utils \
    && rm -rf /var/lib/apt/lists/*

# 创建一个非root用户
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# 创建用户
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    # 添加sudo支持
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 复制requirements.txt并安装依赖
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /tmp/requirements.txt

# 创建保存模型的目录
RUN mkdir -p /workspace/models \
    && mkdir -p /workspace/data \
    && mkdir -p /workspace/static/images \
    && chown -R $USERNAME:$USERNAME /workspace

# 切换到非root用户
USER $USERNAME

# 设置默认命令
CMD ["bash"]