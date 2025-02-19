ARG PYTHON_VER=3.13

# 使用指定版本的官方Python基础镜像
FROM python:${PYTHON_VER}-alpine AS base

# 下载代码并安装依赖
RUN apk add --update git gcc python3-dev musl-dev linux-headers build-base \
    && git clone https://github.com/nov12/summary-chinese-futures-market /codes \
    && cd /codes \
    && git checkout dev \
    && python -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir --upgrade pip \
    && /opt/venv/bin/pip install --no-cache-dir -r requirements.txt \
    && echo "" > /codes/__init__.py


# 创建最终镜像
FROM python:${PYTHON_VER}-alpine AS finale

# 设置工作目录和卷
WORKDIR /workdir
VOLUME /workdir

# 从基础阶段复制 Python 虚拟环境
COPY --from=base /opt/venv /opt/venv
COPY --from=base /codes /pythonlib/summary/

# 激活虚拟环境
ENV PATH="/opt/venv/bin:$PATH"
ENV PYPATH="/pythonlib"
ENV TZ=Asia/Shanghai

# 指定运行应用程序的命令
COPY --from=base /codes/entrypoint.sh /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["python", "-m", "summary.main"]
