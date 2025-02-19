#!/bin/sh
if [ ! -f /tmp/py_env_set ]; then
  # 如果 PYTHONPATH 没有设置，则将其赋值为 PYPATH，否则将 PYPATH 追加到现有 PYTHONPATH 前面
  if [ -z "$PYTHONPATH" ]; then
    export PYTHONPATH="$PYPATH"
  else
    export PYTHONPATH="$PYPATH:$PYTHONPATH"
  fi
  # 创建标记文件表明已经执行过环境变量设置
  touch /tmp/py_env_set
fi

# 执行传入的命令
exec "$@"