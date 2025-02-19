#!/bin/sh
# 每次都根据 PYPATH 设置 PYTHONPATH
if [ -z "$PYTHONPATH" ]; then
  export PYTHONPATH="$PYPATH"
else
  export PYTHONPATH="$PYPATH:$PYTHONPATH"
fi

# 执行传入的命令
exec "$@"