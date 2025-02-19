#!/bin/sh
# 如果 PYTHONPATH 没有设置，则将其赋值为 PYPATH，否则将 PYPATH 追加到现有 PYTHONPATH 前面
if [ -z "$PYTHONPATH" ]; then
  export PYTHONPATH="$PYPATH"
else
  export PYTHONPATH="$PYPATH:$PYTHONPATH"
fi

# 执行传入的命令
exec "$@"