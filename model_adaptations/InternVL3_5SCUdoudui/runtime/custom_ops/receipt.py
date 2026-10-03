"""官方 ``tecoops`` ABI 的**初始化期绑定**调用计数（诊断用）。

用途
----
CP3 要求证明官方算子**在真实模型 forward 里被实际调用**，且调用次数与模型的
「层数 × 步数」结构式对得上。单有「我导入成功了」的收据不能证明这一点，
所以这里绑定一个逐 API 计数器。

热路径契约（AGENTS.md 第 6 条）
------------------------------
所有决策只在 **模块导入期（进程初始化）** 做一次：

* ``TECOOPS_CALL_RECEIPT`` 在导入期通过 ``os.environ`` **只读一次**。
* 未设置时返回**原始 pybind 函数对象本身** —— 交付路径的 forward 里没有计数器、
  没有分支、没有 ``getenv``、没有多余 Python 帧。
* 设置时返回计数绑定，每次调用只做一次 dict 读 + 一次 dict 写，
  **没有后端 / fallback 的 if-else**。

计数在解释器正常退出与 ``SIGTERM``/``SIGINT`` 时落盘，因此被优雅停掉的
engine core 也能给出准确数字。**若收据文件不存在，验证脚本必须报 blocked，
不得自己编一个调用数。**

与 MiniCPM5-1B 分支（只读参考）的差别：本分支按算子名分开计数
（``tecoops.rms_norm`` 与 ``tecoops.rms_norm_add`` 各记一条），
这样收据本身就能与「无残差 / 带残差」两类调用的结构式分别对账。
"""

import atexit
import os
import signal
import sys

_RECEIPT_PATH = os.environ.get("TECOOPS_CALL_RECEIPT")


class _Tally:
    """进程级可变计数器。"""

    __slots__ = ("counts", "path", "_flushed")

    def __init__(self, path: str) -> None:
        self.counts: dict[str, int] = {}
        self.path = path
        self._flushed = False

    def bump(self, name: str) -> None:
        counts = self.counts
        counts[name] = counts.get(name, 0) + 1

    def flush(self, *_args: object) -> None:
        if self._flushed:
            return
        self._flushed = True
        try:
            with open(self.path, "a", encoding="utf-8") as handle:
                for name in sorted(self.counts):
                    handle.write(f"CALL {name}\n")
                    handle.write(f"COUNT {name} {self.counts[name]}\n")
        except OSError as exc:  # pragma: no cover - 诊断不得让主流程崩
            sys.stderr.write(f"[custom_ops.receipt] flush failed: {exc}\n")

    def snapshot(self) -> dict[str, int]:
        return dict(self.counts)


def _make_terminating_handler(tally: "_Tally"):
    """先落盘，再以默认处置重发信号。"""

    def handler(signum, _frame):
        tally.flush()
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    return handler


_TALLY = _Tally(_RECEIPT_PATH) if _RECEIPT_PATH else None

if _TALLY is not None:
    atexit.register(_TALLY.flush)
    for _sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(_sig, _make_terminating_handler(_TALLY))
        except (ValueError, OSError):  # 非主线程或受限平台
            pass


def receipt_enabled() -> bool:
    """本进程是否以调用计数模式启动。"""
    return _TALLY is not None


def bind_official_api(fn, name: str):
    """返回 ``fn`` 本身，或在收据模式下返回计数绑定。

    **只在进程初始化期调用。**
    """
    if _TALLY is None:
        return fn

    tally = _TALLY

    def counted(*args, **kwargs):
        tally.bump(name)
        return fn(*args, **kwargs)

    counted.__name__ = "counted_" + name.replace(".", "_")
    counted.__doc__ = f"Counting binding for {name} (diagnostic launch only)."
    counted.__wrapped__ = fn
    return counted


def snapshot() -> dict[str, int]:
    """单进程 harness 用的进程内视图。"""
    return _TALLY.snapshot() if _TALLY is not None else {}


def receipt_path() -> "str | None":
    return _RECEIPT_PATH
