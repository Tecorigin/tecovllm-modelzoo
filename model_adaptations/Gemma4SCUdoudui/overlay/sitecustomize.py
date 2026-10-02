"""受控启动期注入点（overlay）。

用法（不改 site-packages、不改 transformers/vLLM）：

    export GEMMA4_OVERLAY=1
    export PYTHONPATH=/root/SichuanUniversity_AI_accelerator-gemma/overlay:$PYTHONPATH
    /home/py312/bin/python -m vllm.entrypoints.openai.api_server --model ... --config-format gemma4_shim

`site` 模块在解释器启动时自动 import sitecustomize；PYTHONPATH 上的本目录先于
site-packages 命中，因此在 vLLM 读取 config / 构建 tokenizer 之前完成注册。
必须显式设置 GEMMA4_OVERLAY=1 才生效，避免污染同机其他会话进程。
"""

import os


def install_block_attention_overlay() -> None:
    """把厂商未实现的 BlockAttentionImpl.forward 换成 in-repo 实现。

    厂商缺口：`vllm_sdaa/attention/block_attn.py:881` 的
    `BlockAttentionImpl.forward` 无条件 `raise NotImplementedError
    ("flash attention is not implemented!")`，而 `platform.py:263` 又把
    BlockAttentionBackend 作为唯一的默认 attn 后端返回。结果是任何 SDAA vLLM V1
    模型都会在 KV profiling 的前向里直接死掉。

    修复：进程初始化期绑定 `custom_ops.block_attention.op.sdaa_block_attention_forward`
    （转发到 `_C_cache_ops.reshape_and_cache_flash` 与 `_C_sdaa.block_attention`）。
    不改 site-packages；热路径无分支、无 getenv。
    参考实现来源：model/internvl3_5-8b（已在本目录 custom_ops/ 内注明 provenance）。
    """
    import sys

    try:
        import vllm_sdaa.attention.block_attn as block_attn_module
        from custom_ops.block_attention.op import sdaa_block_attention_forward
    except Exception as exc:
        print(f"[gemma4-overlay] FATAL: block attention bind failed: {exc}", file=sys.stderr)
        raise

    impl_cls = block_attn_module.BlockAttentionImpl
    impl_cls.forward = sdaa_block_attention_forward
    if impl_cls.forward is not sdaa_block_attention_forward:
        raise RuntimeError("block attention identity check failed")

    # 第二个厂商缺口：`_C_sdaa.block_attention` 只接受
    # window_size_left ∈ {-1, 127, 511}（见 host_block_multihead_attention.cpp:252 的
    # 断言）。Gemma4 的 sliding_window=1024 会被 BlockAttentionImpl.__init__ 算成
    # `sliding_window - 1 = 1023`，于是 decode 路径直接触发
    #   Assertion `(window_size_left == -1 || window_size_left == 127 ||
    #               window_size_left == 511)` failed
    #   -> RuntimeError: TECOCUSTOM_STATUS_SUCCESS INTERNAL ASSERT FAILED
    #      at aten/native/adaptor/attention_adaptor.cpp:112
    #
    # 归一化在**构造期**一次性完成，不在 forward 热路径加分支（AGENTS.md 规则 6）。
    # 语义说明：1024 无对应的合法窗口值，取 -1（不设窗）。
    #   * 序列 ≤ 1024 token 时与窗口 1024 完全等价（所有 key 本来就在窗内）；
    #   * 序列 > 1024 时 sliding 层会看到超出训练窗口的 key —— 是**近似**，
    #     会偏离训练语义（质量风险），但不崩溃。已知边界见 experiments 记录。
    if not getattr(impl_cls, "_gemma4_window_patch", False):
        _kernel_windows = (-1, 127, 511)
        _orig_init = impl_cls.__init__

        def __init__(self, *args, **kwargs):
            _orig_init(self, *args, **kwargs)
            left, right = self.sliding_window
            # vLLM stores a causal window W as W-1. Preserve the semantic
            # window before normalizing the vendor-only field below; the
            # in-repo Gemma kernel consumes the real W (1024 for D256).
            self._gemma4_window_size = left + 1 if left >= 0 else None
            if left not in _kernel_windows:
                self.sliding_window = (-1, right)
            # 第三个厂商缺口：flash SDP 只支持 head_dim <= 256，而 Gemma4 的 8 个
            # full_attention 层 global_head_dim=512。按 head_size 在构造期一次性
            # 绑定 prefill/decode 实现（head_size>256 走显式 math），
            # forward 内不保留后端 if-else（AGENTS.md 规则 6）。
            from custom_ops.block_attention.op import bind_attention_impls

            bind_attention_impls(self)

        impl_cls.__init__ = __init__
        impl_cls._gemma4_window_patch = True

    print("[gemma4-overlay] BlockAttention overlay bound successfully.", file=sys.stderr)


if os.environ.get("GEMMA4_OVERLAY") == "1":
    try:
        import gemma4_shim

        # config shim + tokenizer 兼容 + Gemma4 文本主干 mm 权重跳过
        gemma4_shim.install()
        # 厂商未实现的 SDAA BlockAttention 前向
        install_block_attention_overlay()
    except Exception as exc:  # 启动期失败必须可见
        import sys

        print(f"[gemma4-overlay] FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
