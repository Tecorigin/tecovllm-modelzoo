"""Gemma-4-12B-it config/tokenizer shim（overlay，不修改 site-packages）。

背景（两边版本错位）
  * 权重 config.json 由 `transformers_version: 5.10.0.dev0` 写出，`model_type = gemma4_unified`；
    本机 transformers 4.57.6 的 CONFIG_MAPPING 里没有 gemma4，AutoConfig 直接 ValueError。
  * tokenizer_config.json 的 `extra_special_tokens` 是 **list**（5.x 序列化格式），
    而 4.57.6 的 `_set_model_specific_special_tokens` 实现按 **dict** 取 `.keys()/.items()`，
    于是抛 AttributeError: 'list' object has no attribute 'keys'。
  * vLLM 自带 gemma4 模型实现（model_executor/models/gemma4.py）与
    `MODEL_ARCH_CONFIG_CONVERTORS["gemma4"]`，但 `_CONFIG_REGISTRY` 里没有 gemma4，
    所以仅靠 `--hf-overrides {"model_type": "gemma4"}` 无法让 AutoConfig 认得磁盘上的
    `gemma4_unified`（AutoConfig 仍然按磁盘 config.json 的 model_type 分派）。

本模块提供两条不升级 transformers 的最小路径：
  P1  `register_autoconfig_shim()`   —— 把 Gemma4UnifiedConfig 注入 AutoConfig，之后
      `AutoConfig.from_pretrained(...)` 正常返回可用的嵌套 config。
  P2  `Gemma4ShimConfigParser`       —— 用 vLLM 公开 API `register_config_parser()` 注册
      自定义 config format，完全绕开 transformers 的 dispatch。

另附 `patch_tokenizer_extra_special_tokens()` 修掉 list/dict 不兼容。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from transformers import PretrainedConfig

try:  # vLLM 是可选依赖：只做 tokenizer shim 时无需导入
    from vllm.transformers_utils.config_parser_base import ConfigParserBase
except Exception:  # pragma: no cover - 仅用于降级报错

    class ConfigParserBase:  # type: ignore[no-redef]
        pass


__all__ = [
    "Gemma4UnifiedConfig",
    "Gemma4UnifiedTextConfig",
    "Gemma4UnifiedVisionConfig",
    "Gemma4UnifiedAudioConfig",
    "Gemma4ShimConfigParser",
    "register_autoconfig_shim",
    "register_config_parser_shim",
    "patch_tokenizer_extra_special_tokens",
    "install",
]

CONFIG_FORMAT_NAME = "gemma4_shim"


class Gemma4UnifiedTextConfig(PretrainedConfig):
    model_type = "gemma4_unified_text"


class Gemma4UnifiedVisionConfig(PretrainedConfig):
    model_type = "gemma4_unified_vision"


class Gemma4UnifiedAudioConfig(PretrainedConfig):
    model_type = "gemma4_unified_audio"


_SUBCONFIGS = {
    "text_config": Gemma4UnifiedTextConfig,
    "vision_config": Gemma4UnifiedVisionConfig,
    "audio_config": Gemma4UnifiedAudioConfig,
}


def _coerce(value: Any, cls: type[PretrainedConfig]) -> Any:
    if value is None or isinstance(value, PretrainedConfig):
        return value
    if isinstance(value, dict):
        return cls(**value)
    return value


class Gemma4UnifiedConfig(PretrainedConfig):
    """config.json (model_type=gemma4_unified) 的最小可用映射。

    只做「嵌套子配置对象化」，不新增/改写任何数值字段——所有数值原样来自 config.json。
    """

    model_type = "gemma4_unified"

    def __init__(
        self,
        text_config: Any = None,
        vision_config: Any = None,
        audio_config: Any = None,
        **kwargs: Any,
    ) -> None:
        kwargs.pop("transformers_version", None)
        super().__init__(**kwargs)
        self.text_config = _coerce(text_config, Gemma4UnifiedTextConfig)
        self.vision_config = _coerce(vision_config, Gemma4UnifiedVisionConfig)
        self.audio_config = _coerce(audio_config, Gemma4UnifiedAudioConfig)


def register_autoconfig_shim(*, patch_convertor: bool = True) -> None:
    """路径 P1：把 shim config 注册进 transformers AutoConfig（仅进程内，不改磁盘）。"""
    from transformers import AutoConfig

    AutoConfig.register("gemma4_unified", Gemma4UnifiedConfig, exist_ok=True)
    AutoConfig.register("gemma4_unified_text", Gemma4UnifiedTextConfig, exist_ok=True)
    AutoConfig.register("gemma4_unified_vision", Gemma4UnifiedVisionConfig, exist_ok=True)
    AutoConfig.register("gemma4_unified_audio", Gemma4UnifiedAudioConfig, exist_ok=True)
    if patch_convertor:
        register_convertor_alias()


def register_convertor_alias() -> None:
    """让 vLLM 的 Gemma4 专用 convertor 也覆盖 gemma4_unified*。

    不做这一步时 ModelConfig.get_model_arch_config() 会静默回退到
    ModelArchConfigConvertorBase，丢掉两个 Gemma4 特化：
      * is_mm_prefix_lm()  -> use_bidirectional_attention == "vision"
      * get_head_size()    -> max(head_dim=256, global_head_dim=512) = 512
    """
    # 先让 vllm.config 完成初始化：model_arch_config_convertor 与 vllm.config.model
    # 互相 import，直接导入子模块会触发 partially initialized module 循环导入错误。
    import vllm.config  # noqa: F401
    from vllm.transformers_utils.model_arch_config_convertor import (
        MODEL_ARCH_CONFIG_CONVERTORS,
        Gemma4ModelArchConfigConvertor,
    )

    for mt in ("gemma4_unified", "gemma4_unified_text"):
        MODEL_ARCH_CONFIG_CONVERTORS.setdefault(mt, Gemma4ModelArchConfigConvertor)


class Gemma4ShimConfigParser(ConfigParserBase):
    """路径 P2：vLLM `register_config_parser()` 用的自定义 config parser。

    直接读 config.json，绕开 transformers 的 model_type 分派。
    让 `hf_config.model_type` 保持 checkpoint 原值 `gemma4_unified`（不作假），
    Gemma4 特化行为由 register_convertor_alias() 补上。
    """

    def parse(
        self,
        model: str | Path,
        trust_remote_code: bool = False,
        revision: str | None = None,
        code_revision: str | None = None,
        **kwargs: Any,
    ) -> tuple[dict, PretrainedConfig]:
        raw = json.loads((Path(model) / "config.json").read_text())
        config = Gemma4UnifiedConfig(**raw)
        return raw, config


def register_config_parser_shim() -> None:
    from vllm.transformers_utils.config import register_config_parser

    register_config_parser(CONFIG_FORMAT_NAME)(Gemma4ShimConfigParser)
    register_convertor_alias()


def patch_tokenizer_extra_special_tokens() -> None:
    """修 `extra_special_tokens` 为 list 时的 AttributeError（进程内 patch）。

    4.57.6 的实现签名注解是 list[str]，但函数体用 .keys()/.items()，即事实上的 dict 接口。
    checkpoint 交付的是 list（5.x 序列化），这里把 list 规范化为
    {"video_token": "<|video|>"} —— 与 config.json 的 video_token_id=258884 对应。
    """
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

    if getattr(PreTrainedTokenizerBase, "_gemma4_list_shim", False):
        return
    orig = PreTrainedTokenizerBase._set_model_specific_special_tokens

    def patched(self, special_tokens):
        if isinstance(special_tokens, (list, tuple)):
            # 5.x 的 list 形式按顺序落成命名字段；video 是 gemma4 唯一的多模态额外 token
            names = ("video_token",)
            special_tokens = {
                names[i] if i < len(names) else f"extra_token_{i}": tok
                for i, tok in enumerate(special_tokens)
            }
        return orig(self, special_tokens)

    patched._gemma4_orig = orig
    PreTrainedTokenizerBase._set_model_specific_special_tokens = patched
    PreTrainedTokenizerBase._gemma4_list_shim = True


# --------------------------------------------------------------------------
# 厂商缺口：Gemma4ForCausalLM 的 mm 权重跳过名单不覆盖本 checkpoint 的命名
# --------------------------------------------------------------------------
# `vllm/model_executor/models/gemma4.py:1710-1716` 的跳过名单是
#     ["audio_tower.", "vision_tower.", "embed_audio.", "embed_vision."]
# 但本 checkpoint 的视觉 patch embedder 前缀是 `model.vision_embedder.`
# （实测张量：vision_embedder.{patch_dense.weight,patch_dense.bias,
#  patch_ln1.{weight,bias},patch_ln2.{weight,bias},pos_embedding,
#  pos_norm.{weight,bias}}，共 9 个）。
# 且 "vision_embedder" / "patch_dense" 在 vLLM 全部 Python 源码中出现 **0 次**
# （grep 验证），即文本主干与多模态壳都没有任何参数与之对应。
# 结果：Gemma4Model.load_weights 落到 `param = params_dict[name]` →
#     KeyError: 'vision_embedder.patch_dense.weight'
# 文本生成不需要视觉/音频侧权重，最小修复 = 在 overlay 里把该前缀并入跳过集。
GEMMA4_MM_ONLY_PREFIXES: tuple[str, ...] = (
    "vision_embedder.",
    "vision_tower.",
    "audio_tower.",
    "embed_vision.",
    "embed_audio.",
)


def is_mm_only_weight(name: str) -> bool:
    """该 checkpoint 张量是否只属于多模态侧（文本生成不需要）。"""
    return any(p in name for p in GEMMA4_MM_ONLY_PREFIXES)


def iter_lm_only_weights(weights):
    """过滤掉 mm-only 张量，只把文本主干权重交给厂商 loader。"""
    for name, weight in weights:
        if is_mm_only_weight(name):
            continue
        yield name, weight


def patch_gemma4_text_mm_weight_skip() -> None:
    """让 Gemma4ForCausalLM 忽略本 checkpoint 的 mm-only 权重前缀（进程内 patch）。

    不改 site-packages：只包装 `Gemma4ForCausalLM.load_weights`，在进入厂商实现之前
    过滤 checkpoint 的权重流。模型自身的 LM 参数不受影响（666/677 个张量保留）。
    """
    from vllm.model_executor.models.gemma4 import Gemma4ForCausalLM

    if getattr(Gemma4ForCausalLM, "_gemma4_mm_skip_patch", False):
        return
    orig_load_weights = Gemma4ForCausalLM.load_weights

    def load_weights(self, weights):
        return orig_load_weights(self, iter_lm_only_weights(weights))

    load_weights._gemma4_orig = orig_load_weights
    Gemma4ForCausalLM.load_weights = load_weights
    Gemma4ForCausalLM._gemma4_mm_skip_patch = True


def install() -> None:
    """一次性装齐 Config(P1) + Tokenizer + 权重过滤 三个 shim。"""
    patch_tokenizer_extra_special_tokens()
    register_autoconfig_shim()
    patch_gemma4_text_mm_weight_skip()
