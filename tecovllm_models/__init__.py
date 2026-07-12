

from vllm import ModelRegistry

def init():
    ModelRegistry.register_model(
        model_arch="Qwen3DenseForCausalLM", model_cls="tecovllm_models.models.qwen3_dense:Qwen3DenseForCausalLM"
    )
