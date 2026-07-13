# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from collections.abc import Callable
from typing import Optional, Iterable

import torch
from torch import nn

from transformers.integrations import use_kernel_func_from_hub, use_kernelized_func

from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, logging
from transformers.utils.generic import maybe_autocast
from transformers.models.llama.configuration_llama import LlamaConfig

from vllm.config import VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.v1.attention.backend import AttentionType
from .utils import (
    AutoWeightsLoader,
    maybe_prefix,
)



logger = logging.get_logger(__name__)


class MyLlamaRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, vllm_config: VllmConfig, prefix: str, device=None):
        super().__init__()
        self.vllm_config = vllm_config
        self.config = self.vllm_config.model_config.hf_config

        self.rope_type = self.config.rope_parameters["rope_type"]
        rope_init_fn: Callable = self.compute_default_rope_parameters
        if self.rope_type != "default":
            rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        inv_freq, self.attention_scaling = rope_init_fn(self.config, device)

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("original_inv_freq", inv_freq.clone(), persistent=False)

    @staticmethod
    def compute_default_rope_parameters(
        config: LlamaConfig | None = None,
        device: Optional["torch.device"] = None,
        seq_len: int | None = None,
    ) -> tuple["torch.Tensor", float]:
        """
        Computes the inverse frequencies according to the original RoPE implementation
        Args:
            config ([`~transformers.PreTrainedConfig`]):
                The model configuration.
            device (`torch.device`):
                The device to use for initialization of the inverse frequencies.
            seq_len (`int`, *optional*):
                The current sequence length. Unused for this type of RoPE.
        Returns:
            Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
            post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
        """
        base = config.rope_parameters["rope_theta"]
        dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads

        attention_factor = 1.0  # Unused in this type of RoPE

        # Compute the inverse frequencies
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
        )
        return inv_freq, attention_factor

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[:, None].float().expand(-1, 1).to(x.device)
        position_ids_expanded = position_ids[None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with maybe_autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(0, 1)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


@use_kernel_func_from_hub("rotary_pos_emb")
def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class MyLlamaMLP(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str):
        super().__init__()
        self.vllm_config = vllm_config
        self.config = self.vllm_config.model_config.hf_config

        self.gate_proj = ColumnParallelLinear(
            input_size=self.config.hidden_size,
            output_size=self.config.intermediate_size,
            bias=self.config.mlp_bias,
            quant_config=self.vllm_config.quant_config,
            prefix=f"{prefix}.gate_proj"
        )

        self.up_proj = ColumnParallelLinear(
            input_size=self.config.hidden_size,
            output_size=self.config.intermediate_size,
            bias=self.config.mlp_bias,
            quant_config=self.vllm_config.quant_config,
            prefix=f"{prefix}.up_proj"
        )

        self.act_fn = nn.functional.silu

        self.down_proj = RowParallelLinear(
            input_size=self.config.intermediate_size,
            output_size=self.config.hidden_size,
            bias=self.config.mlp_bias,
            quant_config=self.vllm_config.quant_config,
            prefix=f"{prefix}.down_proj"
        )

    def forward(self, x):
        gate_proj, _ = self.gate_proj(x)
        up_proj, _ = self.up_proj(x)
        down_proj, _ = self.down_proj(self.act_fn(gate_proj) * up_proj)
        return down_proj


@use_kernelized_func(apply_rotary_pos_emb)
class MyLlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, vllm_config: VllmConfig, prefix: str, layer_idx: int):
        super().__init__()
        self.vllm_config = vllm_config
        self.config = self.vllm_config.model_config.hf_config
        self.layer_idx = layer_idx
        self.head_dim = getattr(self.config, "head_dim",
                                self.config.hidden_size // self.config.num_attention_heads)
        self.scaling = self.head_dim**-0.5

        tp_size = get_tensor_model_parallel_world_size()
        self.num_heads = self.config.num_attention_heads // tp_size
        self.num_kv_heads = max(1, self.config.num_key_value_heads // tp_size)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        self.qkv_proj = QKVParallelLinear(
            hidden_size=self.config.hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.config.num_attention_heads,
            total_num_kv_heads=self.config.num_key_value_heads,
            bias=self.config.attention_bias,
            quant_config=self.vllm_config.quant_config,
            prefix=f"{prefix}.qkv_proj",
        )

        self.o_proj = RowParallelLinear(
            input_size=self.config.num_attention_heads * self.head_dim,
            output_size=self.config.hidden_size,
            bias=self.config.attention_bias,
            quant_config=self.vllm_config.quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=self.vllm_config.cache_config,
            quant_config=self.vllm_config.quant_config,
            per_layer_sliding_window=None,
            attn_type=AttentionType.DECODER,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape
        hidden_shape = (input_shape[0], -1, self.head_dim)

        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        cos, sin = position_embeddings
        q, k = q.view(hidden_shape), k.view(hidden_shape)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        q = q.reshape(-1, self.q_size)
        k = k.reshape(-1, self.kv_size)

        attn_output = self.attn(q, k, v)

        attn_output = attn_output.reshape(input_shape[0], -1).contiguous()
        attn_output, _ = self.o_proj(attn_output)
        return attn_output


class MyLlamaDecoderLayer(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str, layer_idx: int):
        super().__init__()
        self.vllm_config = vllm_config
        self.config = self.vllm_config.model_config.hf_config

        self.self_attn = MyLlamaAttention(self.vllm_config, f"{prefix}.self_attn", layer_idx=layer_idx)

        self.mlp = MyLlamaMLP(self.vllm_config, f"{prefix}.mlp")
        self.input_layernorm = RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)

    def forward(
        self,
        position_embeddings: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class MyLlamaModel(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str):
        super().__init__()
        self.vllm_config = vllm_config
        self.config = self.vllm_config.model_config.hf_config

        self.embed_tokens = VocabParallelEmbedding(
            self.config.vocab_size,
            self.config.hidden_size,
            quant_config=self.vllm_config.quant_config,
        )

        self.layers = nn.ModuleList(
            [MyLlamaDecoderLayer(self.vllm_config, f"{prefix}.layers.{layer_idx}", layer_idx)
             for layer_idx in range(self.config.num_hidden_layers)]
        )
        self.norm = RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)
        # NOTE: this rotary_emb is used for position embedding,
        # it seems vllm have different impl.
        self.rotary_emb = MyLlamaRotaryEmbedding(self.vllm_config, f"{prefix}.rotary_emb")

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(self,
                input_ids: torch.Tensor | None,
                positions: torch.Tensor,
                intermediate_tensors=None,
                inputs_embeds=None):
        hidden_states = self.embed_tokens(input_ids)
        position_embeddings = self.rotary_emb(hidden_states, position_ids=positions)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(position_embeddings, hidden_states)

        hidden_states = self.norm(hidden_states)
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            # (".gate_up_proj", ".gate_proj", 0),
            # (".gate_up_proj", ".up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class MyLlamaForCausalLM(nn.Module):
    # NOTE: 哪些模块需要 vllmConfig 和 prefix, 哪些不需要, 为什么?
    # xxxForCausalLM 和 xxxModel 是一定需要的;
    # 在这两个之外, 其他子模块, 凡是用到了 vllm 自定义模块的, 都需要这两个入参 (因为 vllm 自定义模块就需要这两个入参);
    # NOTE: 哪些模块需要 embed_input_ids, 哪些不需要, 为什么?
    # xxxForCausalLM 和 xxxModel 都需要, 前者只需要调用后者的接口就可以;
    # NOTE: 哪些模块需要 compute_logits, 哪些不需要, 为什么?
    # 只有 xxxForCausalLM 需要;
    def __init__(self, vllm_config: VllmConfig, prefix: str):
        super().__init__()
        self.vllm_config = vllm_config
        self.config = self.vllm_config.model_config.hf_config

        self.model = MyLlamaModel(self.vllm_config, prefix=maybe_prefix(prefix, "model"))
        self.lm_head = ParallelLMHead(
            self.config.vocab_size,
            self.config.hidden_size,
            quant_config=self.vllm_config.quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        logit_scale = getattr(self.config, "logit_scale", 1.0)
        self.logits_processor = LogitsProcessor(
            self.config.vocab_size, scale=logit_scale
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(self, input_ids: torch.Tensor | None,
                positions: torch.Tensor,
                intermediate_tensors=None,
                inputs_embeds=None):
        return self.model(input_ids, positions)
    
    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."] if self.config.tie_word_embeddings else None),
        )
        return loader.load_weights(weights=weights)