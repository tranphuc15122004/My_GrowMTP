# Copyright 2023-2024 SGLang Team
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
# ==============================================================================

"""Inference-only dense Qwen3 MTP draft model (GrowMTP).

This is the speculative-decoding *draft* class for a plain dense Qwen3 backbone
(e.g. Qwen3-4B, model_type=qwen3) carrying a single injected MTP head under the
`mtp.*` keys. It mirrors ``qwen3_5_mtp.py`` but with the dense-Qwen3 deltas:

  * inner backbone is :class:`Qwen3Model` (returns hidden states; plain
    full-attention, QK-norm, head_dim 128) -- NOT the hybrid Qwen3.5 stack.
  * standard ``RMSNorm`` (x*w) for the pre-fc norms, matching transformers
    ``Qwen3RMSNorm`` so the saved random-init weights load with identical
    semantics (Qwen3.5 uses GemmaRMSNorm = x*(1+w)).
  * no ``full_attention_interval`` (that is a Qwen3.5 hybrid-attn knob).
  * ``load_weights`` does NOT strip ``.self_attn.`` -- Qwen3DecoderLayer keeps
    its attention params under ``self_attn.qkv_proj`` / ``self_attn.o_proj``.
  * no MoE.

Embedding / lm_head are shared from the target via ``set_embed_and_head`` (the
eagle worker injects them), so ``load_weights`` only consumes ``mtp.*`` keys.
"""

import logging
import os
from typing import Iterable, Optional, Tuple

import torch
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.distributed import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.qwen3 import Qwen3Model
from sglang.srt.utils import add_prefix

logger = logging.getLogger(__name__)


class Qwen3ForCausalLMMTP(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config=None,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)

        # The MTP draft is unquantized in the from-scratch bf16 checkpoint.
        if quant_config and quant_config.get_name() == "modelopt_fp4":
            quant_config = None

        self.config = config
        self.tp_size = get_tensor_model_parallel_world_size()
        self.quant_config = quant_config
        self.pp_group = get_pp_group()

        self.fc = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False)
        # Standard RMSNorm (x*w) to match transformers Qwen3RMSNorm — NOT GemmaRMSNorm.
        self.pre_fc_norm_embedding = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.pre_fc_norm_hidden = RMSNorm(config.hidden_size, config.rms_norm_eps)

        # 1-layer draft backbone. Do NOT set full_attention_interval (Qwen3.5-only).
        config.num_hidden_layers = 1
        self.model = Qwen3Model(
            config,
            quant_config=quant_config,
            prefix=add_prefix("model", prefix),
        )

        if get_pp_group().is_last_rank:
            if config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    quant_config=quant_config,
                    prefix=add_prefix("lm_head", prefix),
                )

        self.logits_processor = LogitsProcessor(config)


    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight

    def set_embed_and_head(self, embed, head):
        del self.model.embed_tokens.weight
        if not self.config.tie_word_embeddings:
            del self.lm_head.weight

        self.model.embed_tokens.weight = embed
        self.lm_head.weight = head
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        if input_embeds is None:
            input_embeds = self.model.embed_tokens(input_ids)

        hidden_states = forward_batch.spec_info.hidden_states

        if not forward_batch.forward_mode.is_idle():
            input_embeds = self.pre_fc_norm_embedding(input_embeds)
            hidden_states = self.pre_fc_norm_hidden(hidden_states)
        hidden_states = torch.cat([input_embeds, hidden_states], dim=-1)

        hidden_states = self.fc(hidden_states)

        # 4th positional is input_embeds: feeding the fused vector here bypasses
        # the inner embed lookup. Qwen3Model.forward returns hidden states.
        hidden_states = self.model(
            input_ids,
            positions,
            forward_batch,
            hidden_states,
        )

        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch
        )

    def load_weights(
        self, weights: Iterable[Tuple[str, torch.Tensor]], is_mtp: bool = False
    ):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        ignore_suffixes = (
            ".bias",
            "_bias",
            ".k_scale",
            "_k_scale",
            ".v_scale",
            "_v_scale",
            ".weight_scale",
            "_weight_scale",
            ".input_scale",
            "_input_scale",
        )

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue

            # Only process MTP-branch weights; embed/lm_head are shared from the
            # target via set_embed_and_head, the rest of the checkpoint backbone
            # is the (frozen) verify model and is not loaded into the draft.
            if "mtp" not in name:
                continue

            if name.startswith("mtp."):
                name = name.replace("mtp.", "model.")
                name = name.replace("model.fc", "fc")
                name = name.replace("model.pre_fc", "pre_fc")

            # NOTE: unlike qwen3_5_mtp we do NOT strip ".self_attn." — dense
            # Qwen3DecoderLayer keeps attention params under self_attn.*.

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name_mapped = name.replace(weight_name, param_name)
                if (
                    name_mapped.endswith(ignore_suffixes)
                    and name_mapped not in params_dict
                ):
                    continue
                if name_mapped not in params_dict:
                    continue
                param = params_dict[name_mapped]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight, shard_id)
                name = name_mapped
                break
            else:
                if name.endswith(ignore_suffixes) and name not in params_dict:
                    continue
                if name in params_dict:
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
                else:
                    logger.warning_once(
                        f"Parameter {name} not found in params_dict, skip loading"
                    )

            loaded_params.add(name)
        return loaded_params


EntryClass = [Qwen3ForCausalLMMTP]
