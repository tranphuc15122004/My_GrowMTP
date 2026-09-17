"""Recurrent draft layer; checkpoint names are shared with SGLang's MTP loaders."""

import copy
from pathlib import Path

import torch
from torch import nn


class MTPHead(nn.Module):
    def __init__(self, text_config):
        super().__init__()
        family = text_config.model_type
        if family == "mimo":
            from transformers.models.qwen2.modeling_qwen2 import (
                Qwen2DecoderLayer as Layer,
                Qwen2RMSNorm as Norm,
                Qwen2RotaryEmbedding as Rotary,
            )
        elif family == "qwen3":
            from transformers.models.qwen3.modeling_qwen3 import (
                Qwen3DecoderLayer as Layer,
                Qwen3RMSNorm as Norm,
                Qwen3RotaryEmbedding as Rotary,
            )
        elif family in ("qwen3_5", "qwen3_5_text"):
            from transformers.models.qwen3_5.modeling_qwen3_5 import (
                Qwen3_5DecoderLayer as Layer,
                Qwen3_5RMSNorm as Norm,
                Qwen3_5TextRotaryEmbedding as Rotary,
            )
        else:
            raise ValueError(f"GrowMTP does not support model_type={family!r}")
        config = copy.deepcopy(text_config)
        config.num_hidden_layers = 1
        if hasattr(config, "layer_types"):
            config.layer_types = ["full_attention"]
        config._attn_implementation = "sdpa"
        self.config = config
        h = config.hidden_size
        self.pre_fc_norm_embedding = Norm(h, eps=config.rms_norm_eps)
        self.pre_fc_norm_hidden = Norm(h, eps=config.rms_norm_eps)
        self.fc = nn.Linear(2 * h, h, bias=False)
        self.layers = nn.ModuleList([Layer(config, layer_idx=0)])
        self.norm = Norm(h, eps=config.rms_norm_eps)
        self.rotary = Rotary(config)

    def _apply(self, fn, recurse=True):
        """Keep RoPE frequencies in FP32 across parameter dtype/device conversions."""
        if not recurse:
            return super()._apply(fn, recurse=False)
        frequencies = {
            name: value for name, value in self.rotary.named_buffers(recurse=False)
            if name.endswith("inv_freq")
        }
        result = super()._apply(fn, recurse=True)
        device = self.rotary.inv_freq.device
        if any(value.is_meta for value in frequencies.values()) and device.type != "meta":
            # Materialize deterministic, non-persistent constants from the model config.
            self.rotary = type(self.rotary)(self.rotary.config, device=device).train(self.rotary.training)
        else:
            for name, value in frequencies.items():
                setattr(self.rotary, name, value.to(device=device, dtype=torch.float32))
        return result

    def forward(self, hidden, embeddings, positions, cache, attention_mask):
        if self.config.model_type == "mimo":
            embeddings = embeddings.masked_fill(positions.to(embeddings.device).eq(0).unsqueeze(-1), 0)
        fused = self.fc(
            torch.cat(
                (self.pre_fc_norm_embedding(embeddings), self.pre_fc_norm_hidden(hidden)), dim=-1
            )
        )
        cos_sin = self.rotary(fused, positions)
        length = cache.get_seq_length()
        output = self.layers[0](
            hidden_states=fused,
            position_embeddings=cos_sin,
            position_ids=positions,
            attention_mask=attention_mask,
            past_key_values=cache,
            cache_position=torch.arange(length, length + fused.shape[1], device=fused.device),
            use_cache=True,
        )
        hidden = output[0] if isinstance(output, tuple) else output
        return self.norm(hidden)


def shared_layers(model):
    if hasattr(model.model, "language_model"):
        return model.model.language_model.embed_tokens, model.lm_head
    return model.model.embed_tokens, model.lm_head


def attach_mtp(model, model_path):
    config = getattr(model.config, "text_config", model.config)
    head = MTPHead(config)
    load_mtp_weights(head, Path(model_path), config.model_type)
    model.mtp = head.to(dtype=next(model.parameters()).dtype)
    return model


def load_mtp_weights(head, model_path, family):
    """Load only draft weights, remapping native MiMo names into the common layout."""
    from safetensors import safe_open

    weights = {}
    mimo_names = {
        "token_layernorm": "pre_fc_norm_embedding",
        "hidden_layernorm": "pre_fc_norm_hidden",
        "input_proj": "fc",
        "final_layernorm": "norm",
    }
    for file in sorted(Path(model_path).glob("*.safetensors")):
        with safe_open(file, framework="pt", device="cpu") as tensors:
            for name in tensors.keys():
                if name.startswith("mtp."):
                    key = name[4:]
                elif family == "mimo" and name.startswith("model.mtp_layers.0."):
                    key = name[len("model.mtp_layers.0.") :]
                    prefix, _, suffix = key.partition(".")
                    key = (
                        mimo_names[prefix] + "." + suffix
                        if prefix in mimo_names
                        else "layers.0." + key
                    )
                else:
                    continue
                tensor = tensors.get_tensor(name)
                if family == "mimo" and name.endswith("mtp_layers.0.input_proj.weight"):
                    h = tensor.shape[0]
                    tensor = torch.cat([tensor[:, h:], tensor[:, :h]], dim=1).contiguous()
                weights[key] = tensor
    if not weights:
        raise ValueError("Checkpoint has no MTP weights. Prepare the Qwen3 head before training.")
    head.load_state_dict(weights, strict=True)
