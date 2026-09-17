"""Public launch commands using the native veRL and SGLang components."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config" / "growmtp"


def presets():
    import yaml

    return yaml.safe_load((CONFIG_DIR / "models.yaml").read_text())


def check(args=None):
    import importlib.metadata as metadata
    import torch
    import verl
    import sglang
    from verl.models.transformers.mtp import MTPHead
    from sglang.srt.server_args import ServerArgs

    assert "growmtp_collect_signals" in ServerArgs.__dataclass_fields__
    print("Local veRL and SGLang GrowMTP components import successfully.")
    for name in ("torch", "transformers", "tensordict", "ray", "sglang-kernel"):
        print(name, metadata.version(name))
    print("CUDA devices:", torch.cuda.device_count())


def prepare(args):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from verl.models.transformers.mtp import MTPHead
    from verl.workers.config.model import HFModelConfig  # registers native MiMo

    destination = Path(args.output)
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Model output must be an empty directory")
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=False)
    if config.model_type != "qwen3":
        raise ValueError("Only Qwen3 requires random-head preparation; native heads load directly")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.bfloat16, trust_remote_code=False
    )
    head = MTPHead(config)
    torch.manual_seed(0)
    for module in head.modules():
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=config.initializer_range)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif "RMSNorm" in type(module).__name__:
            torch.nn.init.ones_(module.weight)
    model.mtp = head.to(torch.bfloat16)
    model.config.architectures = ["Qwen3ForCausalLM"]
    model.save_pretrained(destination)
    AutoTokenizer.from_pretrained(args.model_path).save_pretrained(destination)
    print("Saved backbone and seeded MTP head in one checkpoint.")


def train_overrides(args):
    import yaml

    model = presets()[args.model]
    depth = args.depth or model["depth"]
    if depth not in model["supported_depths"]:
        raise ValueError(f"Paper depths for {args.model}: {model['supported_depths']}")
    values = yaml.safe_load((CONFIG_DIR / "training.yaml").read_text())
    values.update(
        {
            "data.train_files": [str(Path(args.train_file).resolve())],
            "data.val_files": [str(Path(args.val_file).resolve())],
            "data.prompt_key": args.prompt_key,
            "actor_rollout_ref.rollout.val_kwargs.n": 4 if args.task == "code" else 16,
            "reward.custom_reward_function.path": str(Path(__file__).with_name("rewards.py")),
            "data.max_response_length": args.response_length or model["response_length"],
            "++data.apply_chat_template_kwargs.enable_thinking": model["thinking"],
            "actor_rollout_ref.model.path": str(Path(args.model_path).resolve()),
            "actor_rollout_ref.model.mtp.speculative_num_steps": depth,
            "actor_rollout_ref.model.mtp.speculative_num_draft_tokens": depth + 1,
            "++actor_rollout_ref.model.mtp.learning_rate": model["head_lr"],
            "trainer.n_gpus_per_node": args.gpus,
            "trainer.nnodes": args.nodes,
            "trainer.total_training_steps": args.steps or model["steps"],
            "trainer.default_local_dir": str(Path(args.output).resolve()),
        }
    )
    return [f"{key}={json.dumps(value)}" for key, value in values.items()] + args.overrides


def train(args):
    overrides = train_overrides(args)
    command = [sys.executable, "-m", "verl.trainer.main_ppo", *overrides]
    if args.dry_run:
        import shlex

        print(shlex.join(command))
        return
    values = dict(item.split("=", 1) for item in overrides if "=" in item)
    checkpoint_marker = Path(args.output) / "latest_checkpointed_iteration.txt"
    if values.get("trainer.resume_mode") == '"auto"' and checkpoint_marker.is_file():
        finished = int(checkpoint_marker.read_text().strip())
        requested = int(values["trainer.total_training_steps"])
        if finished >= requested:
            if not (Path(args.output) / f"global_step_{finished}").is_dir():
                raise ValueError("Checkpoint marker refers to a missing checkpoint directory")
            print(f"Requested {requested} steps already completed (checkpoint {finished}).")
            return
    from transformers import AutoConfig
    from verl.workers.config.model import HFModelConfig

    model_config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=False)
    family = getattr(model_config, "text_config", model_config).model_type
    expected = presets()[args.model]["family"]
    if family not in (expected, expected + "_text"):
        raise ValueError(f"Checkpoint family {family!r} does not match preset {expected!r}")
    Path(args.output).mkdir(parents=True, exist_ok=True)
    subprocess.run(command, check=True, cwd=Path(args.output).resolve())


def infer(args):
    import sglang as sgl

    spec = presets()[args.model]
    depth = args.depth or spec["depth"]
    engine = sgl.Engine(
        model_path=args.model_path,
        dtype="bfloat16",
        tp_size=args.tp,
        speculative_algorithm="EAGLE",
        speculative_num_steps=depth,
        speculative_num_draft_tokens=depth + 1,
        speculative_eagle_topk=1,
        speculative_classic_rejection_sampling=True,
        disable_overlap_schedule=True,
        disable_radix_cache=True,
        mem_fraction_static=args.memory_fraction,
        max_running_requests=4,
        disable_cuda_graph=args.eager,
        disable_piecewise_cuda_graph=args.eager,
    )
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": args.prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=spec["thinking"],
        )
        result = engine.generate(
            prompt, {"temperature": 1.0, "top_p": 0.7, "max_new_tokens": args.max_new_tokens}
        )
        print(result["text"])
        meta = result["meta_info"]
        verifies = int(meta.get("spec_verify_ct", 0))
        accepts = int(meta.get("spec_num_correct_drafts", 0))
        print(
            json.dumps(
                {
                    "draft_depth": depth,
                    "verification_cycles": verifies,
                    "accepted_drafts": accepts,
                    "acceptance_length": 1 + accepts / verifies if verifies else None,
                }
            )
        )
    finally:
        engine.shutdown()




def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check")
    p = sub.add_parser("prepare")
    p.add_argument("--model-path", required=True)
    p.add_argument("--output", required=True)
    for command in ("train", "infer"):
        p = sub.add_parser(command)
        p.add_argument("--model", choices=list(presets()), required=True)
        p.add_argument("--model-path", required=True)
        p.add_argument("--depth", type=int)
        if command == "infer":
            p.add_argument("--tp", type=int, default=1)
            p.add_argument("--memory-fraction", type=float, default=0.7)
            p.add_argument("--eager", action="store_true")
            p.add_argument("--prompt", default="Compute 2+3.")
            p.add_argument("--max-new-tokens", type=int, default=16384)
            continue
        p.add_argument("--output", required=True)
        p.add_argument("--gpus", type=int, default=8)
        if command == "train":
            p.add_argument("--task", choices=["math", "code"], default="math")
            p.add_argument("--train-file", required=True)
            p.add_argument("--val-file", required=True)
            p.add_argument("--prompt-key", default="prompt")
            p.add_argument("--nodes", type=int, default=1)
            p.add_argument("--steps", type=int)
            p.add_argument("--response-length", type=int)
            p.add_argument("--dry-run", action="store_true")
            p.add_argument("overrides", nargs="*", help="Native Hydra overrides")
    args = parser.parse_args()
    globals()[args.command](args)


if __name__ == "__main__":
    main()
