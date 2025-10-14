# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
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

# /// script
# dependencies = [
#     "trl",
#     "peft",
#     "Pillow>=9.4.0",
#     "torchvision",
#     "trackio",
#     "kernels",
# ]
# ///

"""
Without dataset streaming:

```
accelerate launch examples/scripts/dpo_vlm.py \
    --dataset_name HuggingFaceH4/rlaif-v_formatted \
    --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 32 \
    --dataset_num_proc 32 \
    --output_dir dpo_idefics_rlaif-v \
    --dtype bfloat16 \
    --gradient_checkpointing \
    --use_peft \
    --lora_target_modules all-linear
```

With dataset streaming:

```
accelerate launch examples/scripts/dpo_vlm.py \
    --dataset_name HuggingFaceH4/rlaif-v_formatted \
    --dataset_streaming \
    --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
    --per_device_train_batch_size 2 \
    --max_steps 100 \
    --gradient_accumulation_steps 32 \
    --dataset_num_proc 32 \
    --output_dir dpo_idefics_rlaif-v \
    --dtype bfloat16 \
    --gradient_checkpointing \
    --use_peft \
    --lora_target_modules all-linear
```
"""

import os
import sys
import datetime

# Add trl-fork to path if TRL_DIR is set
if "TRL_DIR" in os.environ:
    trl_dir = os.environ["TRL_DIR"]
    if trl_dir not in sys.path:
        sys.path.insert(0, trl_dir)
        print(f"✅ Using local TRL from: {trl_dir}")

import torch
from datasets import load_dataset
from transformers import AutoModelForImageTextToText, AutoProcessor
import wandb

from trl import (
    DPOConfig,
    DPOTrainer,
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from transformers import TrainerCallback


# Enable logging in a Hugging Face Space
os.environ.setdefault("TRACKIO_SPACE_ID", "trl-trackio")

# Set wandb to offline mode
os.environ["WANDB_MODE"] = "offline"
os.environ["WANDB_OFFLINE"] = "true"


class WandBLoggingCallback(TrainerCallback):
    """Custom callback to log detailed training metrics to wandb"""
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and state.is_world_process_zero:
            # Create structured logs for wandb
            wandb_logs = {}
            
            # Training metrics
            if 'loss' in logs:
                wandb_logs['train/loss'] = logs['loss']
            if 'grad_norm' in logs:
                wandb_logs['train/grad_norm'] = logs['grad_norm']
            if 'learning_rate' in logs:
                wandb_logs['train/learning_rate'] = logs['learning_rate']
            if 'epoch' in logs:
                wandb_logs['train/epoch'] = logs['epoch']
            
            # DPO-specific reward metrics
            if 'rewards/chosen' in logs:
                wandb_logs['dpo/rewards_chosen'] = logs['rewards/chosen']
            if 'rewards/rejected' in logs:
                wandb_logs['dpo/rewards_rejected'] = logs['rewards/rejected']
            if 'rewards/accuracies' in logs:
                wandb_logs['dpo/reward_accuracies'] = logs['rewards/accuracies']
            if 'rewards/margins' in logs:
                wandb_logs['dpo/reward_margins'] = logs['rewards/margins']

            # Humanline-specific metrics
            if 'rewards/chosen_unclamped' in logs:
                wandb_logs['humanline/chosen_unclamped'] = logs['rewards/chosen_unclamped']
            if 'rewards/rejected_unclamped' in logs:
                wandb_logs['humanline/rejected_unclamped'] = logs['rewards/rejected_unclamped']
            
            # Log probabilities
            if 'logps/chosen' in logs:
                wandb_logs['dpo/logps_chosen'] = logs['logps/chosen']
            if 'logps/rejected' in logs:
                wandb_logs['dpo/logps_rejected'] = logs['logps/rejected']
            
            # Logits
            if 'logits/chosen' in logs:
                wandb_logs['dpo/logits_chosen'] = logs['logits/chosen']
            if 'logits/rejected' in logs:
                wandb_logs['dpo/logits_rejected'] = logs['logits/rejected']
            
            # Evaluation metrics
            for key in logs.keys():
                if key.startswith('eval_'):
                    wandb_logs[f'eval/{key[5:]}'] = logs[key]
            
            # Log training step
            wandb_logs['train/global_step'] = state.global_step
            
            # Log to wandb
            if wandb_logs:
                wandb.log(wandb_logs, step=state.global_step)

if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, DPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    # Initialize wandb with comprehensive config logging
    run_name = f"{training_args.output_dir}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    wandb.init(
        project="vlm-dpo",
        name=run_name,
        mode="offline",
        config={
            # Model config
            "model_name": model_args.model_name_or_path,
            # "dtype": str(model_args.dtype),
            "use_peft": model_args.use_peft,
            "lora_target_modules": model_args.lora_target_modules if model_args.use_peft else None,
            "lora_r": model_args.lora_r if model_args.use_peft else None,
            "lora_alpha": model_args.lora_alpha if model_args.use_peft else None,

            # Training config
            "learning_rate": training_args.learning_rate,
            "per_device_train_batch_size": training_args.per_device_train_batch_size,
            "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
            "max_steps": training_args.max_steps,
            "num_train_epochs": training_args.num_train_epochs,
            "warmup_steps": training_args.warmup_steps,
            "logging_steps": training_args.logging_steps,
            "save_steps": training_args.save_steps,
            "eval_steps": training_args.eval_steps,
            "gradient_checkpointing": training_args.gradient_checkpointing,

            # DPO specific
            "beta": training_args.beta,
            "loss_type": training_args.loss_type,

            # Humanline specific
            "humanline": training_args.humanline,
            "log_epsilon_P": training_args.log_epsilon_P,
            "log_epsilon_R": training_args.log_epsilon_R,
        },
        tags=["dpo", "vlm", "vision-language", model_args.model_name_or_path.split("/")[-1]]
    )
    
    print(f"🚀 Starting DPO training run: {run_name}")
    print(f"📊 WandB logging in offline mode")

    if training_args.humanline:
        print(f"🎯 Humanline enabled with log_epsilon_P={training_args.log_epsilon_P}, log_epsilon_R={training_args.log_epsilon_R}")

    ################
    # Model & Tokenizer
    ################
    # dtype = model_args.dtype if model_args.dtype in ["auto", None] else getattr(torch, model_args.dtype)

    model_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        # dtype=dtype,
    )
    quantization_config = get_quantization_config(model_args)
    if quantization_config is not None:
        # Passing None would not be treated the same as omitting the argument, so we include it only when valid.
        model_kwargs["device_map"] = get_kbit_device_map()
        model_kwargs["quantization_config"] = quantization_config

    print(f"🔧 Loading model: {model_args.model_name_or_path}")
    model = AutoModelForImageTextToText.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        **model_kwargs,
    )
    
    # Log model info
    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    wandb.log({
        "model/total_parameters": num_params,
        "model/trainable_parameters": trainable_params,
        "model/trainable_percentage": (trainable_params / num_params) * 100
    })
    
    print(f"📏 Model size: {num_params:,} total params, {trainable_params:,} trainable ({(trainable_params/num_params)*100:.2f}%)")
    
    peft_config = get_peft_config(model_args)
    if peft_config is None:
        print("🔄 Loading reference model...")
        ref_model = AutoModelForImageTextToText.from_pretrained(
            model_args.model_name_or_path,
            trust_remote_code=model_args.trust_remote_code,
            **model_kwargs,
        )
    else:
        ref_model = None
        print("✅ Using PEFT - no reference model needed")
        
    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, do_image_splitting=False
    )
    tokenizer = processor.tokenizer

    # Set up the chat template
    if model.config.model_type == "idefics2":
        pass  # the processor already has a valid chat template
    elif model.config.model_type == "paligemma":
        processor.chat_template = """{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}<|im_start|>{% if message['role'] == 'user' %}USER: {% else %}ASSISTANT: {% endif %}{% for item in message['content'] if item['type'] == 'text' %}{{ item['text'] }}<|im_end|>{% endfor %}{% if message['role'] == 'user' %} {% else %}{{eos_token}}{% endif %}{% endfor %}{% if add_generation_prompt %}ASSISTANT: {% endif %}"""
    elif model.config.model_type == "llava":
        processor.chat_template = """{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{% if message['role'] == 'user' %}USER: {% else %}ASSISTANT: {% endif %}{% for item in message['content'] %}{% if item['type'] == 'text' %}{{ item['text'] }}{% elif item['type'] == 'image' %}<image>{% endif %}{% endfor %}{% if message['role'] == 'user' %} {% else %}{{eos_token}}{% endif %}{% endfor %}{% if add_generation_prompt %}ASSISTANT: {% endif %}"""

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if script_args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    ################
    # Dataset
    ################
    print(f"📚 Loading dataset: {script_args.dataset_name}")
    dataset = load_dataset(
        script_args.dataset_name,
        name=script_args.dataset_config,
        streaming=script_args.dataset_streaming,
    )
    
    # Log dataset info
    train_dataset = dataset[script_args.dataset_train_split]
    if not script_args.dataset_streaming:
        wandb.log({"dataset/train_size": len(train_dataset)})
        print(f"📊 Dataset size: {len(train_dataset):,} training examples")
    else:
        print("📊 Dataset: streaming mode enabled")

    ################
    # Training
    ################
    print("🏋️ Initializing DPO trainer...")
    trainer = DPOTrainer(
        model,
        ref_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        processing_class=processor,
        peft_config=peft_config,
        callbacks=[WandBLoggingCallback()],  # Add wandb logging callback
    )

    print("🚀 Starting training...")
    trainer.train()

    # Log final metrics
    print("✅ Training completed!")
    wandb.log({"training/status": "completed"})

    # Save and push to hub
    print(f"💾 Saving model to {training_args.output_dir}")
    trainer.save_model(training_args.output_dir)
    
    if training_args.push_to_hub:
        print("🤗 Pushing to Hugging Face Hub...")
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
        wandb.log({"model/pushed_to_hub": True})
    
    # Finish wandb run
    wandb.finish()
    print("🎉 Run completed successfully!")
    print(f"📊 WandB logs saved locally. Run 'wandb sync' to upload when online.")