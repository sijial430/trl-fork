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

from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset
from huggingface_hub import ModelCard
from transformers import HfArgumentParser


@dataclass
class ScriptArguments:
    r"""
    Arguments for the script.

    Args:
        model_name (`str`, *optional*, defaults to `"gpt-3.5-turbo"`):
            Language model to target. Possible values are:
        aspect (`str`, *optional*, defaults to `"helpfulness"`):
            Aspect to target.
        push_to_hub (`bool`, *optional*, defaults to `False`):
            Whether to push the dataset to the Hugging Face Hub.
        repo_id (`str`, *optional*, defaults to `"trl-lib/ultrafeedback-gpt-3.5-turbo-helpfulness"`):
            Hugging Face repository ID to push the dataset to.
        dataset_num_proc (`int` or `None`, *optional*, defaults to `None`):
            Number of workers to use for dataset processing.
    """
    push_to_hub: bool = field(
        default=True,
        metadata={"help": "Whether to push the dataset to the Hugging Face Hub."},
    )
    repo_id: str = field(
        default="sijial430/llama3-ultrafeedback-armorm",
        metadata={"help": "Hugging Face repository ID to push the dataset to."},
    )
    dataset_num_proc: Optional[int] = field(
        default=8,
        metadata={"help": "Number of workers to use for dataset processing."},
    )


def to_unpaired_preference(examples):
    prompts, completions, labels = [], [], []
    for i in range(len(examples["prompt"])):
        prompt = [{"role": "user", "content": examples["prompt"][i]}]
        chosen_completion = {"role": "assistant", "content": examples["chosen"][i][-1]["content"]}
        rejected_completion = {"role": "assistant", "content": examples["rejected"][i][-1]["content"]}

        prompts.append(prompt)
        completions.append(chosen_completion)
        labels.append(True)

        prompts.append(prompt)
        completions.append(rejected_completion)
        labels.append(False)
    return {"prompt": prompts, "completion": completions, "label": labels}


model_card = ModelCard("""
---
tags: [trl]
---

# UltraFeedback Armorm Dataset

## Summary

The UltraFeedback Armorm dataset contains processed user-assistant interactions filtered for helpfulness, derived from the [princeton-nlp/llama3-ultrafeedback-armorm](https://huggingface.co/datasets/princeton-nlp/llama3-ultrafeedback-armorm) dataset. It is designed for fine-tuning and evaluating models in alignment tasks.

## Data Structure

- **Format**: [Conversational](https://huggingface.co/docs/trl/main/dataset_formats#conversational)
- **Type**: [Unpaired preference](https://huggingface.co/docs/trl/main/dataset_formats#unpaired-preference)

Column:
- `"prompt"`: The input question or instruction provided to the model.
- `"completion"`: The model's response to the prompt.
- `"label"`: A binary value indicating whether the response is sufficiently helpful.

## Generation script

The script used to generate this dataset can be found [here](https://github.com/huggingface/trl/blob/main/examples/datasets/ultrafeedback-armorm.py).
""")

if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    dataset = load_dataset("princeton-nlp/llama3-ultrafeedback-armorm", split="train")    
    dataset = dataset.map(
        to_unpaired_preference,
        remove_columns=["prompt_id", "prompt", "chosen", "rejected", "all_generated_responses", "all_rm_scores"],
        num_proc=script_args.dataset_num_proc,
        batched=True,
    )
    dataset = dataset.shuffle(seed=42)
    dataset = dataset.train_test_split(test_size=0.05, seed=42)

    if script_args.push_to_hub:
        dataset.push_to_hub(script_args.repo_id)
        model_card.push_to_hub(script_args.repo_id, repo_type="dataset")
        print(f"Dataset pushed to {script_args.repo_id}.")
