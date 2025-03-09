import torch
from datasets import load_dataset, Dataset
from trl import GRPOTrainer, GRPOConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainerCallback
from peft import LoraConfig
from rewards import json_consistency_reward, f1_entities_reward, format_reward

dataset_name = '/home/congtri/dev/rabiloo/finetuning_llm/data/passport_en_grpo.jsonl'

def load_training_data(dataset_name) -> Dataset:
    data = load_dataset(path='json', data_files=dataset_name, split='train')
    dataset = data.map(lambda x: {
        'prompt': [
            {'role': x['conversations'][0]['role'],
             'content': x['conversations'][0]['content']},
        ],
        'answer': x['conversations'][1]['content'].strip()
    }
                       )
    return dataset.select_columns(['prompt', 'answer'])

dataset = load_training_data(dataset_name)

model_name = "Qwen/Qwen2-0.5B-Instruct"
output_dir = "outputs/Qwen2-0.5B-GRPO"

training_args = GRPOConfig(
    output_dir=output_dir,
    learning_rate=0.0001,
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,
    warmup_steps=100,
    lr_scheduler_type='cosine_with_restarts',
    logging_steps=1,
    bf16=True,
    # fp16=True,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=1,
    num_generations=2,  #4,
    max_prompt_length=612,
    max_completion_length=1024,
    num_train_epochs=1,
    save_steps=100,
    log_on_each_node=False,
    # use_vllm=True,
    # vllm_gpu_memory_utilization=0.6,
    # vllm_device="cuda:0",
    report_to="none"
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map=None,
).to("cuda")

tokenizer = AutoTokenizer.from_pretrained(model_name)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token
if tokenizer.padding_side is None:
    tokenizer.padding_side = 'right'
if model.config.pad_token_id is None:
    model.config.pad_token_id = model.config.eos_token_id

peft_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules="all-linear",
    task_type="CAUSAL_LM",
    lora_dropout=0.1,
)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        json_consistency_reward,
        f1_entities_reward,
        format_reward
    ],
    args=training_args,
    train_dataset=dataset,
    peft_config=peft_config,
)


class AdjustContextLengthCallback(TrainerCallback):
    """Dynamically increases max_completion_length during training."""

    def on_step_begin(self, args, state, control, **kwargs):
        """Adjusts max_completion_length based on training progress."""
        step = state.global_step

        if step >= 1000:
            args.max_prompt_length = 8192  # Allow longer completions
        elif step >= 500:
            args.max_completion_length = 8192  # Gradually increase

        # Log changes
        if step in [500, 1000]:
            print(f"Adjusted max_completion_length to {args.max_completion_length} at step {step}")


# Add dynamic context adjustment
trainer.add_callback(AdjustContextLengthCallback())

trainer.train()