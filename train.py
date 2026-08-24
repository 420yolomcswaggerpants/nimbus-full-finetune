from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
import torch

# Load dataset
dataset = load_dataset("json", data_files="dataset.jsonl", split="train")

# Load tokenizer and model
model_name = "Qwen/Qwen2.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Set pad token
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id

# Tokenize function
def tokenize_function(examples):
    texts = []
    for i in range(len(examples["prompt"])):
        text = f"### Instruction:\n{examples['prompt'][i]}\n\n### Response:\n{examples['completion'][i]}"
        texts.append(text)
    return tokenizer(texts, truncation=True, max_length=512, padding="max_length")

# Tokenize dataset
tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=25,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-5,
    lr_scheduler_type="linear",
    save_strategy="epoch",
    logging_steps=10,
    report_to="none"
)

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator
)

# Train
trainer.train()

# Save model
model.save_pretrained("./nimbus-full-finetune")
tokenizer.save_pretrained("./nimbus-full-finetune")