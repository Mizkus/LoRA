import os
import time
import torch
import subprocess
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from torch.utils.tensorboard import SummaryWriter
from peft import get_peft_model, LoraConfig, TaskType
import evaluate
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

glue_tasks = {
    "mnli":  ("premise", "hypothesis"),
    "sst2":  ("sentence", None),
    "mrpc":  ("sentence1", "sentence2"),
    "cola":  ("sentence", None),
    "qnli":  ("question", "sentence"),
    "qqp":   ("question1", "question2"),
    "rte":   ("sentence1", "sentence2"),
    "stsb":  ("sentence1", "sentence2")
}

def get_gpu_memory():
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"]
        )
        return int(output.decode().split("\n")[0])
    except:
        return -1

def preprocess_dataset(dataset, tokenizer, s1, s2):
    def tokenize_fn(examples):
        if s2:
            return tokenizer(
                examples[s1],
                examples[s2],
                truncation=True,
                padding="max_length",
                max_length=256
            )
        return tokenizer(
            examples[s1],
            truncation=True,
            padding="max_length",
            max_length=256
        )

    tokenized = dataset.map(tokenize_fn, batched=True)
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    return tokenized

def get_model(base_model_name, num_labels, use_lora):
    model = AutoModelForSequenceClassification.from_pretrained(base_model_name, num_labels=num_labels)
    if use_lora:
        config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=8,
            lora_alpha=8,
            lora_dropout=0.1,
            bias="none",
            target_modules=["query", "value"]
        )
        model = get_peft_model(model, config)
    return model

def compute_metrics(logits, labels, task):
    metric = evaluate.load("glue", task)
    if task == "stsb":
        preds = np.squeeze(logits)
    else:
        preds = np.argmax(logits, axis=1)
    return metric.compute(predictions=preds, references=labels)

def train(task, s1, s2, use_lora):
    mode = "lora" if use_lora else "full"
    run_name = f"{task}_{mode}"
    print(f"\nTraining: {run_name}")

    dataset = load_dataset("glue", task)
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")

    num_labels = 1 if task == "stsb" else dataset["train"].features["label"].num_classes
    tokenized = preprocess_dataset(dataset, tokenizer, s1, s2)

    model = get_model("roberta-base", num_labels, use_lora).to(device)

    log_dir = os.path.join("runs", task, mode)
    writer = SummaryWriter(log_dir)

    output_dir = f"models/{task}/{mode}"

    if task == "mnli":
        train_data = tokenized["train"]
        val_data = tokenized["validation_matched"]
    else:
        train_data = tokenized["train"]
        val_data = tokenized["validation"]

    metric_map = {
        "cola": "accuracy",
        "stsb": "pearson",
        "mrpc": "accuracy",
        "qqp": "accuracy",
        "mnli": "accuracy",
        "qnli": "accuracy",
        "rte": "accuracy",
        "sst2": "accuracy",
    }
    metric_for_best_model = metric_map[task]

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=5,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        logging_dir=log_dir,
        report_to="tensorboard",
        metric_for_best_model=metric_for_best_model,
        greater_is_better=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        tokenizer=tokenizer,
        compute_metrics=lambda p: compute_metrics(
            p.predictions,
            p.label_ids,
            task
        )
    )

    torch.cuda.empty_cache()
    mem_before = get_gpu_memory()
    start_time = time.time()

    trainer.train()

    torch.cuda.synchronize()
    end_time = time.time()
    mem_after = get_gpu_memory()
    max_mem = max(mem_before, mem_after) if mem_before > 0 and mem_after > 0 else -1
    duration = round(end_time - start_time, 2)

    predictions = trainer.predict(val_data)
    logits = predictions.predictions
    labels = predictions.label_ids
    metrics = compute_metrics(logits, labels, task)

    if max_mem > 0:
        writer.add_scalar("gpu/max_memory_MB", max_mem, 0)
    writer.add_scalar("train/time_sec", duration, 0)
    for key, value in metrics.items():
        writer.add_scalar(f"final_eval/{key}", value, 0)
    writer.close()

if __name__ == "__main__":
    for task, (s1, s2) in glue_tasks.items():
        train(task, s1, s2, use_lora=False)
        torch.cuda.empty_cache()
        train(task, s1, s2, use_lora=True)
        torch.cuda.empty_cache()
