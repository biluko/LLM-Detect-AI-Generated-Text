#!pip install fastparquet
import transformers
import datasets
import pandas as pd
import numpy as np
from datasets import Dataset
import os

model_checkpoint = "D:/CodeProject/LLM-Detect-AI-Generated-Text/deberta-v3-small"  # base model

pile2 = pd.read_parquet('D:/CodeProject/LLM-Detect-AI-Generated-Text/plies-and-ultra/pile2.parquet', engine="fastparquet")
pile3 = pd.read_parquet('D:/CodeProject/LLM-Detect-AI-Generated-Text/plies-and-ultra/plies3.parquet', engine="fastparquet")
pile4 = pd.read_parquet('D:/CodeProject/LLM-Detect-AI-Generated-Text/plies-and-ultra/plies4.parquet', engine="fastparquet")
Ultra = pd.read_parquet('D:/CodeProject/LLM-Detect-AI-Generated-Text/plies-and-ultra/Ultra.parquet', engine="fastparquet")
lmsys = pd.read_parquet('D:/CodeProject/LLM-Detect-AI-Generated-Text/plies-and-ultra/lmsys.parquet', engine="fastparquet")
# https://www.kaggle.com/datasets/starblasters8/human-vs-llm-text-corpus?select=data.parquet
Human_LLM = pd.read_parquet('D:/CodeProject/LLM-Detect-AI-Generated-Text/Human vs. LLM Text Corpus/data.parquet', engine="fastparquet")
Human_LLM['label'] = np.where(Human_LLM['source'] == 'Human', 0, 1)
Human_LLM = Human_LLM[['text', 'label']]

valid = pd.read_csv(r'D:\CodeProject\LLM-Detect-AI-Generated-Text\code\nonTargetText_llm_slightly_modified_gen.csv')

train = pd.concat([pile2, pile3, pile4, Ultra, lmsys, Human_LLM], axis=0).sample(frac=1).reset_index(drop=True)

train.label.value_counts()

# train = train.head(10000)

train.text = train.text.fillna("")
valid.text = valid.text.apply(lambda x: x.strip('\n'))
train.text = train.text.apply(lambda x: x.strip('\n'))

ds_train = Dataset.from_pandas(train)
ds_valid = Dataset.from_pandas(valid)

from transformers import AutoTokenizer

# tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)


def preprocess_function(examples):
    return tokenizer(examples['text'], max_length=384, padding=True, truncation=True)


ds_train_enc = ds_train.map(preprocess_function, batched=True)

ds_valid_enc = ds_valid.map(preprocess_function, batched=True)

from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

num_labels = 2
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)

import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Move your model and data to the GPU
model.to(device);

from transformers import EarlyStoppingCallback

early_stopping = EarlyStoppingCallback(early_stopping_patience=20)

num_train_epochs = 60.0

metric_name = "roc_auc"
model_name = "deberta-v3-small_diverse"  # "deberta-large"
batch_size = 8

args = TrainingArguments(
    f"{model_name}-finetuned_v5",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=1e-4,
    lr_scheduler_type="cosine",
    fp16=True,
    optim="adamw_torch",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    gradient_accumulation_steps=1,
    num_train_epochs=num_train_epochs,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model=metric_name,
    report_to='none',
    save_total_limit=15,

)

from sklearn.metrics import roc_auc_score


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)
    auc = roc_auc_score(labels, probs[:, 1], multi_class='ovr')
    return {"roc_auc": auc}


trainer = Trainer(
    model,
    args,
    train_dataset=ds_train_enc,
    eval_dataset=ds_valid_enc,
    tokenizer=tokenizer,
    callbacks=[early_stopping],
    compute_metrics=compute_metrics
)

trainer.train()