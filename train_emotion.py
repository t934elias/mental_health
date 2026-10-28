from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
import numpy as np
import torch
from transformers import default_data_collator

MODEL_NAME = "roberta-base"
TARGET_EMOTIONS = ["sadness", "fear", "anger", "confusion", "joy", "neutral"]
num_labels = len(TARGET_EMOTIONS)

GO_EMOTIONS_LIST = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring", 
    "confusion", "curiosity", "desire", "disappointment", "disapproval", 
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief", 
    "joy", "love", "nervousness", "optimism", "pride", "realization", 
    "relief", "remorse", "sadness", "surprise", "neutral"
]

ORIGINAL_TO_NEW_ID = {
    GO_EMOTIONS_LIST.index(emotion): new_id 
    for new_id, emotion in enumerate(TARGET_EMOTIONS)
}

dataset = load_dataset("go_emotions")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def multi_label_collator(features):
    batch = default_data_collator(features)
    batch["labels"] = batch["labels"].float() 
    return batch

def tokenize(batch):
    tokenized =  tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

    batch_size = len(batch["text"])
    multi_hot_labels = np.zeros((batch_size, num_labels), dtype=np.float32)

    for idx, original_label_list in enumerate(batch["labels"]):
        for orig_id in original_label_list:
            if orig_id in ORIGINAL_TO_NEW_ID:
                new_id = ORIGINAL_TO_NEW_ID[orig_id]
                multi_hot_labels[idx, new_id] = 1.0
        
        if np.sum(multi_hot_labels[idx]) == 0:
            multi_hot_labels[idx, ORIGINAL_TO_NEW_ID[27]] = 1.0

    tokenized["labels"] = multi_hot_labels

    return tokenized

dataset = dataset.map(tokenize, batched=True)

dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "labels"]
)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels,
    problem_type="multi_label_classification"
)

training_args = TrainingArguments(
    output_dir="./models/emotion_model",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=3,
    save_strategy="epoch"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    data_collator=multi_label_collator
)

trainer.train()

trainer.save_model("./models/emotion_model")
tokenizer.save_pretrained("./models/emotion_model")