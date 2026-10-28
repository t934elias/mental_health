from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)

MODEL_NAME = "roberta-base"

data_files = {
    "train": "https://raw.githubusercontent.com/viritaromero/Detecting-Depression-in-Tweets/master/sentiment_tweets3.csv"
}
dataset = load_dataset("csv", data_files=data_files)
dataset = dataset["train"].train_test_split(test_size=0.1, seed=42)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(example):
    return tokenizer(
        example["message"],
        truncation=True,
        padding="max_length",
        max_length=256
    )

dataset = dataset.map(tokenize)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
)

training_args = TrainingArguments(
    output_dir="./models/depression_model",
    eval_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"]
)

trainer.train()

model.save_pretrained("./models/depression_model")
tokenizer.save_pretrained("./models/depression_model")