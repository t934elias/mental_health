import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = "./models/emotion_model"

tokenizer = AutoTokenizer.from_pretrained(
    "./models/emotion_model"
)

model = AutoModelForSequenceClassification.from_pretrained(
    "./models/emotion_model"
)

model.eval()

EMOTIONS = [
    "sadness",
    "fear",
    "anger",
    "confusion",
    "joy",
    "neutral"
]

# EMOTIONS = [
#     "admiration",
#     "amusement",
#     "anger",
#     "annoyance",
#     "approval",
#     "caring",
#     "confusion",
#     "curiosity",
#     "desire",
#     "disappointment",
#     "disapproval",
#     "disgust",
#     "embarrassment",
#     "excitement",
#     "fear",
#     "gratitude",
#     "grief",
#     "joy",
#     "love",
#     "nervousness",
#     "optimism",
#     "pride",
#     "realization",
#     "relief",
#     "remorse",
#     "sadness",
#     "surprise",
#     "neutral"
# ]

def predict_emotions(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128
    )

    with torch.no_grad():

        logits = model(**inputs).logits

        probs = torch.sigmoid(logits)[0]

    return {
        EMOTIONS[i]: float(probs[i])
        for i in range(len(EMOTIONS))
    }

if __name__ == "__main__":

    print(
        predict_emotions(
            "I don't know what to do. this is exhausting."
        )
    )