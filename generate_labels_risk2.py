# https://github.com/swcwang/depression-detection/blob/master/data/tweets_final_1_clean.csv
# https://github.com/danielmlow/construct-tracker/blob/main/src/construct_tracker/data/lexicons/suicide_risk_lexicon_v1-0/suicide_risk_lexicon_validated_24-08-02T21-27-35.csv




import pandas as pd
import torch

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline
)

DEPRESSION_MODEL_PATH = "./models/depression_model"
EMOTION_MODEL_PATH = "./models/emotion_model"

dep_tokenizer = AutoTokenizer.from_pretrained(DEPRESSION_MODEL_PATH)
dep_model = AutoModelForSequenceClassification.from_pretrained(DEPRESSION_MODEL_PATH)
dep_model.eval()


emo_tokenizer = AutoTokenizer.from_pretrained(EMOTION_MODEL_PATH)
emo_model = AutoModelForSequenceClassification.from_pretrained(EMOTION_MODEL_PATH)
emo_model.eval()

EMOTIONS_6 = ["sadness", "fear", "anger", "confusion", "joy", "neutral"]


# Suicide Lexicon
df_lexicon = pd.read_csv(
    "./data/suicide_risk_lexicon_validated_24-08-02T21-27-35.csv"
)

df_lexicon = df_lexicon[4:]
df_lexicon.dropna(how='all', axis=1, inplace=True)

hopeless = df_lexicon['Hopelessness'].str.lower()
hopeless.dropna(inplace=True)
HOPELESS = (
    hopeless
      .to_dict()
)
HOPELESS = list(HOPELESS.values())

passive = df_lexicon['Passive suicidal ideation'].str.lower()
passive.dropna(inplace=True)
PASSIVE = (
    passive
      .to_dict()
)
PASSIVE = list(PASSIVE.values())

active = df_lexicon['Active suicidal ideation & suicidal planning'].str.lower()
active.dropna(inplace=True)
ACTIVE = (
    active
      .to_dict()
)
ACTIVE = list(ACTIVE.values())

loneliness = df_lexicon['Social withdrawal'].str.lower()
loneliness.dropna(inplace=True)
LONELINESS = (
    loneliness
      .to_dict()
)
LONELINESS = list(LONELINESS.values())



def risk_cal(text):
    dep_inputs = dep_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=256
    )
    with torch.no_grad():
        dep_outputs = dep_model(**dep_inputs)
    
    dep_probs = torch.softmax(dep_outputs.logits, dim=1)
    depression_prob = dep_probs[0][1].item()

    # 2. Emotion Scores
    emo_inputs = emo_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128
    )

    with torch.no_grad():
        emo_outputs = emo_model(**emo_inputs)

    emo_probs = torch.sigmoid(emo_outputs.logits)[0]
    
    emotion_dict = {
        EMOTIONS_6[i]: float(emo_probs[i])
        for i in range(len(EMOTIONS_6))
    }
    
    sadness = emotion_dict.get("sadness", 0)
    fear = emotion_dict.get("fear", 0)
    joy = emotion_dict.get("joy", 0)
    anger = emotion_dict.get("anger", 0)
    confusion = emotion_dict.get("confusion", 0)

    # 3. Lexicon Scores (Normalized by word count to prevent length bias)
    lower = text.lower()
    word_count = max(len(lower.split()), 1) 
    
    active_nb = sum(term in lower for term in ACTIVE) / word_count * 10
    passive_nb = sum(term in lower for term in PASSIVE) / word_count * 10
    loneliness_nb = sum(term in lower for term in LONELINESS) / word_count * 10
    hopelessness_nb = sum(term in lower for term in HOPELESS) / word_count * 10

    # 4. Continuous Risk Score Calculation
    risk_score = (
        1.0 * loneliness_nb +
        2.5 * active_nb +
        2.0 * passive_nb +
        2.0 * hopelessness_nb +
        1.0 * depression_prob +
        0.5 * sadness +
        0.15 * fear +
        0.15 * anger +
        0.2 * confusion -
        0.5 * joy
    )
    return risk_score, active_nb, passive_nb, depression_prob, sadness, fear, joy, loneliness_nb, hopelessness_nb, anger


if __name__ == "__main__":
    data_files = {
    "train": "https://raw.githubusercontent.com/viritaromero/Detecting-Depression-in-Tweets/master/sentiment_tweets3.csv"
    }
    dataset = load_dataset("csv", data_files=data_files, split="train")

    # dataset = ["i am happy", "i want to die", "why is life so hard", "I don't want to live anymore."]

    rows = []

    for sample in dataset:

        text = sample["message"]
        if not text or not isinstance(text, str):
            continue

        risk_score, active_nb, passive_nb, depression_prob, sadness, fear, joy, loneliness_nb, hopelessness_nb, anger = risk_cal(text)

        # if risk_score < 1.0:
        #     label = 0
        # elif risk_score < 3.0:
        #     label = 1
        # elif risk_score < 5.0:
        #     label = 2
        # else:
        #     label = 3

        rows.append({
            "text": text, 
            "risk_score": float(risk_score), 
            "depression_prob": float(depression_prob),
            "sadness": float(sadness),
            "fear": float(fear),
            "joy":float(joy),
            "active_nb": float(active_nb),
            "passive_nb": float(passive_nb),
            "anger":float(anger),
            "loneliness_nb": float(loneliness_nb),
            "hopelessness_nb": float(hopelessness_nb),
        })

    df_out = pd.DataFrame(rows)
    df_out.to_csv("./data/risk_dataset_regression.csv", index=False)
    # print(df_out)

    print(df_out["risk_score"].describe())

