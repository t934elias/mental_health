# from transformers import pipeline

# classifier = pipeline(
#     "text-classification",
#     model="./models/final_risk_model"
# )

# dataset = ["i am happy", "i want to die", "why is life so hard", "I don't want to live anymore."]

# rows = []

# for sample in dataset:

#     text = sample
#     if not text or not isinstance(text, str):
#         continue


#     result = classifier(text)

#     print(result)


import torch
import numpy as np
import pickle
from transformers import AutoTokenizer, AutoModel
from generate_labels_risk2 import risk_cal
from train_risk import RiskNN  


EMBEDDING_MODEL_PATH = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_SAVE_PATH = "./models/risk_nn_model.pt"
STATS_SAVE_PATH = "./models/normalization_stats.pkl"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

embed_tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_PATH)
embed_model = AutoModel.from_pretrained(EMBEDDING_MODEL_PATH).to(DEVICE)
embed_model.eval()

def get_embedding(text):
    inputs = embed_tokenizer(text, return_tensors="pt", truncation=True, max_length=256, padding=True).to(DEVICE)
    with torch.no_grad():
        outputs = embed_model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()
    return embeddings


def predict_state_builder(model, text, structured_features, label_mean, label_std, num_passes=30):
    model.train()  

    emb = get_embedding(text)
    full_feature = np.concatenate([emb, structured_features])
    x = torch.tensor(full_feature, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    preds = []
    for _ in range(num_passes):
        with torch.no_grad():
            preds.append(model(x).cpu().numpy())

    preds = np.array(preds)
    mean = preds.mean(axis=0)
    var = preds.var(axis=0)
    
    var_original = var * (label_std ** 2)
    mean_original = (mean * label_std) + label_mean

    return mean_original, var_original

def compute_risk(predic):
    pred = predic[0]
    risk = (
        3.0 * pred[0] +   # active
        2.5 * pred[1] +   # passive
        2.0 * pred[7] +   # hopelessness
        1.5 * pred[6] +   # loneliness
        1.0 * pred[2] +   # depression
        0.5 * pred[3] +   # sadness
        0.3 * pred[4] +   # fear
        0.2 * pred[8] -   # anger
        1.5 * pred[5]     # joy
    )
    return 1 / (1 + np.exp(-risk))

def risk_level(prob):
    if prob > 0.75:
        return "HIGH RISK"
    elif prob > 0.4:
        return "MEDIUM RISK"
    else:
        return "LOW RISK"


def init_engine():

    with open(STATS_SAVE_PATH, "rb") as f:
        meta_stats = pickle.load(f)
        
 
    model = RiskNN(input_dim=meta_stats["input_dim"]).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
    
    return model, meta_stats["label_mean"], meta_stats["label_std"]

if __name__ == "__main__":
    print("\nInitializing State Builder Engine...")
    model, label_mean, label_std = init_engine()

    dataset = [
        "i am happy", 
        "i want to die", 
        "why is life so hard", 
        "I don't want to live anymore.", 
        "I feel so lonely and isolated from everyone, nothing matters anymore."
    ]

    print("\nSimulating Live Production Inferences:")
    for text in dataset:
        if not text:
            continue
            
        cal_outputs = risk_cal(text)
        structured_features = cal_outputs[1:]
    
        pred, uncert = predict_state_builder(model, text, structured_features, label_mean, label_std)
        risk_score = compute_risk(pred)
        level = risk_level(risk_score)
    
        print(
            f"Text: '{text[:30]}...' | "
            f"Risk: {risk_score:.4f} ({level}) | "
            f"MC Uncertainty: {uncert.mean():.6f}"
        )