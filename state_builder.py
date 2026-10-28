import numpy as np
import torch
import pickle
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
from sentence_transformers import SentenceTransformer
from generate_labels_risk2 import risk_cal
from train_risk import RiskNN

EMBEDDING_MODEL_PATH = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_SAVE_PATH = "./models/risk_nn_model.pt"
STATS_SAVE_PATH = "./models/normalization_stats.pkl"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

embed_tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_PATH)
embed_model = AutoModel.from_pretrained(EMBEDDING_MODEL_PATH).to(DEVICE)
embed_model.eval()

def get_embedding_model(text):
    inputs = embed_tokenizer(text, return_tensors="pt", truncation=True, max_length=256, padding=True).to(DEVICE)
    with torch.no_grad():
        outputs = embed_model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()
    return embeddings

def predict_emotions(text, model, tokenizer):
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        outputs = model(**inputs)
    
    probs = torch.sigmoid(outputs.logits)[0].cpu().numpy()
    emotions_list = ["sadness", "fear", "anger", "confusion", "joy", "neutral"]
    return {emotions_list[i]: float(probs[i]) for i in range(len(emotions_list))}


def get_embedding(text, encoder):
    embedding = encoder.encode(text)
    return embedding.tolist()


def predict_risk_with_uncertainty(model, text, structured_features, label_mean, label_std, num_passes=20):
    model.train()  # Active for MC Dropout analysis

    emb = get_embedding_model(text)
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
    # Load metadata configurations
    with open(STATS_SAVE_PATH, "rb") as f:
        meta_stats = pickle.load(f)
        
    # Construct model dynamically
    model = RiskNN(input_dim=meta_stats["input_dim"]).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
    
    return model, meta_stats["label_mean"], meta_stats["label_std"]



def build_state(text, models_dict, tokenizers_dict):
    emotions = predict_emotions(text, models_dict["emotion"], tokenizers_dict["emotion"])
    embedding = get_embedding(text, models_dict["encoder"])

    model, label_mean, label_std = init_engine()
    
    cal_outputs = risk_cal(text)
    structured_features = cal_outputs[1:]

    pred, uncert = predict_risk_with_uncertainty(model, text, structured_features, label_mean, label_std)
    risk_score = compute_risk(pred)
    level = risk_level(risk_score)
    
    state = {
        "emotion": emotions,
        "risk": round(risk_score, 4),
        "risk_vector": pred[0].tolist(),
        "risk_level": level,
        "risk_uncertainty": round(uncert.mean(), 4),
        "embedding": embedding
    }
    
    return state



if __name__ == "__main__":
    tokenizers = {
        "emotion": AutoTokenizer.from_pretrained("./models/emotion_model"),
        "risk": AutoTokenizer.from_pretrained("./models/risk_model")
    }
    
    models = {
        "emotion": AutoModelForSequenceClassification.from_pretrained("./models/emotion_model"),
        "risk": AutoModelForSequenceClassification.from_pretrained("./models/risk_model"),
        "encoder": SentenceTransformer("all-MiniLM-L6-v2")
    }

    text = "i am happy"
    # text = "Everything feels overwhelming and I don't know what to do."
    # text = "i want to die."
    
    user_state = build_state(text, models, tokenizers)
    
    print("\n--- STATE METRICS ---")
    print(f"Text: {text}")
    print("Emotion Score:", user_state["emotion"])
    print("Risk Level Score:", user_state["risk"])
    print("Model Confidence:", user_state["risk_level"])
    print("Model Uncertainty (Variance):", user_state["uncertainty"])
    # print("Embedding:", user_state["embedding"])