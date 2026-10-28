# import torch
# import torch.nn as nn
# import numpy as np
# from state_builder import build_state
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# from sentence_transformers import SentenceTransformer
# import re
# import pickle

# with open("./models/policy_scaler.pkl", "rb") as f:
#     scaler = pickle.load(f)

# EMOTION_ORDER = [
#     "sadness",
#     "fear",
#     "anger",
#     "confusion",
#     "joy",
#     "neutral"
# ]

# class Policy(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(414, 256),
#             nn.ReLU(),
#             nn.Linear(256, 128),
#             nn.ReLU(),
#             nn.Linear(128, 7)
#         )

#     def forward(self, x):
#         return self.net(x)

# ACTIONS = {
#     0: "empathy",
#     1: "reflection",
#     2: "clarifying_question",
#     3: "coping_strategy",
#     4: "safety_check",    
#     5: "professional_help",
#     6: "abstain"
# }

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MODEL_PATH = "./models/policy_model.pt"
# model = Policy().to(device)

# try:
#     model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
#     model.eval() 
#     print(f"Successfully loaded trained weights from {MODEL_PATH}\n")
# except FileNotFoundError:
#     print(f"Error: Could not find trained weights at {MODEL_PATH}. Run training first!")
#     exit()


# tokenizers = {
#         "emotion": AutoTokenizer.from_pretrained("./models/emotion_model"),
#         "risk": AutoTokenizer.from_pretrained("./models/risk_model")
#     }
    
# models = {
#         "emotion": AutoModelForSequenceClassification.from_pretrained("./models/emotion_model"),
#         "risk": AutoModelForSequenceClassification.from_pretrained("./models/risk_model"),
#         "encoder": SentenceTransformer("all-MiniLM-L6-v2")
#     }

# # text = "Everything is hopeless and I have a bottle of pills right next to me."

# # user_state = build_state(text, models, tokenizers)

# # flat_vector = []

# # if 'emotion' in user_state and isinstance(user_state['emotion'], dict):
# #     flat_vector.extend([float(v) for v in user_state['emotion'].values()])
# # else:
# #     flat_vector.extend([0.0] * 6)

# # if 'risk' in user_state:
# #     if isinstance(user_state['risk'], dict):
# #         flat_vector.extend([float(v) for v in user_state['risk'].values()])
# #     else:
# #         flat_vector.append(float(user_state['risk']))
# # else:
# #     flat_vector.append(0.0)
        
# # if 'risk_uncertainty' in user_state:
# #     flat_vector.append(float(user_state['risk_uncertainty']))
# # else:
# #     flat_vector.append(0.0)

# # emb_key = 'embedding' if 'embedding' in user_state else 'embeddings'
# # if emb_key in user_state and user_state[emb_key] is not None:
# #     flat_vector.extend([float(v) for v in user_state[emb_key]])

# # if len(flat_vector) < 414:
# #     flat_vector += [0.0] * (414 - len(flat_vector))
# # elif len(flat_vector) > 414:
# #     flat_vector = flat_vector[:414]

# # state_tensor = torch.tensor(flat_vector, dtype=torch.float32).unsqueeze(0).to(device)

# # with torch.no_grad():
# #     logits = model(state_tensor)
# #     probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]
# #     predicted_id = torch.argmax(logits, dim=1).item()

# # print("\n" + text)
# # print(f"Predicted Action ID: {predicted_id}")
# # print(f"Policy Selection:   '{ACTIONS[predicted_id]}'")
# # print("Action Probabilities:")
# # for aid, label in ACTIONS.items():
# #     print(f"  - {label:<20}: {probabilities[aid]:.4f}")

    

# dataset = [
#         "i am happy", 
#         "i want to die", 
#         "why is life so hard", 
#         "I don't want to live anymore.", 
#         "I feel so lonely and isolated from everyone, nothing matters anymore."
#     ]

# for text in dataset:
#     user_state = build_state(text, models, tokenizers)
    
#     flat_vector = []
    
#     if 'emotion' in user_state and isinstance(user_state['emotion'], dict):
#         flat_vector.extend([
#             float(user_state["emotion"].get(k, 0.0))
#             for k in EMOTION_ORDER
#         ])
#     else:
#         flat_vector.extend([0.0] * 6)
    
#     if 'risk' in user_state:
#         if isinstance(user_state['risk'], dict):
#             flat_vector.extend([float(v) for v in user_state['risk'].values()])
#         else:
#             flat_vector.append(float(user_state['risk']))
#     else:
#         flat_vector.append(0.0)
            
#     if 'risk_uncertainty' in user_state:
#         flat_vector.append(float(user_state['risk_uncertainty']))
#     else:
#         flat_vector.append(0.0)
    
#     emb_key = 'embedding' if 'embedding' in user_state else 'embeddings'
#     if emb_key in user_state and user_state[emb_key] is not None:
#         flat_vector.extend([float(v) for v in user_state[emb_key]])
    
#     if len(flat_vector) < 414:
#         flat_vector += [0.0] * (414 - len(flat_vector))
#     elif len(flat_vector) > 414:
#         flat_vector = flat_vector[:414]

#     flat_vector = np.array(flat_vector).reshape(1, -1)
#     flat_vector = scaler.transform(flat_vector)
    
#     state_tensor = torch.tensor(flat_vector, dtype=torch.float32).to(device)

#     with torch.no_grad():
#         logits = model(state_tensor)
#         probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]
#         predicted_id = torch.argmax(logits, dim=1).item()
#         predicted_label = ACTIONS[predicted_id]

#     print("LOGITS:", logits)
#     print("MAX:", logits.max().item())
#     print("MIN:", logits.min().item())
        
#     print("\n" + text)
#     print(f"Risk: {user_state['risk']}")
#     print(f"Predicted Action ID: {predicted_id}")
#     print(f"Policy Selection:   '{ACTIONS[predicted_id]}'")
#     print("Action Probabilities:")
#     for aid, label in ACTIONS.items():
#         print(f"  - {label:<20}: {probabilities[aid]:.4f}")




import torch
import torch.nn as nn
import numpy as np
from state_builder import build_state
import pickle
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification


with open("./models/policy_scaler_llm_2.pkl", "rb") as f:
    scaler = pickle.load(f)

EMOTION_ORDER = ["sadness", "fear", "anger", "confusion", "joy", "neutral"]


# ----------------------------
# MUST MATCH TRAIN ARCHITECTURE
# ----------------------------
class Policy(nn.Module):
    def __init__(self):
        super().__init__()

        self.emotion_net = nn.Linear(6, 16)
        # self.risk_net = nn.Linear(1, 8)
        self.risk_net = nn.Sequential(
            nn.Linear(1,32),
            nn.ReLU(),
            nn.Linear(32,32)
        )
        # self.risk_vec_net = nn.Linear(9, 27)
        self.risk_vec_net = nn.Sequential(
            nn.Linear(9,64),
            nn.ReLU(),
            nn.Linear(64,64)
        )
        self.uncert_net = nn.Linear(1, 4)
        self.embed_net = nn.Linear(384, 128)

        self.classifier = nn.Sequential(
            nn.Linear(16 + 32 + 64 + 4 + 128,256),
            nn.ReLU(),
            nn.Dropout(0.3),
        
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Dropout(0.3),
        
            nn.Linear(128,64),
            nn.ReLU(),
        
            nn.Linear(64,7)
        )

    def forward(self, e, r, rv, u, emb):
        e = torch.relu(self.emotion_net(e))
        r = torch.relu(self.risk_net(r))
        rv = torch.relu(self.risk_vec_net(rv))
        u = torch.relu(self.uncert_net(u))
        emb = torch.relu(self.embed_net(emb))

        x = torch.cat([e, r, rv, u, emb], dim=1)
        return self.classifier(x)


# ----------------------------
# ACTION MAP
# ----------------------------
ACTIONS = {
    0: "empathy",
    1: "reflection",
    2: "clarifying_question",
    3: "coping_strategy",
    4: "safety_check",
    5: "professional_help",
    6: "abstain"
}


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Policy().to(device)
model.load_state_dict(torch.load("./models/policy_model_llm_2.pt", map_location=device))
model.eval()


# ----------------------------
# MODELS (STATE BUILDER)
# ----------------------------
tokenizers = {
    "emotion": AutoTokenizer.from_pretrained("./models/emotion_model"),
    "risk": AutoTokenizer.from_pretrained("./models/risk_model")
}

models = {
    "emotion": AutoModelForSequenceClassification.from_pretrained("./models/emotion_model"),
    "risk": AutoModelForSequenceClassification.from_pretrained("./models/risk_model"),
    "encoder": SentenceTransformer("all-MiniLM-L6-v2")
}

CSV_PATH = "./data/action_label_dataset_llm_2.csv"


# ----------------------------
# TEST DATA
# ----------------------------
dataset = [
    "i am happy",
    "i want to die",
    "why is life so hard",
    "I don't want to live anymore.",
    "I feel so lonely and isolated from everyone, nothing matters anymore."
]


# ----------------------------
# INFERENCE LOOP (CORRECT)
# ----------------------------
for text in dataset:

    user_state = build_state(text, models, tokenizers)
    
    emotion = np.array([
        float(user_state["emotion"].get(k, 0.0))
        for k in EMOTION_ORDER
    ]).reshape(1, -1)

    risk = np.array([[float(user_state.get("risk", 0.0))]])
    risk_vec = np.array(user_state["risk_vector"]).reshape(1, -1)
    uncert = np.array([[float(user_state.get("risk_uncertainty", 0.0))]])

    embedding = np.array(user_state["embedding"]).reshape(1, -1)

    # Build one feature vector
    features = np.concatenate([
        emotion,
        risk,
        risk_vec,
        uncert,
        embedding
    ], axis=1)

    # Scale
    features = scaler.transform(features)
    
    # Split back
    e_dim = emotion.shape[1]      # 6
    r_dim = risk.shape[1]         # 1
    rv_dim = risk_vec.shape[1]    # 9
    u_dim = uncert.shape[1]       # 1

    emotion = features[:, :e_dim]
    risk = features[:, e_dim:e_dim+r_dim]
    risk_vec = features[:, e_dim+r_dim:e_dim+r_dim+rv_dim]
    uncert = features[:, e_dim+r_dim+rv_dim:e_dim+r_dim+rv_dim+u_dim]
    embedding = features[:, e_dim+r_dim+rv_dim+u_dim:]

    # # ---- APPLY SAME SCALER ----
    # emotion = scaler.transform(emotion)
    # risk = scaler.transform(risk)
    # uncert = scaler.transform(uncert)
    # embedding = scaler.transform(embedding)

    # ---- convert to torch ----
    e = torch.tensor(emotion, dtype=torch.float32).to(device)
    r = torch.tensor(risk, dtype=torch.float32).to(device)
    rv = torch.tensor(risk_vec, dtype=torch.float32).to(device)
    u = torch.tensor(uncert, dtype=torch.float32).to(device)
    emb = torch.tensor(embedding, dtype=torch.float32).to(device)

    # ---- forward pass ----
    with torch.no_grad():
        logits = model(e, r, rv, u, emb)
        probs = torch.softmax(logits, dim=1)[0]

        pred = torch.argmax(probs).item()

    print("\n========================")
    print(text)
    print("Risk:", user_state["risk"])
    print("Prediction:", ACTIONS[pred])
    print("Probs:")

    for i, name in ACTIONS.items():
        print(f"{name:<25} {probs[i].item():.4f}")

    topk = torch.topk(probs, k=3)

    print("\nTop predictions:")
    for idx, p in zip(topk.indices.tolist(), topk.values.tolist()):
        print(f"{ACTIONS[idx]:<25} {p:.3f}")











# (user_env) [d3879@gpustrain_policy.py
# Epoch 1/30 | Loss: 1.1166 | Acc: 0.6035
# Epoch 2/30 | Loss: 0.8852 | Acc: 0.6760
# Epoch 3/30 | Loss: 0.7818 | Acc: 0.7073
# Epoch 4/30 | Loss: 0.6722 | Acc: 0.7445
# Epoch 5/30 | Loss: 0.5747 | Acc: 0.7813
# Epoch 6/30 | Loss: 0.4746 | Acc: 0.8203
# Epoch 7/30 | Loss: 0.3831 | Acc: 0.8543
# Epoch 8/30 | Loss: 0.3056 | Acc: 0.8853
# Epoch 9/30 | Loss: 0.2417 | Acc: 0.9117
# Epoch 10/30 | Loss: 0.1802 | Acc: 0.9331
# Epoch 11/30 | Loss: 0.1658 | Acc: 0.9368
# Epoch 12/30 | Loss: 0.1190 | Acc: 0.9574
# Epoch 13/30 | Loss: 0.1065 | Acc: 0.9634
# Epoch 14/30 | Loss: 0.0925 | Acc: 0.9666
# Epoch 15/30 | Loss: 0.1048 | Acc: 0.9635
# Epoch 16/30 | Loss: 0.0917 | Acc: 0.9676
# Epoch 17/30 | Loss: 0.0881 | Acc: 0.9675
# Epoch 18/30 | Loss: 0.0617 | Acc: 0.9793
# Epoch 19/30 | Loss: 0.0634 | Acc: 0.9799
# Epoch 20/30 | Loss: 0.0688 | Acc: 0.9754
# Epoch 21/30 | Loss: 0.0843 | Acc: 0.9703
# Epoch 22/30 | Loss: 0.0600 | Acc: 0.9807
# Epoch 23/30 | Loss: 0.0610 | Acc: 0.9783
# Epoch 24/30 | Loss: 0.0544 | Acc: 0.9807
# Epoch 25/30 | Loss: 0.0513 | Acc: 0.9815
# Epoch 26/30 | Loss: 0.0511 | Acc: 0.9841
# Epoch 27/30 | Loss: 0.0488 | Acc: 0.9815
# Epoch 28/30 | Loss: 0.0632 | Acc: 0.9787
# Epoch 29/30 | Loss: 0.0705 | Acc: 0.9754
# Epoch 30/30 | Loss: 0.0439 | Acc: 0.9853
# Model saved.
# (user_env) [d3879@gpus-hultig code]$ python policy_model.py

# ========================
# i am happy
# Risk: 0.205
# Prediction: clarifying_question
# Probs:
# empathy                   0.0540
# reflection                0.0215
# clarifying_question       0.3408
# coping_strategy           0.2171
# safety_check              0.2412
# professional_help         0.0490
# abstain                   0.0764

# ========================
# i want to die
# Risk: 1.0
# Prediction: coping_strategy
# Probs:
# empathy                   0.0127
# reflection                0.0137
# clarifying_question       0.2577
# coping_strategy           0.3772
# safety_check              0.2622
# professional_help         0.0592
# abstain                   0.0172

# ========================
# why is life so hard
# Risk: 0.6812
# Prediction: clarifying_question
# Probs:
# empathy                   0.0397
# reflection                0.0116
# clarifying_question       0.3651
# coping_strategy           0.2361
# safety_check              0.2505
# professional_help         0.0529
# abstain                   0.0442

# ========================
# I don't want to live anymore.
# Risk: 0.6785
# Prediction: clarifying_question
# Probs:
# empathy                   0.0301
# reflection                0.0123
# clarifying_question       0.4266
# coping_strategy           0.1027
# safety_check              0.3418
# professional_help         0.0469
# abstain                   0.0396

# ========================
# I feel so lonely and isolated from everyone, nothing matters anymore.
# Risk: 0.8699
# Prediction: clarifying_question
# Probs:
# empathy                   0.0277
# reflection                0.0120
# clarifying_question       0.4025
# coping_strategy           0.0932
# safety_check              0.3809
# professional_help         0.0434
# abstain                   0.0403
