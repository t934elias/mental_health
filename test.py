# import pandas as pd

# df = pd.read_csv(
#     "./data/suicide_risk_lexicon_validated_24-08-02T21-27-35.csv"
# )


# df = df[4:]
# df.dropna(how='all', axis=1, inplace=True)

# hopeless = df['Hopelessness'].str.lower()
# hopeless.dropna(inplace=True)
# HOPELESS = (
#     hopeless
#       .to_dict()
# )

# # dict = list(HOPELESS.values())
# # print(dict)


# passive = df['Passive suicidal ideation'].str.lower()
# passive.dropna(inplace=True)
# PASSIVE = (
#     passive
#       .to_dict()
# )

# active = df['Active suicidal ideation & suicidal planning'].str.lower()
# active.dropna(inplace=True)
# ACTIVE = (
#     active
#       .to_dict()
# )

# loneliness = df['Social withdrawal'].str.lower()
# loneliness.dropna(inplace=True)
# LONELINESS = (
#     loneliness
#       .to_dict()
# )



# https://github.com/swcwang/depression-detection/blob/master/data/tweets_final_1_clean.csv
# https://github.com/danielmlow/construct-tracker/blob/main/src/construct_tracker/data/lexicons/suicide_risk_lexicon_v1-0/suicide_risk_lexicon_validated_24-08-02T21-27-35.csv




















#________________________________________________________________________________________________________










# import pandas as pd
# import torch

# from datasets import load_dataset
# from transformers import (
#     AutoTokenizer,
#     AutoModelForSequenceClassification,
#     pipeline
# )

# DEPRESSION_MODEL_PATH = "./models/depression_model"
# EMOTION_MODEL_PATH = "./models/emotion_model"

# dep_tokenizer = AutoTokenizer.from_pretrained(DEPRESSION_MODEL_PATH)
# dep_model = AutoModelForSequenceClassification.from_pretrained(DEPRESSION_MODEL_PATH)
# dep_model.eval()


# emo_tokenizer = AutoTokenizer.from_pretrained(EMOTION_MODEL_PATH)
# emo_model = AutoModelForSequenceClassification.from_pretrained(EMOTION_MODEL_PATH)
# emo_model.eval()

# EMOTIONS_6 = ["sadness", "fear", "anger", "confusion", "joy", "neutral"]


# # Suicide Lexicon
# df_lexicon = pd.read_csv(
#     "./data/suicide_risk_lexicon_validated_24-08-02T21-27-35.csv"
# )

# df_lexicon = df_lexicon[4:]
# df_lexicon.dropna(how='all', axis=1, inplace=True)

# hopeless = df_lexicon['Hopelessness'].str.lower()
# hopeless.dropna(inplace=True)
# HOPELESS = (
#     hopeless
#       .to_dict()
# )
# HOPELESS = list(HOPELESS.values())

# passive = df_lexicon['Passive suicidal ideation'].str.lower()
# passive.dropna(inplace=True)
# PASSIVE = (
#     passive
#       .to_dict()
# )
# PASSIVE = list(PASSIVE.values())

# active = df_lexicon['Active suicidal ideation & suicidal planning'].str.lower()
# active.dropna(inplace=True)
# ACTIVE = (
#     active
#       .to_dict()
# )
# ACTIVE = list(ACTIVE.values())

# loneliness = df_lexicon['Social withdrawal'].str.lower()
# loneliness.dropna(inplace=True)
# LONELINESS = (
#     loneliness
#       .to_dict()
# )
# LONELINESS = list(LONELINESS.values())


# # Dataset
# # data_files = {
# #     "train": "https://raw.githubusercontent.com/viritaromero/Detecting-Depression-in-Tweets/master/sentiment_tweets3.csv"
# # }
# # dataset = load_dataset("csv", data_files=data_files, split="train")

# dataset = ["i am happy", "i want to die", "why is life so hard"]


# rows = []

# for sample in dataset:

#     # text = sample["message"]
#     text = sample
#     if not text or not isinstance(text, str):
#         continue

#     # Depression Probability
#     dep_inputs = dep_tokenizer(
#         text,
#         return_tensors="pt",
#         truncation=True,
#         max_length=256
#     )

#     with torch.no_grad():
#         dep_outputs = dep_model(**dep_inputs)

#     dep_probs = torch.softmax(dep_outputs.logits, dim=1)
#     depression_prob = dep_probs[0][1].item()


#     # Emotion Scores
#     emo_inputs = emo_tokenizer(
#         text,
#         return_tensors="pt",
#         truncation=True,
#         max_length=128
#     )

#     with torch.no_grad():
#         emo_outputs = emo_model(**emo_inputs)

#     emo_probs = torch.sigmoid(emo_outputs.logits)[0]
    
#     emotion_dict = {
#         EMOTIONS_6[i]: float(emo_probs[i])
#         for i in range(len(EMOTIONS_6))
#     }

#     sadness = emotion_dict.get("sadness", 0.0)
#     joy = emotion_dict.get("joy", 0.0)
#     fear = emotion_dict.get("fear", 0)
#     anger = emotion_dict.get("anger", 0)
#     confusion = emotion_dict.get("confusion", 0)

#     # Lexicon Scores
#     lower = text.lower()

#     active_nb = sum(
#         term in lower
#         for term in ACTIVE
#     )

#     passive_nb = sum(
#         term in lower
#         for term in PASSIVE
#     )

#     loneliness_nb = sum(
#         term in lower
#         for term in LONELINESS
#     )

#     hopelessness_nb = sum(
#         term in lower
#         for term in HOPELESS
#     )

#     # Risk Score
#     risk_score = (
#         1.0 * loneliness_nb +
#         2.5 * active_nb +
#         2.0 * passive_nb +
#         2.0 * hopelessness_nb +
#         1.0 * depression_prob +
#         0.5 * sadness +
#         0.15 * fear +
#         0.15 * anger +
#         0.2 * confusion -
#         0.5 * joy
#     )

#     # Risk Label
#     if risk_score < 1:
#         label = 0

#     elif risk_score < 3:
#         label = 1

#     elif risk_score < 5:
#         label = 2

#     else:
#         label = 3

#     rows.append({
#         "text": text,
#         "risk_label": label
#     })


# df = pd.DataFrame(rows)
# # df.to_csv(
# #     "./data/risk_dataset.csv",
# #     index=False
# # )
# # print(df["risk_label"].value_counts())

# print(df)


























#________________________________________________________________________________________________________





# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from transformers import AutoTokenizer, AutoModel
# from generate_labels_risk2 import risk_cal
# import math


# CSV_PATH = "./data/risk_dataset_regression.csv"
# EMBEDDING_MODEL_PATH = "sentence-transformers/all-MiniLM-L6-v2"  
# MODEL_SAVE_PATH = "./models/risk_nn_model.pt"
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# embed_tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_PATH)
# embed_model = AutoModel.from_pretrained(EMBEDDING_MODEL_PATH).to(DEVICE)
# embed_model.eval()

# def get_embedding(text):
#     inputs = embed_tokenizer(text, return_tensors="pt", truncation=True, max_length=256, padding=True).to(DEVICE)
#     with torch.no_grad():
#         outputs = embed_model(**inputs)
#         embeddings = outputs.last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()
#     return embeddings




# class RiskNN(nn.Module):
#     def __init__(self, input_dim):
#         super(RiskNN, self).__init__()
#         self.network = nn.Sequential(
#             nn.Linear(input_dim, 128),
#             nn.ReLU(),
#             nn.Dropout(p=0.4),
            
#             nn.Linear(128, 64),
#             nn.ReLU(),
#             nn.Dropout(p=0.3),
            
#             nn.Linear(64, 32),
#             nn.ReLU(),
#             nn.Dropout(p=0.2),

#             nn.Linear(32, 9)
#         )
#         criterion = nn.MSELoss()
        
#     def forward(self, x):
#         return self.network(x)




# class RiskRegressionDataset(Dataset):
#     def __init__(self, dataframe, is_training=True, min_score=None, max_score=None):

#         if is_training:
#             self.min_score = dataframe["risk_score"].min()
#             self.max_score = dataframe["risk_score"].max()
#         else:
#             self.min_score = min_score
#             self.max_score = max_score

#         self.label_cols = [
#             "active_nb",
#             "passive_nb",
#             "depression_prob",
#             "sadness",
#             "fear",
#             "joy",
#             "loneliness_nb",
#             "hopelessness_nb",
#             "anger"
#         ]


#         labels = dataframe[self.label_cols].values.astype(np.float32)
        
#         self.label_mean = labels.mean(axis=0)
#         self.label_std = labels.std(axis=0) + 1e-8
        
#         labels = (labels - self.label_mean) / self.label_std
#         self.labels = torch.tensor(labels, dtype=torch.float32)

        
        
#         feature_cols = self.label_cols
#         structured_features = dataframe[feature_cols].values
#         embeddings = np.stack([
#             get_embedding(t) for t in dataframe["text"].values
#         ])
        
#         self.features = torch.tensor(np.concatenate([embeddings, structured_features], axis=1), dtype=torch.float32)

#     def __len__(self):
#         return len(self.features)

#     def __getitem__(self, idx):
#         return self.features[idx], self.labels[idx]



    
# def predict_state_builder(model, text, structured_features, label_mean, label_std, num_passes=30):
#     model.train()

#     emb = get_embedding(text)
#     full_feature = np.concatenate([emb, structured_features])
#     x = torch.tensor(full_feature, dtype=torch.float32).unsqueeze(0).to(DEVICE)

#     preds = []
#     for _ in range(num_passes):
#         with torch.no_grad():
#             preds.append(model(x).cpu().numpy())

#     preds = np.array(preds)

#     mean = preds.mean(axis=0)
#     var = preds.var(axis=0)
    
#     var_original = var * (label_std ** 2)
#     mean_original = (mean * label_std) + label_mean

#     return mean_original, var_original




# def compute_risk(predic):
#     pred = predic[0]
#     active = pred[0]
#     passive = pred[1]
#     depression = pred[2]
#     sadness = pred[3]
#     fear = pred[4]
#     joy = pred[5]
#     loneliness = pred[6]
#     hopelessness = pred[7]
#     anger = pred[8]

#     risk = (
#         3.0 * active +
#         2.5 * passive +
#         2.0 * hopelessness +
#         1.5 * loneliness +
#         1.0 * depression +
#         0.5 * sadness +
#         0.3 * fear +
#         0.2 * anger -
#         1.5 * joy
#     )

#     return 1 / (1 + np.exp(-risk))

# def risk_level(prob):
#     if prob > 0.75:
#         return "HIGH RISK"
#     elif prob > 0.4:
#         return "MEDIUM RISK"
#     else:
#         return "LOW RISK"


# # ==========================================
# # 5. EXECUTION PIPELINE (TRAINING)
# # ==========================================
# if __name__ == "__main__":
#     df = pd.read_csv(CSV_PATH)
    
#     at_risk_df = df[df["risk_score"] > 0.05]
#     zero_risk_df = df[df["risk_score"] <= 0.05]
    
#     print(f"Original Data Breakdown -> At-Risk: {len(at_risk_df)}, Zero-Risk: {len(zero_risk_df)}")
    
#     zero_risk_downsampled = zero_risk_df.sample(n=min(len(at_risk_df) * 2, len(zero_risk_df)), random_state=42)
    
#     balanced_df = pd.concat([at_risk_df, zero_risk_downsampled]).reset_index(drop=True)
#     print(f"Balanced Dataset Size: {len(balanced_df)}")
    
#     train_df, val_df = train_test_split(balanced_df, test_size=0.15, random_state=42)
    
#     print("Processing Training Dataset...")
#     train_dataset = RiskRegressionDataset(train_df, is_training=True)
#     val_dataset = RiskRegressionDataset(
#         val_df, 
#         is_training=False, 
#         min_score=train_dataset.min_score, 
#         max_score=train_dataset.max_score
#     )
    
#     train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
#     # Initialize model dynamically using structural shape
#     input_dim = train_dataset.features.shape[1]
#     model = RiskNN(input_dim=input_dim).to(DEVICE)
    
#     criterion = nn.MSELoss()
#     optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
#     print("\nStarting Training Pipeline...")
#     for epoch in range(30):
#         model.train()
#         train_loss = 0
#         for X_b, y_b in train_loader:
#             X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
#             optimizer.zero_grad()
#             pred = model(X_b)
#             loss = criterion(pred, y_b)
#             loss.backward()
#             optimizer.step()
#             train_loss += loss.item()
            
#         # Validation Check
#         model.eval() # Standard eval context for val loss tracking
#         val_loss = 0
#         with torch.no_grad():
#             for X_v, y_v in val_loader:
#                 X_v, y_v = X_v.to(DEVICE), y_v.to(DEVICE)
#                 val_loss += criterion(model(X_v), y_v).item()
                
#         print(f"Epoch {epoch+1}/30 | Train MSE: {train_loss/len(train_loader):.4f} | Val MSE: {val_loss/len(val_loader):.4f}")
        
#     # Save parameters for deployment
#     torch.save(model.state_dict(), MODEL_SAVE_PATH)
#     print(f"\nModel saved successfully at {MODEL_SAVE_PATH}")
    
#     # --- TEST RUN THE STATE BUILDER PIPELINE ---
#     print("\nSimulating State Builder Engine Execution:")

#     dataset = ["i am happy", "i want to die", "why is life so hard", "I don't want to live anymore.", "I feel so lonely and isolated from everyone, nothing matters anymore."]

#     rows = []
    
#     for sample in dataset:
#         text = sample
#         if not text:
#             continue
    
#         cal_outputs = risk_cal(text)
#         structured_features = cal_outputs[1:]
    
#         pred, uncert = predict_state_builder(model, text, structured_features,train_dataset.label_mean, train_dataset.label_std)
#         # print(pred)
#         risk_score = compute_risk(pred)
    
#         print(
#             f"Risk score: {risk_score:.4f} | "
#             f"MC uncertainty: {uncert.mean():.6f}"
#         )


















import pandas as pd
df = pd.read_csv("./data/action_label_dataset.csv")
print(df['action_id'].value_counts())
