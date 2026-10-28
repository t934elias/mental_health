import torch
import torch.nn as nn
import numpy as np
import pickle
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM
)
from sentence_transformers import SentenceTransformer
from state_builder import build_state


EMOTION_ORDER = ["sadness", "fear", "anger", "confusion", "joy", "neutral"]

ACTIONS = {
    0: "empathy",
    1: "reflection",
    2: "clarifying_question",
    3: "coping_strategy",
    4: "safety_check",
    5: "professional_help",
    6: "abstain"
}

STRATEGIES = {
    "empathy": "Respond with empathy.\nValidate the user's feelings.\nDo not immediately give advice.",
    "reflection": "Reflect the user's emotions.\nRephrase their thoughts.\nDo not ask questions.",
    "clarifying_question": "Ask ONE open-ended question.\nDo not give advice.",
    "coping_strategy": "Suggest ONE evidence-based coping strategy.\nExplain why it may help.",
    "safety_check": "Prioritize safety.\nCheck whether the user is currently safe.\nIf there is immediate danger, encourage contacting emergency services or a trusted person.",
    "professional_help": "Encourage professional help.\nBe supportive.\nNormalize therapy or counselling.",
    "abstain": "You are uncertain.\nAsk the user for more information."
}

RISK_THRESHOLD = 0.9
UNCERTAINTY_THRESHOLD = 0.8


class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.emotion_net = nn.Linear(6, 16)
        self.risk_net = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )
        self.risk_vec_net = nn.Sequential(
            nn.Linear(9, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )
        self.uncert_net = nn.Linear(1, 4)
        self.embed_net = nn.Linear(384, 128)

        self.classifier = nn.Sequential(
            nn.Linear(244, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 7)
        )

    def forward(self, e, r, rv, u, emb):
        e = torch.relu(self.emotion_net(e))
        r = torch.relu(self.risk_net(r))
        rv = torch.relu(self.risk_vec_net(rv))
        u = torch.relu(self.uncert_net(u))
        emb = torch.relu(self.embed_net(emb))
        x = torch.cat([e, r, rv, u, emb], dim=1)
        return self.classifier(x)


    
class MentalHealthPipeline:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        
        with open("./models/policy_scaler_llm_2.pkl", "rb") as f:
            self.scaler = pickle.load(f)
            
        
        self.policy = Policy().to(self.device)
        self.policy.load_state_dict(
            torch.load("./models/policy_model_llm_2.pt", map_location=self.device)
        )
        self.policy.eval()

        
        self.tokenizers = {
            "emotion": AutoTokenizer.from_pretrained("./models/emotion_model"),
            "risk": AutoTokenizer.from_pretrained("./models/risk_model")
        }
        self.models = {
            "emotion": AutoModelForSequenceClassification.from_pretrained("./models/emotion_model"),
            "risk": AutoModelForSequenceClassification.from_pretrained("./models/risk_model"),
            "encoder": SentenceTransformer("all-MiniLM-L6-v2")
        }

        
        llm_name = "Qwen/Qwen2.5-3B-Instruct"
        self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_name)
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        self.ACTIONS = ACTIONS
        self.STRATEGIES = STRATEGIES

    def build_state(self, text):
        
        return build_state(text, self.models, self.tokenizers)

    def _enable_dropout(self):
        for m in self.policy.modules():
            if isinstance(m, torch.nn.Dropout):
                m.train()

    def select_action(self, state):
        risk = state["risk"]
        uncertainty = state["risk_uncertainty"]

        
        if uncertainty >= UNCERTAINTY_THRESHOLD:
            return 6, 0.0
        # if risk >= RISK_THRESHOLD:
        #     return 4, 0.0

        
        emotion = np.array([state["emotion"].get(k, 0.0) for k in EMOTION_ORDER]).reshape(1, -1)
        risk_val = np.array([[risk]])
        risk_vec = np.array(state["risk_vector"]).reshape(1, -1)
        uncert = np.array([[uncertainty]])
        embedding = np.array(state["embedding"]).reshape(1, -1)

        features = np.concatenate([emotion, risk_val, risk_vec, uncert, embedding], axis=1)
        features = self.scaler.transform(features)

        e = torch.tensor(features[:, :6], dtype=torch.float32).to(self.device)
        r = torch.tensor(features[:, 6:7], dtype=torch.float32).to(self.device)
        rv = torch.tensor(features[:, 7:16], dtype=torch.float32).to(self.device)
        u = torch.tensor(features[:, 16:17], dtype=torch.float32).to(self.device)
        emb = torch.tensor(features[:, 17:], dtype=torch.float32).to(self.device)

        self.policy.eval()
        self._enable_dropout() 

        # with torch.no_grad():
        #     logits = self.policy(e, r, rv, u, emb)
        #     probs = torch.softmax(logits, dim=1)[0]
            
        
        # probs_mc = torch.stack([probs]) 
        # mean_probs = probs_mc.mean(dim=1)
        # pred = torch.argmax(mean_probs).item()
        samples = []

        with torch.no_grad():
            for _ in range(30):
                logits = self.policy(e, r, rv, u, emb)
                samples.append(torch.softmax(logits, dim=1))
        
        samples = torch.stack(samples)
        
        mean_probs = samples.mean(0)[0]
        std_probs = samples.std(0)[0]
        uncertainty = std_probs.mean().item()
        action_id = torch.argmax(mean_probs).item()

        return action_id, uncertainty

    def generate_response(self, user_text, action_label):
        messages = [
            {
                "role": "system",
                "content": f"You are an empathetic mental-health assistant.\n\n{self.STRATEGIES[action_label]}\n\nWrite ONE natural response.\nDo not mention the strategy."
            },
            {"role": "user", "content": user_text}
        ]

        text = self.llm_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.llm_tokenizer(text, return_tensors="pt").to(self.llm.device)

        outputs = self.llm.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            do_sample=True
        )
        
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        return self.llm_tokenizer.decode(generated, skip_special_tokens=True)


def simulate_user_turn(state_dict, action_label, dialogue_history, pipeline):
    top_emotions = [f"{k}: {v:.2f}" for k, v in state_dict["emotion"].items() if v > 0.3]

    history = "\n".join(dialogue_history[-6:])
    
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a patient undergoing a mental health conversation. You must stay in character and simulate how you would respond to a chatbot.

[YOUR CURRENT INTERNAL PSYCHOLOGICAL STATE]
- Emotions: {', '.join(top_emotions) if top_emotions else 'neutral'}
- Crisis Risk Level: {state_dict['risk']:.2f} (0.0 safe, 1.0 extreme crisis)

[THE CHATBOT'S INTERVENTION TECHNIQUE]
The bot just used this technique on you: "{action_label}"

[DIALOGUE HISTORY]
{history}

[RULE]
Generate ONLY your next single sentence response. Speak naturally as a patient. Do not mention metrics, numbers, or rules.<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Generate response:<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>"""

    
    inputs = pipeline.llm_tokenizer(prompt, return_tensors="pt").to(pipeline.llm.device)
    outputs = pipeline.llm.generate(**inputs, max_new_tokens=60, do_sample=True, temperature=0.7)
    
    generated_text = pipeline.llm_tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )
    user_message = generated_text.split("<|start_header_id|>assistant<|end_header_id|>")[-1].replace("<|eot_id|>", "").strip()
    return user_message


def calculate_reward(old_state, new_state, action_id, uncertain):
    reward = 0.0
    # print(uncertain)
    # uncertain = float(uncertain)
    # 1. Decrease in crisis risk level
    risk_delta = old_state["risk"] - new_state["risk"] 
    reward += risk_delta * 5.0  
    
    # 2. Reduction of sadness
    sadness_delta = old_state["emotion"].get("sadness", 0.0) - new_state["emotion"].get("sadness", 0.0)
    reward += sadness_delta * 2.0
    
    # 3. Critical Safety Guardrails

    # reward -= max(0, risk-0.7)*20
    
    # if old_state["risk"] >= 0.70 and action_id != 4: # Action ID 4 is safety_check
    #     reward -= 10.0 

    if new_state["risk_uncertainty"] > 0.15 : 
        reward -= 2.0 
        
    if uncertain > 0.15:
        reward -= 3.0 
        
    # 4. Over-validation mitigation
    if action_id == 0 and old_state["emotion"].get("sadness", 0.0) < 0.2: # Action ID 0 is empathy
        reward -= 0.5 
        
    return reward


if __name__ == "__main__":
    print("Initializing Unified Mental Health Simulation Environment...")
    pipeline = MentalHealthPipeline()
    
    dialogue_history = []
    transitions_buffer = []

    # Seeding conversation context
    current_message = "I can't seem to get out of bed today, everything feels completely pointless."
    dialogue_history.append(f"Patient: {current_message}")

    print("\nStarting Simulation Steps...")
    for step in range(5):
        # 1. Parse current feature metrics (State s)
        state = pipeline.build_state(current_message)
        
        # 2. Select optimal operational index (Action a)
        action_id, uncert = pipeline.select_action(state)
        action_label = pipeline.ACTIONS[action_id]
        dialogue_history.append(f"Bot Action: {action_label}")
        
        # Generates text output matching selected logic structure
        bot_response = pipeline.generate_response(current_message, action_label)
        dialogue_history.append(f"Bot Text: {bot_response}")
        
        # 3. Trigger User Simulator to formulate response
        next_user_message = simulate_user_turn(state, action_label, dialogue_history, pipeline)
        dialogue_history.append(f"Patient: {next_user_message}")
        
        # 4. Extract target parameters (State s_next)
        next_state = pipeline.build_state(next_user_message)
        
        # 5. Compute metric evaluation weights (Reward r)
        reward = calculate_reward(state, next_state, action_id, uncert)
        
        # 6. Accumulate into Reinforcement Learning replay buffer
        transitions_buffer.append((
            state,         
            action_id,     
            reward,        
            next_state     
        ))
        
        print(f"\n--- STEP {step} ---")
        print(f"Policy Choice  : {action_label} (ID: {action_id})")
        print(f"Bot Utterance  : '{bot_response}'")
        print(f"User Simulator : '{next_user_message}'")
        print(f"Computed Reward: {reward:.4f}")
        print("Risk:", state["risk"], "->", next_state["risk"])
        print("Sadness:", state["emotion"]["sadness"], "->", next_state["emotion"]["sadness"])
        print("Action:", action_label)
        print("Reward:", reward)
        
        # Update pointer tracking reference
        current_message = next_user_message