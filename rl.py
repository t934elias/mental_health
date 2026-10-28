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
from user_simulation import MentalHealthPipeline


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

RISK_THRESHOLD = 0.80
UNCERTAINTY_THRESHOLD = 0.8


def calculate_reward(old_state, new_state, action_id, uncertain):
    reward = 0.0
    # 1. Decrease in crisis risk level
    risk_delta = old_state["risk"] - new_state["risk"] 
    reward += risk_delta * 5.0  
    
    # 2. Reduction of sadness
    sadness_delta = old_state["emotion"].get("sadness", 0.0) - new_state["emotion"].get("sadness", 0.0)
    reward += sadness_delta * 2.0
    
    # 3. Critical Safety Guardrails
    if old_state["risk"] >= 0.70 and action_id != 4: # Action ID 4 is safety_check
        reward -= 10.0 

    if new_state["risk_uncertainty"] > 0.15 : 
        reward -= 2.0 
        
    if uncertain > 0.15:
        reward -= 3.0 
        
    # 4. Over-validation mitigation
    if action_id == 0 and old_state["emotion"].get("sadness", 0.0) < 0.2: # Action ID 0 is empathy
        reward -= 0.5 
        
    return reward

















def extract_tensors(batch, device, scaler):
    """
    Extracts and scales states from a transition batch.
    """
    emotions, risks, risk_vecs, uncerts, embeddings = [], [], [], [], []
    actions, rewards, next_emotions, next_risks, next_risk_vecs, next_uncerts, next_embeddings = [], [], [], [], [], [], []
    
    for state, action, reward, next_state in batch:
        # Scale current state features
        emo = [state["emotion"].get(k, 0.0) for k in EMOTION_ORDER]
        f = np.concatenate([
            np.array(emo), 
            [state["risk"]], 
            state["risk_vector"], 
            [state["risk_uncertainty"]], 
            state["embedding"]
        ]).reshape(1, -1)
        f_scaled = scaler.transform(f)[0]
        
        emotions.append(f_scaled[:6])
        risks.append([f_scaled[6]])
        risk_vecs.append(f_scaled[7:16])
        uncerts.append([f_scaled[16]])
        embeddings.append(f_scaled[17:])
        
        actions.append(action)
        rewards.append(reward)
        
        # Scale next state features
        next_emo = [next_state["emotion"].get(k, 0.0) for k in EMOTION_ORDER]
        fn = np.concatenate([
            np.array(next_emo), 
            [next_state["risk"]], 
            next_state["risk_vector"], 
            [next_state["risk_uncertainty"]], 
            next_state["embedding"]
        ]).reshape(1, -1)
        fn_scaled = scaler.transform(fn)[0]
        
        next_emotions.append(fn_scaled[:6])
        next_risks.append([fn_scaled[6]])
        next_risk_vecs.append(fn_scaled[7:16])
        next_uncerts.append([fn_scaled[16]])
        next_embeddings.append(fn_scaled[17:])

    # Move all components directly to device
    return (
        torch.tensor(emotions, dtype=torch.float32).to(device),
        torch.tensor(risks, dtype=torch.float32).to(device),
        torch.tensor(risk_vecs, dtype=torch.float32).to(device),
        torch.tensor(uncerts, dtype=torch.float32).to(device),
        torch.tensor(embeddings, dtype=torch.float32).to(device),
        torch.tensor(actions, dtype=torch.long).to(device),
        torch.tensor(rewards, dtype=torch.float32).to(device),
        torch.tensor(next_emotions, dtype=torch.float32).to(device),
        torch.tensor(next_risks, dtype=torch.float32).to(device),
        torch.tensor(next_risk_vecs, dtype=torch.float32).to(device),
        torch.tensor(next_uncerts, dtype=torch.float32).to(device),
        torch.tensor(next_embeddings, dtype=torch.float32).to(device),
    )







import copy
import random
import torch.optim as optim

# --- Hyperparameters ---
BATCH_SIZE = 16
GAMMA = 0.99          # Discount factor
LR = 1e-4             # Learning rate for policy weights
TARGET_UPDATE = 5     # Synchronize target network every X updates

# Initialize Optimization modules inside your pipeline initialization or main script:
device = pipeline.device
policy_net = pipeline.policy

# 1. Create target network for target stabilization
target_net = copy.deepcopy(policy_net).to(device)
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LR)
loss_fn = torch.nn.MSELoss()

def train_dqn_step(buffer, update_counter):
    if len(buffer) < BATCH_SIZE:
        return None # Not enough data points gathered yet
    
    # 2. Sample random mini-batch from Replay Buffer
    batch = random.sample(buffer, BATCH_SIZE)
    
    # Parse items to PyTorch
    (e, r, rv, u, emb, 
     actions, rewards, 
     next_e, next_r, next_rv, next_u, next_emb) = extract_tensors(batch, device, pipeline.scaler)
    
    # 3. Calculate Current Q-values (predicted actions using behavior policy)
    policy_net.train() # Set to train mode for optimization
    current_q_values = policy_net(e, r, rv, u, emb).gather(1, actions.unsqueeze(1)).squeeze(1)
    
    # 4. Calculate Expected Future Q-values using Target Network
    with torch.no_grad():
        max_next_q_values = target_net(next_e, next_r, next_rv, next_u, next_emb).max(1)[0]
        expected_q_values = rewards + (GAMMA * max_next_q_values)
        
    # 5. Compute loss and optimize policy weights
    loss = loss_fn(current_q_values, expected_q_values)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # 6. Periodic Target Update
    if update_counter % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())
        
    return loss.item()









if __name__ == "__main__":
    print("Initializing Unified Mental Health Simulation Training...")
    pipeline = MentalHealthPipeline()
    
    # Max size replay buffer 
    from collections import deque
    transitions_buffer = deque(maxlen=10000)
    
    update_counter = 0
    NUM_EPISODES = 50 
    STEPS_PER_EPISODE = 5

    for episode in range(NUM_EPISODES):
        # Seed conversation context fresh for every episode
        current_message = "I can't seem to get out of bed today, everything feels completely pointless."
        dialogue_history = [f"Patient: {current_message}"]
        
        print(f"\n=== STARTING EPISODE {episode} ===")
        
        for step in range(STEPS_PER_EPISODE):
            # 1. State Processing
            state = pipeline.build_state(current_message)
            
            # Epsilon-Greedy action exploration strategy can be injected here if needed
            action_id, uncert = pipeline.select_action(state)
            action_label = pipeline.ACTIONS[action_id]
            
            # 2. Environment Interaction & Execution
            bot_response = pipeline.generate_response(current_message, action_label)
            dialogue_history.append(f"Bot Text: {bot_response}")
            
            next_user_message = simulate_user_turn(state, action_label, dialogue_history, pipeline)
            dialogue_history.append(f"Patient: {next_user_message}")
            
            next_state = pipeline.build_state(next_user_message)
            reward = calculate_reward(state, next_state, action_id, uncert)
            
            # 3. Add to Replay Buffer
            transitions_buffer.append((state, action_id, reward, next_state))
            
            # 4. Trigger Training Loop step
            loss = train_dqn_step(transitions_buffer, update_counter)
            update_counter += 1
            
            # Metrics Console Output
            if loss:
                print(f"Step {step} | Reward: {reward:.2f} | DQN Loss: {loss:.4f}")
            else:
                print(f"Step {step} | Reward: {reward:.2f} | Filling Replay Buffer...")
                
            current_message = next_user_message
            
        # 5. Periodically Save Updated Weights 
        if episode % 5 == 0:
            print("--> Saving optimized weights checkpoint...")
            torch.save(pipeline.policy.state_dict(), "./models/policy_model_llm_2_trained.pt")