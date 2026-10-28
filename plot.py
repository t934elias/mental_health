# import matplotlib.pyplot as plt
# import numpy as np
# import seaborn as sns

# # Configure scientific paper styling
# # plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available() else 'default')
# plt.rcParams.update({
#     'font.family': 'serif',
#     'font.size': 11,
#     'axes.labelsize': 12,
#     'axes.titlesize': 13,
#     'xtick.labelsize': 10,
#     'ytick.labelsize': 10,
#     'figure.titlesize': 14
# })

# # Mock data generation based on your training logs
# np.random.seed(42)
# episodes = np.arange(1, 401)
# # Simulate reward progression climbing over 400 episodes
# raw_rewards = -8.0 + 9.5 * (1 - np.exp(-episodes/120)) + np.random.normal(0, 3, 400)
# # Create a 20-episode rolling window average
# rolling_avg = np.convolve(raw_rewards, np.ones(20)/20, mode='valid')
# rolling_episodes = episodes[19:]

# # Initialize a clean, single-column figure layout
# fig, ax1 = plt.subplots(figsize=(7, 4.5), dpi=300)

# # Plot raw reward data with low opacity to show distribution variance
# ax1.plot(episodes, raw_rewards, color='#3498db', alpha=0.25, label='Raw Episode Reward')
# # Plot smoothed trendline to emphasize convergence profile
# ax1.plot(rolling_episodes, rolling_avg, color='#2c3e50', linewidth=2.2, label='20-Episode Moving Average')

# # Scientific detailing
# ax1.set_xlabel('Training Episodes', fontweight='bold')
# ax1.set_ylabel('Cumulative Reward', fontweight='bold')
# ax1.set_title('Figure 1: Policy Reward Convergence Across Training Manifest', pad=15)
# ax1.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.7) # Highlights threshold crossing
# ax1.legend(loc='lower right', frameon=True, facecolor='white', edgecolor='none')

# plt.tight_layout()
# plt.savefig('results/rl_model_performance.pdf', bbox_inches='tight', dpi=300) # Save as vector format for LaTeX integration
# plt.show()




# 	total_reward
# 1	{'episode_reward': np.float32(-9.266072), 'losses': [], 'actions': [1, 2, 5, 0, 5, 6, 0, 5], 'risks': [np.float32(0.5421), np.float32(0.8301), np.float32(0.6127), np.float32(0.9243), np.float32(0.7583), np.float32(0.7451), np.float32(0.7402), np.float32(0.5933)], 'uncertainties': [0.012644858099520206, 0.019195258617401123, 0.08824042230844498, 0.01357739046216011, 0.07111405581235886, 0.02240510657429695, 0.05240369960665703, 0.0051825693808496]}






# averg_loss = uncertainty
import ast
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re

# 1. Enforce Scientific Publishing Style
# plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available() else 'default')
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'figure.titlesize': 13
})

action_names = {
    0: 'empathy',
    1: 'reflection',
    2: 'clarifying_q',
    3: 'coping',
    4: 'safety_check',
    5: 'prof_help',
    6: 'abstain'
}

def parse_safely(value):

    if isinstance(value, dict):
        return value

    return eval(
        value,
        {"np": np, "__builtins__": {}}
    )


def compute_rolling_stats(values, window=20):
    values = pd.Series(values)
    mean = values.rolling(window=window, min_periods=1).mean().values
    std = values.rolling(window=window, min_periods=1).std().fillna(0).values
    return mean, std


df = pd.read_csv("./models/training_logs.csv")
df = df[df["episode"] != "episode"].copy()
df["episode"] = df["episode"].astype(int)

df.reset_index(drop=True, inplace=True)
# print(df.columns)
# print(df.head())
# print(df.iloc[0])

flat_rewards = []
flat_losses = []
flat_risks = []
flat_uncertainties = []
action_matrix = []

print("Processing training logs...")
# print(df["total_reward"].iloc[0])
for _, row in df.iterrows():
    
    if not str(row["total_reward"]).startswith("{"):
        continue
    data = parse_safely(row["total_reward"])
    

    flat_rewards.append(data.get("episode_reward", 0))

    losses = data.get("losses", [])
    flat_losses.append(np.mean(losses) if len(losses) else np.nan)

    risks = data.get("risks", [])
    flat_risks.append(np.mean(risks) if len(risks) else np.nan)

    

    # uncertainties = data.get("uncertainties", [])
    # flat_uncertainties.append(
    #     np.mean(uncertainties) if len(uncertainties) else np.nan
    # )

    action_matrix.append(data.get("actions", []))

episodes = np.arange(1, len(flat_rewards) + 1)

flat_rewards = np.array(flat_rewards)
flat_uncertainties = df["avg_loss"].astype(float).values

flat_losses = (
    pd.Series(flat_losses)
    .ffill()
    .bfill()
    .values
)

flat_risks = (
    pd.Series(flat_risks)
    .ffill()
    .bfill()
    .values
)

flat_uncertainties = (
    pd.Series(flat_uncertainties)
    .ffill()
    .bfill()
    .values
)

window = 20

reward_smooth = pd.Series(flat_rewards).rolling(window, min_periods=1).mean().values
reward_std = pd.Series(flat_rewards).rolling(window, min_periods=1).std().fillna(0).values

loss_smooth = pd.Series(flat_losses).rolling(window, min_periods=1).mean().values

risk_smooth = pd.Series(flat_risks).rolling(window, min_periods=1).mean().values

unc_smooth, unc_std = compute_rolling_stats(flat_uncertainties, window)

# --- 3. Process Action Profiles over Time ---
# We count how often each action was picked in 4 chronological training quarters
n = len(action_matrix)
quarter_size = n // 4

quarters = [
    action_matrix[:quarter_size],
    action_matrix[quarter_size:2*quarter_size],
    action_matrix[2*quarter_size:3*quarter_size],
    action_matrix[3*quarter_size:]
]

# quarters = np.array_split(action_matrix, 4)
quarter_counts = []
for q in quarters:
    all_actions_in_q = [action for episode in q for action in episode]
    counts = {name: all_actions_in_q.count(id) for id, name in action_names.items()}
    quarter_counts.append(counts)

n = len(df)
quarter_labels = []
for i in range(4):
    start = i * n // 4 + 1
    end = (i + 1) * n // 4
    quarter_labels.append(f"Q{i+1} ({start}-{end})")

df_actions = pd.DataFrame(
    quarter_counts,
    index=quarter_labels
)

df_actions_pct = df_actions.div(df_actions.sum(axis=1), axis=0) * 100

fig, axes = plt.subplots(3, 2, figsize=(12, 12), dpi=300)

# Panel A: Rewards
axes[0, 0].plot(episodes, flat_rewards, color='#3498db', alpha=0.2, label='Raw Episodic')
axes[0, 0].plot(episodes, reward_smooth, color='#2c3e50', linewidth=1.8, label='Moving Mean')
axes[0, 0].fill_between(episodes, reward_smooth - reward_std, reward_smooth + reward_std, color='#2c3e50', alpha=0.1)
axes[0, 0].axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
axes[0, 0].set_title("A: Cumulative Reward Trajectory")
axes[0, 0].set_xlabel("Training Episodes")
axes[0, 0].set_ylabel("Total Return")
axes[0, 0].legend(loc='lower right')

# Panel B: Optimization Loss
axes[0, 1].plot(episodes, flat_losses, color='#e74c3c', alpha=0.25)
axes[0, 1].plot(episodes, loss_smooth, color='#c0392b', linewidth=1.8, label='Smoothed Loss')
axes[0, 1].set_title("B: Neural Network Optimization Convergence")
axes[0, 1].set_xlabel("Training Episodes")
axes[0, 1].set_ylabel("Mean Episode Loss")
axes[0, 1].legend(loc='upper right')

# Panel C: Monitored Risk Management
axes[1, 0].plot(episodes, flat_risks, color='#2ecc71', alpha=0.25)
axes[1, 0].plot(episodes, risk_smooth, color='#27ae60', linewidth=1.8, label='Smoothed Risk')
axes[1, 0].axhline(0.75, color='#e67e22', linestyle=':', label='Target Risk Threshold')
axes[1, 0].set_title("C: Tracked Environment Safety Risk Profile")
axes[1, 0].set_xlabel("Training Episodes")
axes[1, 0].set_ylabel("Mean User Crisis Risk")
axes[1, 0].legend(loc='upper right')

# Panel D: Behavioral Shift Stacked Chart
df_actions_pct.plot(kind='bar', stacked=True, ax=axes[1, 1], cmap='tab20', edgecolor='white', linewidth=0.5)
axes[1, 1].set_title("D: Policy Strategy Evolution Across Quarters")
axes[1, 1].set_xlabel("Training Timeline Split")
axes[1, 1].set_ylabel("Action Selection Distribution (%)")
axes[1, 1].set_xticklabels(df_actions_pct.index, rotation=0)
axes[1, 1].legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True)

axes[2,0].plot(episodes, flat_uncertainties, color="#9b59b6", alpha=0.2, label="Raw")
axes[2,0].plot(episodes, unc_smooth, color="#8e44ad", linewidth=2, label="Moving Mean")
axes[2,0].fill_between(episodes, unc_smooth - unc_std, unc_smooth + unc_std, color="#8e44ad", alpha=0.12)
axes[2,0].set_title("E: Model Policy Epistemic Uncertainty")
axes[2,0].set_xlabel("Training Episodes")
axes[2,0].set_ylabel("Predictive Uncertainty")
axes[2,0].legend()

fig.delaxes(axes[2,1])

plt.tight_layout()
plt.savefig("./plots/scientific_metrics.pdf", bbox_inches='tight', dpi=300)
print("Complete! Look for 'unpacked_scientific_metrics.pdf' in your folder.")
plt.show()






# import matplotlib.pyplot as plt

# # Stylesheet settings for academic documents
# plt.rcParams.update({
#     'font.family': 'serif',
#     'font.size': 9,
#     'axes.labelsize': 10
# })

# fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), dpi=300)

# # Common styling details
# box_props_user = dict(boxstyle='round,pad=0.6', facecolor='#e0e0e0', alpha=0.9, edgecolor='none')
# box_props_early = dict(boxstyle='round,pad=0.6', facecolor='#fce4e4', alpha=0.9, edgecolor='#c0392b', linewidth=1)
# box_props_opt = dict(boxstyle='round,pad=0.6', facecolor='#e8f8f5', alpha=0.9, edgecolor='#27ae60', linewidth=1)

# # ----------------------------------------------------
# # LEFT PANEL: Early Unoptimized Policy Defect
# # ----------------------------------------------------
# ax1 = axes[0]
# ax1.text(0.05, 0.88, "User:\n\"I don't see the point of trying anymore...\"", bbox=box_props_user, va='top', ha='left', wrap=True)#, width=280)
# ax1.text(0.40, 0.65, "Agent (Step 1) [Action: ABSTAIN]:\n\"I am an AI assistant. Please let me\nknow if you need help writing code.\"", bbox=box_props_early, va='top', ha='left', wrap=True)

# # Annotate the early termination failure
# ax1.text(0.5, 0.35, "⚠️ Catastrophic Termination\nUser Risk Spiked to 0.968", color='#c0392b', 
#          fontweight='bold', ha='center', bbox=dict(boxstyle='square,pad=0.4', facecolor='white', edgecolor='#c0392b'))

# ax1.set_title("A: Early Policy (Passive Failure Path)", color='#c0392b', fontweight='bold', pad=10)
# ax1.axis('off')

# # ----------------------------------------------------
# # RIGHT PANEL: Optimized RL Policy Success
# # ----------------------------------------------------
# ax2 = axes[1]
# ax2.text(0.05, 0.88, "User:\n\"I don't see the point of trying anymore...\"", bbox=box_props_user, va='top', ha='left', wrap=True)#, width=280)
# ax2.text(0.40, 0.72, "Agent (Step 1) [Action: SAFETY_CHECK]:\n\"I hear how heavy things are. Are you\nhaving thoughts of hurting yourself?\"", bbox=box_props_opt, va='top', ha='left')
# ax2.text(0.40, 0.48, "Agent (Step 2) [Action: COPING]:\n\"Thank you for being open. Let's take a\nslow breath. You aren't alone.\"", bbox=box_props_opt, va='top', ha='left')
# ax2.text(0.40, 0.24, "Agent (Step 3) [Action: PROF_HELP]:\n\"Please reach out to 988. There are real\npeople ready to support you right now.\"", bbox=box_props_opt, va='top', ha='left')

# # Annotate success
# ax2.text(0.7, 0.05, "✅ Safe De-escalation\nRisk Dropped to 0.48", color='#27ae60', fontweight='bold', ha='center')

# ax2.set_title("Optimized RL Policy (Active De-escalation)", color='#27ae60', fontweight='bold', pad=10)
# ax2.axis('off')

# plt.tight_layout()
# plt.savefig("./plots/dialogue_transcript_comparison.pdf", bbox_inches='tight', dpi=300)
# print("Transcript visual compiled beautifully into: dialogue_transcript_comparison.pdf")