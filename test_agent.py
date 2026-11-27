import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import Counter

from environment.digital_twin_env import DigitalTwinEnv
from models.dqn_model import DQNAgent

ACTION_NAMES = [
    "🔄 Maintain",
    "📶 Boost Signal",
    "📡 Switch Frequency",
    "🗜️  Compress Data",
    "🔀 Switch Ground Station"
]

def test():
    print("=" * 60)
    print("  COGNITIVE SATELLITE DOWNLINK - AGENT EVALUATION")
    print("  ISRO Internship Project")
    print("=" * 60)

    env = DigitalTwinEnv()
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = DQNAgent(state_size, action_size)

    if os.path.exists("dqn_model.pth"):
        agent.model.load_state_dict(torch.load("dqn_model.pth"))
        agent.epsilon = 0.0
        print("✅ Trained model loaded successfully\n")
    else:
        print("⚠️  No trained model found. Run train_agent.py first!\n")

    agent.model.eval()

    test_episodes = 5
    all_rewards   = []
    all_latencies = []

    for ep in range(test_episodes):
        state = env.reset()
        total_reward = 0
        latencies    = []
        action_log   = []

        print(f"\n{'─'*60}")
        print(f"  TEST EPISODE {ep+1}")
        print(f"{'─'*60}")
        print(f"{'Step':>4} | {'Action':<25} | {'Signal':>6} | {'Latency':>8} | {'Reward':>8} | GS")
        print(f"{'─'*80}")

        for step in range(30):
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)

            signal  = state[0]
            latency = state[2]
            gs      = info.get("ground_station", "?")

            print(f"{step+1:>4} | {ACTION_NAMES[action]:<25} | "
                  f"{signal:>6.1f} | {latency:>7.1f}ms | "
                  f"{reward:>8.3f} | {gs}")

            latencies.append(latency)
            action_log.append(action)
            total_reward += reward
            state = next_state

            if done:
                break

        avg_latency = np.mean(latencies)
        all_rewards.append(total_reward)
        all_latencies.append(avg_latency)

        action_dist = Counter(action_log)
        print(f"\n  📊 Episode {ep+1} Summary:")
        print(f"     Total Reward : {total_reward:.3f}")
        print(f"     Avg Latency  : {avg_latency:.1f} ms")
        print(f"     Action Usage :", {ACTION_NAMES[k]: v for k, v in sorted(action_dist.items())})

    print(f"\n{'='*60}")
    print(f"  OVERALL TEST SUMMARY ({test_episodes} episodes)")
    print(f"{'='*60}")
    print(f"  Avg Total Reward : {np.mean(all_rewards):.3f} ± {np.std(all_rewards):.3f}")
    print(f"  Avg Latency      : {np.mean(all_latencies):.1f} ms")
    print(f"  Best Episode     : Episode {np.argmax(all_rewards)+1} ({max(all_rewards):.3f})")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Agent Test Performance - ISRO Project", fontsize=13, fontweight='bold')

    ax1.bar(range(1, test_episodes+1), all_rewards, color='steelblue')
    ax1.set_title('Total Reward per Test Episode')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.grid(True, alpha=0.3, axis='y')

    ax2.bar(range(1, test_episodes+1), all_latencies, color='orange')
    ax2.set_title('Avg Latency per Test Episode')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Latency (ms)')
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('test_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\n  Saved: test_results.png")


if __name__ == "__main__":
    test()