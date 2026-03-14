**Reinforcement Learning in LLM**

There are different types of Reinforcement Learning techniques used in building LLMs.
Widely adopted methods such as Reinforcement Learning from Human Feedback (RLHF) and Direct Preference Optimization (DPO) finetune pre-trained models to follow instructions and reflect human preferences.



RLHF uses PPO (Proximal Policy Optimization).

#PPO

Running code ppo.py gives us an idea of PPO clipping.


This script contains all PPO components used in RLHF.
Policy network
policy_head
Generates tokens.

Value network (critic)
value_head
Predicts expected reward.

Reward
reward_fn()
External signal.

Advantage
advantage = reward - value
Measures how good the action was.

PPO clipping
ratio = exp(new_logprob - old_logprob)
clipped_ratio = clamp(...)
Prevents large policy updates.

Real RLHF systems (used in early ChatGPT models) do the same loop:
prompt
↓
generate response
↓
reward model
↓
PPO update
The only differences:
Demo
Real systems
tiny RNN
huge transformers
simple reward
reward model
1 token output
long text

Let’s move on to GRPO now.

**GRPO**
GRPO is an RL algorithm.
For each prompt the model samples multiple responses and compares them.
Example:
Prompt: Solve this math problem

y1 → reward = 0
y2 → reward = 1
y3 → reward = 0
y4 → reward = 0
Then GRPO computes relative advantages inside the group:
Ai=ri−μGσGA_i = \frac{r_i - \mu_G}{\sigma_G}Ai​=σG​ri​−μG​​
where μ is the group average reward.
This advantage is used in a policy gradient update.
So GRPO is still reinforcement learning, just without a critic. 
Signal = reward function
Example reward:
correct answer → 1


incorrect → 0


or
code passes tests → reward


The model must explore outputs and learn via reward.
GRPO requires sampling outputs during training
prompt
→ generate 8–16 responses
→ compute reward
→ RL update
So it is online RL training.
GRPO is widely used in reasoning models (e.g. math/coding).
Key reasons:
works with verifiable rewards


no critic model required


more memory efficient than PPO


good for reasoning exploration


Many reasoning models like DeepSeek-R1 rely on GRPO-style RL.
GRPO Pipeline
Prompt
 ↓
Generate multiple outputs
 ↓
Score outputs (reward)
 ↓
Compute relative advantages
 ↓
Policy gradient update

Use GRPO when
reward is automatically verifiable


you want exploration


you train reasoning models


Example rewards:
math answer correct


code passes tests


proof verifies
The main difference between GRPO and PPO techniques are-
PPO
prompt
 ↓
generate response
 ↓
reward
 ↓
value network estimates expected reward
 ↓
advantage = reward − value
 ↓
policy update
Needs a critic (value network).
GRPO
prompt
 ↓
generate multiple responses
 ↓
compute rewards
 ↓
compare rewards within group
 ↓
advantage = normalized group reward
 ↓
policy update
No critic needed.
That’s why GRPO is simpler and more memory efficient.
Run grpo.py for GRPO demo. The output is like this:

$ python3 grpo.py
step: 0
prompt: 2+3=
 sample: 1  reward: -1.0
 sample:    reward: -1.0
 sample: 8  reward: -1.0
 sample: 6  reward: -1.0
 sample: 0  reward: -1.0
 sample: 4  reward: -1.0
mean reward: -1.0

step: 200
prompt: 5+2=
 sample: 7  reward: 1.0
 sample: 7  reward: 1.0
 sample: 7  reward: 1.0
 sample: 7  reward: 1.0
 sample: 7  reward: 1.0
 sample: 7  reward: 1.0
mean reward: 1.0

step: 400
prompt: 2+3=
 sample: 5  reward: 1.0
 sample: 5  reward: 1.0
 sample: 5  reward: 1.0
 sample: 5  reward: 1.0
 sample: 5  reward: 1.0
 sample: 5  reward: 1.0
mean reward: 1.0

step: 600
prompt: 5+2=
 sample: 7  reward: 1.0
 sample: 7  reward: 1.0
 sample: 7  reward: 1.0
 sample: 7  reward: 1.0
 sample: 7  reward: 1.0
 sample: 7  reward: 1.0
mean reward: 1.0

step: 800
prompt: 4+4=
 sample: 8  reward: 1.0
 sample: 8  reward: 1.0
 sample: 8  reward: 1.0
 sample: 8  reward: 1.0
 sample: 8  reward: 1.0
 sample: 8  reward: 1.0
mean reward: 1.0

step: 1000
prompt: 3+6=
 sample: 9  reward: 1.0
 sample: 9  reward: 1.0
 sample: 9  reward: 1.0
 sample: 9  reward: 1.0
 sample: 9  reward: 1.0
 sample: 9  reward: 1.0
mean reward: 1.0

step: 1200
prompt: 5+2=
 sample: 7  reward: 1.0
 sample: 7  reward: 1.0
 sample: 7  reward: 1.0
 sample: 7  reward: 1.0
 sample: 7  reward: 1.0
 sample: 7  reward: 1.0
mean reward: 1.0

step: 1400
prompt: 3+6=
 sample: 9  reward: 1.0
 sample: 9  reward: 1.0
 sample: 9  reward: 1.0
 sample: 9  reward: 1.0
 sample: 9  reward: 1.0
 sample: 9  reward: 1.0
mean reward: 1.0

step: 1600
prompt: 3+6=
 sample: 9  reward: 1.0
 sample: 9  reward: 1.0
 sample: 9  reward: 1.0
 sample: 9  reward: 1.0
 sample: 9  reward: 1.0
 sample: 9  reward: 1.0
mean reward: 1.0

step: 1800
prompt: 4+4=
 sample: 8  reward: 1.0
 sample: 8  reward: 1.0
 sample: 8  reward: 1.0
 sample: 8  reward: 1.0
 sample: 8  reward: 1.0
 sample: 8  reward: 1.0
mean reward: 1.0



Memory usage is smaller in GRPO.

Why GRPO Works Well for Reasoning
Suppose model generates:
2+3=
Group outputs:
5
6
7
5
4
5
Rewards:
[1,0,0,1,0,1]
Normalized advantages:
[+,+,-,+,-,+]
The model learns:
increase probability of "5"
decrease others
No critic required.

The code demonstrates the same core idea used in reasoning models like:
DeepSeek-R1


DeepSeek-R1-Zero


Their pipeline is basically:
pretraining
↓
instruction tuning
↓
RLVR training
↓
GRPO optimization

Now let us then understand RLVR

**RLVR**
RLVR = Reinforcement Learning with Verifiable Rewards.
Instead of learning from human preferences, the model learns from objective correctness signals.
Example tasks:
Task
Verifier
math
check final answer
coding
run unit tests
logic puzzle
compare with ground truth
structured output
format checker

Why RLVR Became Popular
Earlier systems used RLHF (Reinforcement Learning from Human Feedback).
RLHF pipeline:
human comparisons
     ↓
reward model
     ↓
RL optimization
Problems:
expensive human labeling


reward model can be gamed


noisy signals


RLVR replaces that with objective rewards:
math correctness
code execution
formal verification
This gives:
cheaper training


longer RL runs


better reasoning behavior


Many 2025 reasoning models rely on this paradigm.
RLVR still needs an RL algorithm.


DeepSeek-R1 specifically used GRPO.
GRPO works like this:
Generate multiple responses for a prompt


Compute rewards


Compare them relative to each other


**Why RLVR Produces Reasoning**
A surprising effect discovered with DeepSeek-R1-Zero:
They trained a model using:
RLVR
+
math problems
+
binary reward
No reasoning examples.
Yet the model started doing:
step-by-step reasoning


backtracking


self-verification


Why?
Because reasoning increases probability of correct answers, so RL reinforces it.
So reasoning emerges as a strategy.

Typical RLVR Training Pipeline
Real systems look roughly like this:
Pretraining
  ↓
Instruction tuning
  ↓
RLVR phase
  ↓
Reasoning model
During RLVR:
prompt
 ↓
generate N responses
 ↓
verifier
 ↓
reward
 ↓
GRPO update

Why RLVR Works Best for Reasoning Tasks
RLVR requires verifiable outcomes.
Good domains:
math


code


logic puzzles


theorem proving


tool usage


Bad domains:
creative writing


empathy


subjective answers


Because correctness can't be verified easily.
Now Let’s modify GRPO to RLVR.

We’ll build a RLVR + GRPO demo that mimics the core idea behind models like:
DeepSeek-R1


DeepSeek-R1-Zero


The script will:
Generate math problems automatically


Use a verifier to check answers


Train using GRPO


Improve accuracy over time


Run rlvr_grpo.py and you will get the below output:
python3 rlvr_grpo.py
step: 0
problem: 7+7= 14
 sample: 4  reward: -1.0
 sample: 5  reward: -1.0
 sample: 0  reward: -1.0
 sample:    reward: -1.0
 sample:    reward: -1.0
 sample: 0  reward: -1.0
 sample: 9  reward: -1.0
 sample: 8  reward: -1.0
mean reward: -1.0

step: 300
problem: 3+8= 11
 sample: 5  reward: -1.0
 sample: 9  reward: -1.0
 sample: 7  reward: -1.0
 sample: 2  reward: -1.0
 sample: 9  reward: -1.0
 sample: 9  reward: -1.0
 sample: 9  reward: -1.0
 sample: 3  reward: -1.0
mean reward: -1.0

step: 600
problem: 9+9= 18
 sample: 9  reward: -1.0
 sample: 5  reward: -1.0
 sample: 7  reward: -1.0
 sample: 9  reward: -1.0
 sample: 9  reward: -1.0
 sample: 5  reward: -1.0
 sample: 5  reward: -1.0
 sample: 5  reward: -1.0
mean reward: -1.0

step: 900
problem: 6+9= 15
 sample: 9  reward: -1.0
 sample: 9  reward: -1.0
 sample: 9  reward: -1.0
 sample: 9  reward: -1.0
 sample: 9  reward: -1.0
 sample: 9  reward: -1.0
 sample: 9  reward: -1.0
 sample: 9  reward: -1.0
mean reward: -1.0

step: 1200
problem: 4+4= 8
 sample: 5  reward: -1.0
 sample: 5  reward: -1.0
 sample: 5  reward: -1.0
 sample: 5  reward: -1.0
 sample: 5  reward: -1.0
 sample: 5  reward: -1.0
 sample: 5  reward: -1.0
 sample: 5  reward: -1.0
mean reward: -1.0

step: 1500
problem: 7+7= 14
 sample: 9  reward: -1.0
 sample: 9  reward: -1.0
 sample: 9  reward: -1.0
 sample: 9  reward: -1.0
 sample: 9  reward: -1.0
 sample: 9  reward: -1.0
 sample: 9  reward: -1.0
 sample: 9  reward: -1.0
mean reward: -1.0

step: 1800
problem: 5+3= 8
 sample: 9  reward: -1.0
 sample: 9  reward: -1.0
 sample: 9  reward: -1.0
 sample: 9  reward: -1.0
 sample: 9  reward: -1.0
 sample: 9  reward: -1.0
 sample: 9  reward: -1.0
 sample: 9  reward: -1.0
mean reward: -1.0

step: 2100
problem: 7+9= 16
 sample: 9  reward: -1.0
 sample: 9  reward: -1.0
 sample: 9  reward: -1.0
 sample: 9  reward: -1.0
 sample: 9  reward: -1.0
 sample: 9  reward: -1.0
 sample: 9  reward: -1.0
 sample: 9  reward: -1.0
mean reward: -1.0

step: 2400
problem: 6+2= 8
 sample: 9  reward: -1.0
 sample: 9  reward: -1.0
 sample: 9  reward: -1.0
 sample: 9  reward: -1.0
 sample: 9  reward: -1.0
 sample: 9  reward: -1.0
 sample: 9  reward: -1.0
 sample: 9  reward: -1.0
mean reward: -1.0

step: 2700
problem: 1+5= 6
 sample: 9  reward: -1.0
 sample: 9  reward: -1.0
 sample: 9  reward: -1.0
 sample: 9  reward: -1.0
 sample: 9  reward: -1.0
 sample: 9  reward: -1.0
 sample: 9  reward: -1.0
 sample: 9  reward: -1.0
mean reward: -1.0

During RLVR training in real systems, researchers observed something surprising.
Models began producing:
Let me calculate step by step:
3 + 4 = 7
Even when reasoning text was never trained.
Reason:
reasoning → higher correctness probability
→ RL reinforces it
This phenomenon was reported in training of DeepSeek-R1-Zero.
After all this, we missed something popular in the 2023 - DPO or Direct Policy Optimization

**DPO**
DPO was introduced due to the below problems of RLHF:
Problems with RLHF
The paper argues RLHF is overcomplicated:
Requires 3 models


policy model


reward model


reference model


RL training is unstable


expensive


lots of hyperparameters


Instead of training a reward model + RL loop, DPO directly trains the model using preference pairs.
Given:
(prompt x, chosen y_w, rejected y_l)
We want:
P(y_w | x) > P(y_l | x)
But with a constraint:
stay close to a reference model.
Think of it like this:
Instead of:
Human → reward model → reinforcement learning
DPO does:
Human preference pairs → directly train model
So training becomes just supervised learning.
Why the DPO paper became so relevant and important so as to be adapted by Llama 3?
The title of the paper was - “Your Language Model is Secretly a Reward Model”
The authors show that the RLHF objective:
maximize reward − KL penalty
has a closed-form optimal policy:
π*(y|x) ∝ π_ref(y|x) * exp( reward(x,y) / β )
Rearranging gives:
reward(x,y) ≈ log π(y|x) − log π_ref(y|x)
Meaning:
The difference between model probabilities already behaves like a reward.
So instead of learning reward → RL → policy
You can train the policy directly.
Visual Comparison
RLHF
Human ranking
     ↓
Reward model
     ↓
RL optimization (PPO)
     ↓
Aligned model
DPO
Human ranking
     ↓
Direct optimization
     ↓
Aligned model

Difference between DPO and GRPO
DPO
DPO trains directly on human preference pairs:
(prompt, good_answer, bad_answer)
It simply increases the probability of the preferred answer relative to the rejected one.
Training looks like normal supervised learning:
maximize log P(good_answer) - log P(bad_answer)
No rollouts, no reward model, no policy gradient.
Signal = human preference dataset

GRPO
GRPO is an RL algorithm.
For each prompt the model samples multiple responses and compares them.
Example:
Prompt: Solve this math problem

y1 → reward = 0
y2 → reward = 1
y3 → reward = 0
y4 → reward = 0
Then GRPO computes relative advantages inside the group:
Ai=ri−μGσGA_i = \frac{r_i - \mu_G}{\sigma_G}Ai​=σG​ri​−μG​​
where μ is the group average reward.
This advantage is used in a policy gradient update.
So GRPO is still reinforcement learning, just without a critic.
Signal = reward function

DPO pipeline
Human preference data
       ↓
DPO loss
       ↓
Aligned model

GRPO pipeline
Prompt
 ↓
Generate multiple outputs
 ↓
Score outputs (reward)
 ↓
Compute relative advantages
 ↓
Policy gradient update

When Each Works Best
Use DPO when
you have human preference data


you want stable training


you are aligning chat assistants



Use GRPO when
reward is automatically verifiable


you want exploration


you train reasoning models


Example rewards:
math answer correct


code passes tests


proof verifies

Method
Signal
What it demonstrates


DPO
human preference pairs
classic LLM alignment


PPO
reward model
RLHF pipeline


GRPO
group relative reward
reasoning RL


DPO
quickly learns preferred responses


stable


PPO
slower


depends on reward signal


GRPO
improves reasoning


benefits from multiple samples
Relationship Between DPO, PPO, GRPO, RLVR
Here’s the key conceptual hierarchy:
Alignment / RL methods
│
├─ DPO
│   (supervised preference learning)
│
├─ RLHF
│   └─ PPO
│
└─ RLVR
   ├─ PPO
   └─ GRPO
So:
DPO → preference learning


RLHF → RL with human rewards


RLVR → RL with automatic verifiers


GRPO → algorithm used inside RLVR
Use the below code if you wish to see the relative differences:
Dataset for the Experiment
Create a small dataset.
dataset = [
   {
       "prompt": "What is 2+3?",
       "chosen": "5",
       "rejected": "6"
   },
   {
       "prompt": "What is 4+4?",
       "chosen": "8",
       "rejected": "6"
   },
   {
       "prompt": "What is 10-3?",
       "chosen": "7",
       "rejected": "5"
   }
]
This dataset is used by DPO.

DPO Example (Simplest)
DPO directly trains using preference pairs.
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig

model = AutoModelForCausalLM.from_pretrained("distilgpt2")
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")

config = DPOConfig(
   per_device_train_batch_size=1,
   learning_rate=5e-6
)

trainer = DPOTrainer(
   model=model,
   args=config,
   train_dataset=dataset,
   tokenizer=tokenizer
)

trainer.train()
What happens internally:
maximize log P(chosen) - log P(rejected)
No RL loop.

PPO Example (Classic RLHF)
Pipeline:
prompt
  ↓
generate response
  ↓
reward model score
  ↓
PPO update
Example:
from trl import PPOTrainer, PPOConfig

ppo_config = PPOConfig(
   batch_size=1,
)

ppo_trainer = PPOTrainer(
   model=model,
   config=ppo_config,
   tokenizer=tokenizer
)

prompt = "What is 2+3?"

response = ppo_trainer.generate(prompt)

reward = 1 if "5" in response else -1

ppo_trainer.step([prompt], [response], [reward])
This simulates RLHF.

GRPO Example (Key Difference)
GRPO generates multiple outputs per prompt.
Example:
Prompt: What is 2+3?

Samples:
y1 = "5"  reward=1
y2 = "6"  reward=0
y3 = "4"  reward=0
y4 = "5"  reward=1
Then compute relative advantage.
Simplified code:
import numpy as np

responses = generate_n(model, prompt, n=4)

rewards = []
for r in responses:
   rewards.append(1 if "5" in r else 0)

mean = np.mean(rewards)
std = np.std(rewards) + 1e-8

advantages = [(r - mean)/std for r in rewards]
Then update policy:
loss = - advantage * log_prob(response)
Pseudo-update:
loss = 0
for resp, adv in zip(responses, advantages):
   logprob = model_logprob(prompt, resp)
   loss += -adv * logprob

loss.backward()
optimizer.step()

