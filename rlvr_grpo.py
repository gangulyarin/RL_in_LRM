import torch
import torch.nn as nn
import torch.optim as optim
import random

device = "cpu"

###Vocab##

chars = list("0123456789+-= ")
stoi = {c:i for i,c in enumerate(chars)}
itos = {i:c for c,i in stoi.items()}

vocab_size = len(chars)

def encode(s):
    return torch.tensor([stoi[c] for c in s], dtype=torch.long)

def decode(t):
    return "".join([itos[int(i)] for i in t])


##Tiny LLM##

class TinyLM(nn.Module):

    def __init__(self):

        super().__init__()

        self.embed = nn.Embedding(vocab_size, 32)
        self.rnn = nn.GRU(32, 64, batch_first=True)

        self.policy_head = nn.Linear(64,vocab_size)
        

    def forward(self, x):
        e = self.embed(x)
        o,_ = self.rnn(e)

        logits = self.policy_head(o)
        

        return logits 

model = TinyLM().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

""" Let's use a realistic problem generator instead of preset Envrionment
#Environment
problems = [
    ("2+3=",5),
    ("4+4=",8),
    ("3+6=",9),
    ("5+2=",7)
]
"""

# Problem Generator (Environment)
def sample_problem():

    a = random.randint(1,9)
    b = random.randint(1,9)

    prompt = f"{a}+{b}="
    answer = a + b

    return prompt, answer



# For RLVR the reward function is known as Verifier
#def reward_fn(promt, response, answer):
def verifier(prompt, response, answer):
    try:
        r = int(response.strip())
        if r==answer:
            return 1.0
    except:
        pass

    return -1.0




#Sampling
def generate(model, prompt):
    tokens = encode(prompt).unsqueeze(0)
    logits = model(tokens)
    probabilities = torch.softmax(logits[:,-1],dim=-1)
    dist = torch.distributions.Categorical(probabilities)
    action = dist.sample()
    logprob = dist.log_prob(action)
    return action.item(), logprob


# Now RLVR + GRPO Training

group_size = 8

for step in range(3000):
    prompt,answer = sample_problem()

    log_probs = []
    rewards = []
    actions = []

    for _ in range(group_size):
        action,logprob = generate(model, prompt)
        response = decode([action])
        r = verifier(prompt, response, answer) # Same reward_fn
        log_probs.append(logprob)
        rewards.append(r)
        actions.append(response)

    rewards = torch.tensor(rewards, dtype=torch.float)

    mean = rewards.mean()
    std = rewards.std() + 1e-8

    advantage = (rewards-mean)/std

    loss = 0

    for logp, adv in zip(log_probs, advantage):

        loss += -logp * adv

    loss = loss / group_size

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 300 == 0:

        print("step:", step)
        print("problem:", prompt, answer)
        for a,r in zip(actions,rewards):
            print(" sample:", a, " reward:", r.item())
        print("mean reward:", rewards.mean().item())
        print()