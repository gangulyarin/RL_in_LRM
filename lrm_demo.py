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
        digits = "".join([c for c in response if c.isdigit()])
        if digits != "" and int(digits) == answer:
            return 1.0
    except:
        pass

    return -1.0



# Improve Generate to generate more then 1 tokens (4 tokens like 4+5=9)

def generate(model, prompt):
    max_tokens = 4
    generated = []
    logprob_sum = 0

    tokens = encode(prompt).unsqueeze(0)

    for _ in range(max_tokens):
        logits = model(tokens)
        probabilities = torch.softmax(logits[:,-1],dim=-1)
        dist = torch.distributions.Categorical(probabilities)
        action = dist.sample()
        logprob = dist.log_prob(action)

        generated.append(action.item())
        logprob_sum += logprob

        tokens = torch.cat([tokens, action.unsqueeze(0)], dim=1)
    return generated, logprob_sum


# Now RLVR + GRPO Training

group_size = 8

for step in range(3000):
    prompt,answer = sample_problem()

    log_probs = []
    rewards = []
    actions = []

    for _ in range(group_size):
        action,logprob = generate(model, prompt)
        #response = decode([action])
        response = decode(action)
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


"""
After some training the model may produce things like:

3+4=7
7
=7

instead of random digits.

This happens because longer outputs give the policy more ways to reach reward, 
which is exactly what happens in large reasoning models.
"""