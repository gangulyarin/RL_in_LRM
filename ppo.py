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
        self.value_head = nn.Linear(64, 1)

    def forward(self, x):
        e = self.embed(x)
        o,_ = self.rnn(e)

        logits = self.policy_head(o)
        values = self.value_head(o).squeeze(-1)

        return logits, values

model = TinyLM().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

#Environment
problems = [
    ("2+3=",5),
    ("4+4=",8),
    ("3+6=",9),
    ("5+2=",7)
]

def reward_fn(promt, response, answer):
    try:
        r = int(response.strip())
        if r==answer:
            return 1.0
    except:
        pass

    return -1.0

#Sampling
def generate(model, prompt):
    model.eval()
    tokens = encode(prompt).unsqueeze(0)
    with torch.no_grad():
        logits,_ = model(tokens)
    probabilities = torch.softmax(logits[:,-1],dim=-1)
    dist = torch.distributions.Categorical(probabilities)
    action = dist.sample()
    logprob = dist.log_prob(action)
    return action.item(), logprob

#PPO Training

clip_eps = 0.2
gamma = 1.0
value_coef = 0.5

for step in range(2000):
    prompt,answer = random.choice(problems)
    action,old_logprob = generate(model, prompt)
    response = decode([action])
    r = reward_fn(prompt, response, answer)
    tokens = encode(prompt).unsqueeze(0)
    logits, values = model(tokens)
    probs = torch.softmax(logits[:,-1], dim=-1)
    dist = torch.distributions.Categorical(probs)
    new_logprob = dist.log_prob(torch.tensor(action))
    value = values[:,-1]
    advantage = torch.tensor(r) - value.detach()
    ratio = torch.exp(new_logprob - old_logprob)
    clipped_ratio = torch.clamp(ratio, 1-clip_eps, 1+clip_eps)
    policy_loss = -torch.min(
        ratio * advantage,
        clipped_ratio * advantage
    )
    value_loss = (value - r)**2
    loss = policy_loss + value_coef * value_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 200 == 0:

        print("step:", step)
        print("prompt:", prompt)
        print("model response:", response)
        print("reward:", r)
        print()