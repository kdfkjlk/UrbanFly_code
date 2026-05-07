import torch
import torch.nn as nn


FixedCategorical = torch.distributions.Categorical

old_sample = FixedCategorical.sample
FixedCategorical.sample = lambda self: old_sample(self)

log_prob_cat = FixedCategorical.log_prob
FixedCategorical.log_probs = lambda self, actions: \
    log_prob_cat(self, actions.squeeze(-1))
FixedCategorical.mode = lambda self: self.probs.argmax(dim=1, keepdim=True)


class Categorical(nn.Module):

    def __init__(self, num_inputs, num_outputs):
        super(Categorical, self).__init__()
        self.linear = nn.Linear(num_inputs, num_outputs)

    def forward(self, x):
        x = self.linear(x)
        return FixedCategorical(logits=x)