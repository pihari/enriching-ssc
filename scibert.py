#%%
from transformers import *
import torch
import torch.nn as nn

tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_cased")
model = AutoModel.from_pretrained("allenai/scibert_scivocab_cased")


#%%
model.config

#%%
inputs = tokenizer("When using the one-hot encoding, the input sequence is represented by a $4 \times L$ matrix where $L$ is the length of the sequence and each position in the sequence has a four element vector with a single nonzero element corresponding to the nucleotide in that position.", return_tensors="pt")
outputs = model(**inputs)

#%%
def tokenize_file(file, max_seq_length, tokenizer):
    """

    Parameters

    file : .txt file
        - the specified text file.
    max_seq_length : int
        - maximum length of input sequences
    tokenizer
        - a pretrained BERT tokenizer

    Returns
    - vector list of tokenized features
    """
    with open(file) as f:
        tokens = []
        labels = []
        for sample in f:
            ts, ls = tokenize_string(sample, tokenizer)
            tokens.extend(ts)
            labels.extend(ls)

    return tokens

#%%
import re
def identify_variable(expr):
    to_remove = [" ", "$", "(", ")", "{", "}", "\\"]
    var = None
    if re.match(r"(.*?)\$(.*?)\$(.*?)", expr):
        for e in to_remove:
            expr = expr.replace(e, "")
        split_expr = expr.split("=")
        var = split_expr[0]
    return var


#%%
def tokenize_string(s, tokenizer):
    tokens = []
    labels = []
    word_list  = s.split(' ')
    for word in word_list:
        token = tokenizer.tokenize(word)
        print(token)
        tokens.extend(token)
        # ref: https://huggingface.co/transformers/custom_datasets.html
        # identify $..$ block and set variable label
        var = identify_variable(word)
        for t in token:
            if var:
                if t == var.lower():
                    labels.append("1") # is variable
                else:
                    labels.append("-100") # is from variable environment but no variable
            else:
                labels.extend("0") # is not a variable
    return tokens, labels

example_str = "More importantly, while the model's hyper-parameters were tuned on the validation set, the performance improvements translate to the private test set as scored by the ConvAI2 evaluation server with a $45\%$ absolute improvement in perplexity (PPL), $46\%$ absolute improvement in Hits@1 and $20\%$ improvement in F1."
ts, ls = tokenize_string(example_str, tokenizer)

#%%
example_in = tokenizer(example_str, return_tensors="pt")
example_out = model(**example_in)


#%%
cats = tokenizer("We describe a cat.", return_tensors="pt")
catm = model(**cats)
dogs = tokenizer("We describe a dog.", return_tensors="pt")
dogm = model(**dogs)
coss = nn.CosineSimilarity(dim=1, eps=1e-6)
sim = coss(catm[0], dogm[0])
torch.mean(sim)

#%%
e1 = tokenizer("budget", return_tensors="pt")
e1m = model(**e1)
e2 = tokenizer("resource", return_tensors="pt")
e2m = model(**e2)
sim_e1e2 = coss(e1m[0], e2m[0])
torch.mean(sim_e1e2)


#%%
