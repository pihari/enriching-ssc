
#%%
# load torch model
import os
#%%
# util functions
import re

import numpy as np
import torch
import torch.nn as nn
#%%
# for each code function, encode its embeddings
# in shared embedding space look for closest encoded sentence
import torch.nn.functional as F
from transformers import *

from ast_traverser import *
from simple_ae import *

modelname = "full_ae.pt"
ae_model = torch.load(modelname)

text_tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_cased")
text_model = AutoModel.from_pretrained("allenai/scibert_scivocab_cased")
code_tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
code_model = AutoModel.from_pretrained("microsoft/codebert-base")


def identify_variable(expr):
    """
    Extracts a variable from a .tex line.
    """
    math_envs = ["mathcal", "mathbb", "mathbf", "mathnormal"]
    math_symb = ["widetilde", "bar", "hat"]
    to_remove = ["$", "{", "}", "\\"]
    var = None
    if re.match(r"(.*?)\$(.*?)\$(.*?)", expr):
        split_expr = expr.split("$")
        for e in to_remove:
            expr = expr.replace(e, "")
        var = split_expr[1] # TODO: only gets first $..$ context
        for e in to_remove+[" "]+math_envs+math_symb:
            var = var.replace(e, "")
        var = var.split("=")[0]
        var = var.split("_")[0]
        # Prevent stuff like "Lemma $2$ shows that..."
        if re.match(r"(?<!\S)\d(?!\S)", var):
            var = None
    return var

def populate_sentence_dict(sentences):
    sdict = {}
    for line in sentences:
        v = identify_variable(line)
        entry = []
        if v in sdict:
            entry = sdict[v]
        entry.append(line)
        sdict[v] = entry
    if None in sdict:
        del sdict[None]
    return sdict

filter_list = [
    ("begin", "figure"),
    ("end", "figure"),
    ("begin", "equation"),
    ("end", "equation"),
    ("begin", "table"),
    ("end", "table"),
    ("begin", "algorithm"),
    ("end", "algorithm"),
    ("begin", "align"),
    ("end", "align"),
    ("section", None),
    ("subsection", None),
    ("subfigure", None),
    ("city", None) 
]
def compile_retex(t):
    cmd, env = t
    if env:
        _retex = re.compile(rf"\\{cmd}{{{env}}}")
    else:
        _retex = re.compile(rf"\\{cmd}{{(.*?)({{(.*?)}})*(.*?)}}")
    return _retex

def process_tex_file(f_path, filename):
    sentences = []
    prefix_filters = ["%", "\\", "\n", "}", " ", "<", ">", "\\\\"]
    with open(f_path+filename) as f:
        print("file open")
        in_env = False
        in_eq = False
        try:
            for line in f:
                any_match = False
                any_prefix = False
                for e in filter_list:
                    if re.match(compile_retex(e), line):
                        any_match = True
                        cmd, _ = e
                        in_env = True if cmd is "begin" else False
                if re.findall(r"\$\$(.*?)\$\$", line):
                    any_match = True
                if not any_match and re.match(r"\$\$", line):
                    in_eq = not in_eq
                for e in prefix_filters:
                    if line.startswith(e):
                        #print(line)
                        any_prefix = True
                if not (
                    any_match or 
                    any_prefix or 
                    in_env or
                    in_eq or
                    len(line.split()) < 3
                    ):
                    # split only at sentence delimeter dots
                    #lines = re.split(r"(?<![\d\.(\.\w)])\.(?![\d\.(\w\.)])", line)
                    lines = re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s", line)
                    for l in lines:
                        # FIXME: excludes blank lines
                        if not re.match(r"^(?:[\t ]*(?:\r?\n|\r))+", l):
                            s = l.replace("\n", "")#+"."
                            sentences.append(s)
        except UnicodeDecodeError as err:
            print("OS error: {0}".format(err))
    return sentences

def extract_functions_from_py_file(path, filename):
    file = path+filename
    t = None
    with open(file, "r") as source:
        try:
            t = ast.parse(source.read())
        except SyntaxError:
            pass

    if t:
        funcFinder = FunctionFinder()
        funcFinder.visit(t)
        fdict = funcFinder.getFdict()
        trav = ASTTraverser()
        trav.populate_dicts(file, fdict)
        return trav.get_var_dict()



#%%
# process repo and text
datapath = "weighted-removal"
papername = "arXiv/2104.05274/"
reponame = ""
file_regex = re.compile(r'\.tex$')
repo_dircontent = [name for name in os.walk(os.path.join(datapath, reponame))]
paperpath = os.path.join(datapath, papername)
files = os.listdir(paperpath)
for fl in files:
    if re.search(file_regex, fl):
        print("found ", fl)
        paper_sentences = process_tex_file(paperpath+"/", fl)
        paper_dict = populate_sentence_dict(paper_sentences)
        repo = repo_dircontent[0][2]
        if reponame is not "":
            repo_dir = os.path.join(datapath, reponame)
        else:
            repo_dir = datapath + "/"
        for py_file in os.listdir(repo_dir):
            if re.match(r'.*\.py$', py_file):
                print(f"Parsing {py_file} from {repo_dir}...")
                code_dict = extract_functions_from_py_file(repo_dir, py_file)


#%%
def normalize_tensor(t):
        t_max = np.max(t)
        t_min = np.min(t)
        t = (t - t_min) / (t_max - t_min)
        return t
    
VEC_LEN = 512
def calc_tensor(data, text=True, is_2d=False):
    if text:
        tokenizer = text_tokenizer
        model = text_model
    else:
        tokenizer = code_tokenizer
        model = code_model

    tok = tokenizer(data, return_tensors="pt", padding=True, truncation=(not text or is_2d))
    emb = model(**tok, output_hidden_states=True)
    emb_allhidden = emb[2]
    emb_list = []
    for h in emb_allhidden:
        emb_list.append(h.detach().cpu().numpy())
    emb_np = np.asarray(emb_list)
    emb_avg = np.mean(emb_np, axis=0)
    emb_avg = normalize_tensor(emb_avg)
    if not is_2d:
        emb_avg = np.mean(emb_avg, axis=1)
    else:
        emb_avg = np.mean(emb_avg, axis=0)
        shape = np.shape(emb_avg)
        input_size = 768
        emb_padded = np.zeros((VEC_LEN,input_size))
        emb_padded[:shape[0],:shape[1]] = emb_avg
    return torch.from_numpy(emb_padded).float()

#%%
# encode all paper embeddings and save them
encoded_sentences = {}
for k, v in paper_dict.items():
    v_key = ' '.join(v)
    encoded_sentences[v_key] = calc_tensor(v, text=True, is_2d=True)


#%%
encoded_functions = {}
y = torch.Tensor(1)
STATE_SIZE = 128
#flat_list_paper = [item for sublist in encoded_sentences.values() for item in sublist]
flat_list_code = [item for sublist in code_dict.values() for item in sublist]


#%%
sim_max = 0
max_comment = ""
comment_list = []
cossim = nn.CosineSimilarity()
n = 0
for vc in flat_list_code:
    code_t = calc_tensor(vc, text=False, is_2d=True)
    for ks, vs in paper_dict.items():
        paper_t = calc_tensor(vs, text=True, is_2d=True)
        #print(paper_t.shape, code_t.shape)
        input = torch.cat((paper_t, code_t), dim=1)
        ae_model.forward(input)
        state = ae_model.get_state()
        split_state = torch.split(state, STATE_SIZE, 1)
        loss_cos_sim_enc = 1 - cossim(split_state[0], split_state[1]).mean()
        cur_sim = loss_cos_sim_enc.item()
        #print("sim: ", cur_sim)
        if len(ks) > 20:
            if cur_sim > sim_max:
                sim_max = cur_sim
                max_comment = ks
    comment_list.append(max_comment)
    n += 1
    sim_max = 0

print(max_comment)


#%%
sim_max = 0
comment_dict = {}
for kc, vc in code_dict.items():
    for func in vc:
        code_t = calc_tensor(func, text=False, is_2d=True)
        for ks, vs in paper_dict.items():
            # paper dict should have var: sentences
            for sentence in vs:
                paper_t = calc_tensor(sentence, text=True, is_2d=True)
                if ks == kc:
                    input = torch.cat((paper_t, code_t), dim=1)
                    ae_model.forward(input)
                    state = ae_model.get_state()
                    split_state = torch.split(state, STATE_SIZE, 1)
                    loss_cos_sim_enc = 1 - cossim(split_state[0], split_state[1]).mean()
                    cur_sim = loss_cos_sim_enc.item()
                    if cur_sim > sim_max:
                        sim_max = cur_sim
                        max_comment = sentence
        funcname = func[2]
        if funcname in comment_dict:
            entry = comment_dict[funcname]
            if entry:
                comment_dict[funcname] = entry.append(max_comment)
            else:
                comment_dict[funcname] = [max_comment]
        else:
            comment_dict[funcname] = [max_comment]
        sim_max = 0

#%%
