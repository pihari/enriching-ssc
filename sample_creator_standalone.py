
import re
# 1. Split lines into sentences
# 2a. Remove comments, latex environments, algorithms, ...
# 2b. Keep variables in $..$ envs

filter_list_standalone = ["section", "subsection", "subfigure"]
filter_list_enclosed = ["figure", "equation", "table"]
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
    ("city", None)  #FIXME: \\city{...} still shows
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
                    # TODO: double dot at sentence end
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
        var = split_expr[1] # FIXME: only gets first $..$ context
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

def extract_all_lines_with_variable(file, var):
    lines_with_var = []
    with open(file) as f:
        for line in f:
            _revar = re.compile(rf'(.*?){var}(.*?)')
            if re.match(_revar, line):
                lines_with_var.append(line)
    return lines_with_var

from transformers import *

def main_scs():
    text_tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_cased")
    text_model = AutoModel.from_pretrained("allenai/scibert_scivocab_cased")
    data_path = "/data/s1/haritz/scraped"
    paper_dirpath = "1901 pwc"
    paper_dircontent = [name for name in os.walk(os.path.join(data_path, paper_dirpath))]
    repo_dirpath = "repos"
    repo_dircontent = [name for name in os.walk(os.path.join(data_path, repo_dirpath))]
    samples = []
    datadirs = []
    for e in paper_dircontent[0][1]:
        # folder name format: YYMM.#####
        if re.match(r'(\d){4}\.(\d)+', e):
            datadirs.append(e)
    cur_dir_counter = 1
    n_dirs = len(datadirs)
    for subdir in datadirs:
        print(f"Parsing directory {cur_dir_counter} / {n_dirs}", end="\r")
        cur_dir_counter += 1
        cur_dir = os.path.join(data_path, paper_dirpath+"/"+subdir)
        files = os.listdir(cur_dir)
        for fl in files:
            if re.search(r'main\.tex$', fl):
                paper_sentences = process_tex_file(cur_dir+"/", fl)
                paper_dict = populate_sentence_dict(paper_sentences)
                # find corresponding repo
                for repo in repo_dircontent[0][1]:
                    repo_dir = os.path.join(data_path, repo_dirpath+"/"+repo+"/")
                    for py_file in os.listdir(repo_dir):
                        if re.match(r'.*\.py$', py_file):
                            #print(f"Parsing {py_file} from {repo_dir}...")
                            code_dict = traverse_py_file(repo_dir, py_file)
                            if code_dict: # None if tree can not be parsed
                                sample = generate_positive_samples(paper_dict, code_dict)
                                for e in sample:
                                    samples.append(e)
    # load code model
    code_tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    code_model = AutoModel.from_pretrained("microsoft/codebert-base")

    # sample always in form (sentence, code tokens, label)
    # loop through samples
    # tokenize text with text_tokenizer
    # code is already tokenized
    embeddings = []
    counter = 0
    n_samp = len(samples)
    for ti, ci, li in samples:
        ti_tokens = text_tokenizer(ti, return_tensors="pt", padding=True)
        ti_emb = text_model(**ti_tokens)
        ci_tokens = code_tokenizer(ci, return_tensors="pt", padding=True, truncation=True)
        ci_emb = code_model(**ci_tokens)
        enc_sample = (ti_emb, ci_emb, li)
        counter += 1
        print(f"Encoding samples {counter}/{n_samp}", end="\r")
        embeddings.append(enc_sample)
    print("Total samples created: ", counter)


from ast_traverser import *
def traverse_py_file(path, filename):
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

def generate_positive_samples(sdict, vdict):
    all_vars_in_code = [k for k,v in vdict.items()]
    samples = []
    for var in all_vars_in_code:
        if var in sdict:
            #print(f"Matched variable {var}!")
            for sk, sv in sdict.items():
                if sk in vdict and len(vdict[sk]) > 0:
                    for i in range(len(sv)):
                        samples.append((sdict[sk][i], vdict[sk][0], "1"))
                        # this is stupid, index needs fixing
                        # FIXME: apparently problem is not sdict but empty vdict!
    return samples 


def tokenize_string(s, tokenizer):
    tokens = []
    word_list  = s.split(' ')
    for word in word_list:
        token = tokenizer.tokenize(word)
        tokens.extend(token)
    return tokens

if __name__ == "__main__":
    main_scs()

