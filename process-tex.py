
#%%
#%%
import os
import re
from builtins import UnicodeDecodeError, bytes

# Testing LaTeX parser
from tex2py import tex2py
# Testing TexSoup
from TexSoup import TexSoup


def search_tex_command(line, command, env=None):
    '''
    Searches a line of text for a LaTeX command with an optional env variable via regex.
    \nExample: search_tex_command(line, "begin", "equation") looks for occurences of \\begin{equation}
    \nExample: search_tex_command(line, "section") looks for occurences of \\section{*}
    '''
    if env:
        _texenv = re.compile(rf"\\{command}{{{env}(.*?)}}")
    else:
        _texenv = re.compile(rf"\\{command}{{(.*?)({{(.*?)}})*(.*?)}}")

    return re.match(_texenv, line)

def compile_retex(t):
    cmd, env = t
    if env:
        _retex = re.compile(rf"\\{cmd}{{{env}}}")
    else:
        _retex = re.compile(rf"\\{cmd}{{(.*?)({{(.*?)}})*(.*?)}}")
    return _retex

def replace_retex(tlist, s):
    for t in tlist:
        _cmd, _ = t
        _retex = re.compile(rf"\\{_cmd}{{(.*?)({{(.*?)}})*(.*?)}}")
        _match = re.search(_retex, s)
        if _match:
            print(_match.group())
            s = re.sub(_retex, _match.group(1), s)
    return s


filename = "yang2020www.tex"
folder_path = "HATCH/arXiv/2004.01136/"
f_path = os.path.join(os.getcwd(), folder_path)
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

#%%
# Weird boolean matching
with open(f_path+filename) as f:
    for line in f:
        bs = [re.match(compile_retex(e), line)!=None for e in filter_list]
        if any(bs):
            continue
        print(line)

#%%
# Extract all potentially usable sentences
sentences = []
with open(f_path+filename) as f:
    in_cmd = False
    for line in f:
        any_match = False
        for e in filter_list:
            if re.match(compile_retex(e), line):
                any_match = True
                # Also filter out lines that are in between \begin{..} and \end{..}
                cmd, _ = e
                in_cmd = True if cmd is "begin" else False

        if any_match or in_cmd or line is "\n" or line.startswith("%"):
            print(line)
        else:
            sentences.append(line)
            

#%%
# Filter out all non-alphanumeric sentences
sentences = []
with open(f_path+filename) as f:
    for line in f:
        if not re.search(r'^\W+$', line) or re.search(r'^\w+', line):
            sentences.append(line)

#%%
# 1. Split lines into sentences
# 2a. Remove comments, latex environments, algorithms, ...
# 2b. Keep variables in $..$ envs
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

#%%
def write_list_to_file(l, file, mode="w"):
    txt_file = open(file, mode)
    for e in l:
        txt_file.write(e+"\n")
    txt_file.close()


#%%
dirpath = "1901 pwc"
dircontent = [name for name in os.walk(os.path.join(os.getcwd(), dirpath))]
datadirs = []
outfile = "corpus.txt"
for e in dircontent[0][1]:
    # format: YYMM.#####
    if re.match(r'(\d){4}\.(\d)+', e):
        datadirs.append(e)
for subdir in datadirs:
    cur_dir = os.path.join(os.getcwd(), dirpath+"/"+subdir)
    files = os.listdir(cur_dir)
    for fl in files:
        if re.search('main\.tex$', fl):
            sentences = process_tex_file(cur_dir+"/", fl)
            write_list_to_file(sentences, outfile, "a")
    write_list_to_file(["\n"], outfile, "a")
            


#%%
# Extract only sentences with variables in it
# filter useless lines
line_filter = [
    "",
    "$$\n",
    "\n",
    " \n"
]
replace_list = [
    ("textit", None),
    ("textbf", None)
    ]
variables_ctx = []
with open(f_path+filename) as f:
    for line in f:
        if re.match(r"^.*\$(.*)\$.*$",  line):
            # split only at single . (sentence delimeter)
            # also prevents splitting numbers like 1.5
            split_line = re.split(r"(?<![\d\.])\.(?![\d\.])", line)
            for s in split_line:
                if s not in line_filter:
                    s = replace_retex(replace_list, s)
                    # TODO: test replace_retex
                    variables_ctx.append(s)

#%%
txt_file = open("varctx.txt","w")
for v in variables_ctx:
    txt_file.write(v+".\n")
txt_file.close()

with open(f_path+filename) as f:
    data = f.read()
toc = tex2py(data)

#%%
filename = "varctx.txt"
with open(filename) as f:
    for line in f:
        vs = ["alpha"]
        for v in vs:
            _revar = re.compile(rf'(.*?){v}(.*?)')
            if re.match(_revar, line):
                print(line)

#TODO: now align variables from code AST and these to create pairs


#%%
soup = TexSoup(open(f_path+filename))
cmd_filter = [
    "textit", 
    "textbf"
    ]
for cmd in cmd_filter:
    cmd_list = list(soup.find_all(cmd))
    for i in range(len(cmd_list)):
        e = cmd_list[i]
        #if e:
            #attr = getattr(soup, cmd)
            #e.replace_with(e.string)
            # FIXME: AttributeError: 'NoneType' object has no attribute 'replace_child'

#%%
# TODO: explore TexSoup => how to replace strings in the tree?
# TODO: further modify extracted sentences to proper NL
# TODO: maybe replace stuff like "\{alpha}" with "alpha" for training SciBERT
