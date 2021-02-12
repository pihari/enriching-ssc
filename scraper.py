
#%%
import json
import os
#%%
# writes the list of arXiv ids to a file
import sys

import git

with open('links/links-between-papers-and-code.json') as f:
  data = json.load(f)
  '''
  paper_url
  paper_title
  paper_arxiv_id
  paper_url_abs
  paper_url_pdf
  repo_url
  mentioned_in_paper
  mentioned_in_github
  framework
  '''

  aidlist = []
  count = {
    10: 0,
    11: 0,
    12: 0,
    13: 0,
    14: 0,
    15: 0,
    16: 0,
    17: 0,
    18: 0,
    19: 0,
    20: 0,
    2001: 0,
    2002: 0
          }

  for i in range(len(data)):
    aid = data[i]['paper_arxiv_id']
    for k, v in count.items():
      if aid is not None and aid.startswith(str(k)):
        count[k] = v+1
    
    if aid is not None:
      aidlist.append(aid)

  print(count)

original_stdout = sys.stdout
with open('aidlist.txt', 'w') as f:
  sys.stdout = f # Change the standard output to the file we created.
  for e in sorted(aidlist):
    print(e)
  sys.stdout = original_stdout # Reset the standard output to its original value


#%%
# go through all files and get list of filenames without endings
# remove .gz extension from all files in the process
path = os.path.join(os.getcwd(), "papers")
files = []
ext = ".gz"
for dirpath, dirnames, filenames in os.walk(path):
  for file in filenames:
    cur_file = os.path.join(dirpath, file)
    if file.endswith(ext):
      file_noext = os.path.splitext(file)[:-1][0]
      files += file_noext
      fpath = os.path.join(os.getcwd(), "gz")
      # renaming destroys file, move is better
      os.rename(cur_file, os.path.join(fpath, file))

#%%
# remove files that are not on the arXiv id list
foldername = "xxxx"
fpath = os.path.join(os.getcwd(), foldername)
for file in os.listdir(fpath):
  file_noext = os.path.splitext(file)[:-1][0]
  if file_noext not in aidlist:
    os.remove(os.path.join(fpath, file))


#%%
# remove files with gz ending from folder with unpack files
import os
foldername = "1901 pwc"
path = os.path.join(os.getcwd(), foldername)
for file in os.listdir(path):
  if file.endswith(".gz"):
    os.remove(os.path.join(path, file))

#%%
# clone all repos
import json
import git
with open('links/links-between-papers-and-code.json') as f:
  data = json.load(f)
  for i in range(len(data)):
    url = data[i]['repo_url']
    aid = data[i]['paper_arxiv_id']
    if aid is not None and aid.startswith('1901'):
      path = os.path.join("repos", aid)
      if not os.path.isdir(path):
        print(f"Cloning {url} into {path}")
        try:
          git.Repo.clone_from(url, path)
        except git.exc.GitError:
          print(f'ERROR! {url} does not exist')
          


#%%
