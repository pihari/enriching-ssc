import json
import os
import sys
import git
import requests

def main_scraper():
    out_path = "~/data/s1/haritz/scraped"
    with open('links/links-between-papers-and-code.json') as f:
        data = json.load(f)
        for i in range(len(data)):
            url = data[i]['repo_url']
            aid = data[i]['paper_arxiv_id']
            if aid is not None and aid.startswith('1901'):
                path = os.path.join(out_path, "repos", aid)
                if not os.path.isdir(path):
                    print(f"Cloning {url} into {path}")
                    try:
                        git.Repo.clone_from(url, path)
                    except git.exc.GitError:
                        print(f'ERROR! {url} does not exist')
                tex_path = os.path.join(out_path, "1901 pwc", aid)
                if not os.path.isdir(tex_path):
                    print(f"Downloading {aid} from arXiv")
                    a_url = f"https://arxiv.org/e-print/{aid}"
                    r = requests.get(a_url, allow_redirects=True)
                    open(f'{aid}', 'wb').write(r.content)

if __name__=="__main__":
    main_scraper()
