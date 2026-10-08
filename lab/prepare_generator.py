"""Explicitly prepare a small CPU model; never load remote custom code or tokens."""
import json
from pathlib import Path
from huggingface_hub import HfApi,snapshot_download
ROOT=Path(__file__).resolve().parents[1]
MODEL='HuggingFaceTB/SmolLM2-135M-Instruct'
def main():
    revision='12fd25f77366fa6b3b4b768ec3050bf629380bac'
    target=snapshot_download(MODEL,revision=revision,token=False,cache_dir=ROOT/'data/generator-model',
        allow_patterns=['*.json','*.txt','*.safetensors','*.model','*.jinja','README.md'])
    manifest={'model':MODEL,'revision':revision,'snapshot_path':str(Path(target).resolve()),
        'device':'cpu','trust_remote_code':False,'weights':'safetensors'}
    (ROOT/'data/generator-manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in manifest.items() if k!='snapshot_path'},indent=2))
if __name__=='__main__':main()
