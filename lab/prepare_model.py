"""Explicit public model download; no inference service or paid GPU account."""
import json
from pathlib import Path
from huggingface_hub import HfApi,snapshot_download
ROOT=Path(__file__).resolve().parents[1]
MODEL='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'

def main():
    revision=HfApi(token=False).model_info(MODEL,token=False).sha
    target=snapshot_download(MODEL,revision=revision,token=False,cache_dir=ROOT/'data/models',
        allow_patterns=['*.json','*.txt','*.safetensors','README.md'])
    manifest={'model':MODEL,'revision':revision,'snapshot_path':str(Path(target).resolve()),
        'license':'apache-2.0','device':'cpu','trust_remote_code':False,'weights':'safetensors'}
    (ROOT/'data/model-manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in manifest.items() if k!='snapshot_path'},indent=2))
if __name__=='__main__':main()
