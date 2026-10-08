"""Explicit safetensors-only download; never download inside an API request."""
import argparse,json
from pathlib import Path
from huggingface_hub import snapshot_download
ROOT=Path(__file__).resolve().parents[1]
MODELS={'compact':('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1','1427fd652930e4ba29e8149678df786c240d8825'),'bge':('BAAI/bge-reranker-v2-m3','953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e')}
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--model',choices=MODELS,default='compact');args=parser.parse_args()
    model,revision=MODELS[args.model]
    target=snapshot_download(model,revision=revision,token=False,cache_dir=ROOT/'data/models',allow_patterns=['*.json','*.txt','*.model','model.safetensors'])
    manifest={'model':model,'revision':revision,'snapshot_path':str(Path(target).resolve()),'device':'cpu','trust_remote_code':False,'weights':'safetensors'}
    (ROOT/'data/reranker-manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in manifest.items() if k!='snapshot_path'}))
if __name__=='__main__':main()
