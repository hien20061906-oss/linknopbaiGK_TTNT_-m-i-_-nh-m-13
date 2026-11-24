"""Lấy hai file dữ liệu từ ZIP có sẵn hoặc tải mã khung chính thức."""
import argparse
import hashlib
import io
import urllib.request
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parent.parent
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--zip',type=Path,help='ZIP mã khung Berkeley hoặc ZIP Q4 có thư mục data')
args=parser.parse_args()
if args.zip:
    payload=args.zip.read_bytes()
else:
    url='https://inst.eecs.berkeley.edu/~cs188/fa25/assets/projects/machinelearning.zip'
    print('Downloading official starter data...')
    with urllib.request.urlopen(url,timeout=120) as r: payload=r.read()
with zipfile.ZipFile(io.BytesIO(payload)) as z:
    for filename in ['mnist.npz','lang_id.npz']:
        matches=[n for n in z.namelist() if n.replace('\\','/').endswith('/data/'+filename)]
        if len(matches)!=1: raise SystemExit('Cannot uniquely locate '+filename)
        content=z.read(matches[0])
        (ROOT/'data').mkdir(exist_ok=True)
        (ROOT/'data'/filename).write_bytes(content)
        print(filename,len(content),'bytes; SHA256:',hashlib.sha256(content).hexdigest())
