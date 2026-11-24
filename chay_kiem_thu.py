"""Chạy kiểm thử gốc và lưu kết quả cho Q1 đến Q6."""
import argparse
import json
import os
import re
import runpy
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'ket_qua'
OUT.mkdir(exist_ok=True)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache_matplotlib'))
sys.dont_write_bytecode = True

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('question', choices=['all','q1','q2','q3','q4','q5','q6'])
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--do-thi', action='store_true', help='Lưu và mở ảnh biểu đồ sau khi kiểm thử Q1–Q5')
parser.add_argument('--phan-tich', action='store_true',
                    help='Lưu ảnh và đánh giá bổ sung cho Q3–Q5')
args = parser.parse_args()

if args.question == 'all':
    scores = []
    for q in ['q1','q2','q3','q4','q5','q6']:
        command = [sys.executable, str(__file__), q, '--seed', str(args.seed)]
        if args.do_thi:
            command.append('--do-thi')
        if args.phan_tich:
            command.append('--phan-tich')
        result = subprocess.run(command,
                                cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding='utf-8', errors='replace')
        (OUT / f'{q}_seed{args.seed}.txt').write_text(result.stdout, encoding='utf-8')
        print(result.stdout, flush=True)
        match = re.search(r'### Question '+q+r': (\d+)/(\d+) ###', result.stdout)
        scores.append({'question': q, 'score': int(match[1]) if match else 0,
                       'maximum': int(match[2]) if match else [5,5,5,5,3,2][len(scores)],
                       'process_exit_code': result.returncode})
    summary = {'seed':args.seed, 'questions':scores,
               'total':sum(x['score'] for x in scores), 'maximum':25}
    (OUT / 'tong_hop_kiem_thu.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(f"TOTAL: {summary['total']}/25")
    raise SystemExit(0 if summary['total']==25 and all(x['process_exit_code']==0 for x in scores) else 1)

if args.question in ('q3','q4','q5'):
    name = 'lang_id.npz' if args.question=='q4' else 'mnist.npz'
    if not (ROOT/'data'/name).exists():
        raise SystemExit('Thiếu data/'+name+'. Chạy python cong_cu/chuan_bi_du_lieu.py trước.')

sys.path.insert(0,str(ROOT/'cong_cu'))
sys.path.insert(0,str(ROOT))
print('Đang nạp NumPy và PyTorch... Vui lòng chờ, không bấm Ctrl+C.', flush=True)
import numpy as np
import torch
import train
print(f'Đã nạp thư viện. Bắt đầu kiểm tra {args.question.upper()} với seed {args.seed}.', flush=True)

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.set_num_threads(2)
functions={'q1':'train_perceptron','q2':'train_regression','q3':'train_digitclassifier',
           'q4':'train_languageid','q5':'Train_DigitConvolution'}
visual = {}

if args.question in functions:
    original = getattr(train,functions[args.question])
    def record_training(model,dataset):
        start=time.perf_counter()
        original(model,dataset)
        record={'question':args.question,'seed':args.seed,'torch':torch.__version__,
                'numpy':np.__version__, 'seconds':time.perf_counter()-start,
                'history':getattr(model,'training_history',[])}
        (OUT/f'{args.question}_history.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
        if args.do_thi or (args.phan_tich and args.question in ('q3', 'q4', 'q5')):
            visual.update(model=model, dataset=dataset)
    setattr(train,functions[args.question],record_training)

sys.argv=['autograder.py','-q',args.question,'--no-graphics']
runpy.run_path(str(ROOT/'cong_cu/autograder.py'),run_name='__main__')
if args.phan_tich and args.question in ('q3', 'q4', 'q5') and visual:
    from phan_tich_ket_qua import luu_phan_tich
    luu_phan_tich(args.question, visual['model'], visual['dataset'], OUT, args.seed)
if args.do_thi:
    if visual:
        from ve_bieu_do import tao_bieu_do
        image_path = tao_bieu_do(args.question, visual['model'], visual['dataset'], OUT, args.seed)
        print(f'Đã lưu biểu đồ: {image_path}', flush=True)
        try:
            os.startfile(str(image_path))
        except (OSError, AttributeError) as error:
            print(f'Không tự mở được ảnh; hãy mở file trên. Chi tiết: {error}', flush=True)
    else:
        print('Q6 kiểm tra khối Attention, không có đường huấn luyện để vẽ.', flush=True)
