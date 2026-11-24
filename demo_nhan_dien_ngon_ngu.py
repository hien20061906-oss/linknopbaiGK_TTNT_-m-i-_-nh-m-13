"""Dự đoán ngôn ngữ từ bằng trọng số Q4 đã huấn luyện trên máy."""
import argparse
from pathlib import Path
from unicodedata import normalize

import numpy as np
import torch

from models import LanguageIDModel

ROOT = Path(__file__).resolve().parent


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('words', nargs='*', default=['hello', 'hola', 'kiitos', 'paleis', 'cześć'])
    parser.add_argument('--weights', type=Path, default=ROOT / 'ket_qua/q4_seed42_weights.pt')
    args = parser.parse_args()
    data = ROOT / 'data/lang_id.npz'
    if not data.is_file():
        parser.error('Thiếu dữ liệu Q4. Chạy python cong_cu/chuan_bi_du_lieu.py.')
    if not args.weights.is_file():
        parser.error('Thiếu trọng số. Chạy python chay_kiem_thu.py q4 --seed 42 --phan-tich.')
    with np.load(data) as source:
        lookup = {char: index for index, char in enumerate(source['chars'])}
    model = LanguageIDModel()
    model.load_state_dict(torch.load(args.weights, map_location='cpu', weights_only=True))
    model.eval()
    for word in args.words:
        cleaned = normalize('NFC', word.lower().strip())
        invalid = sorted(set(cleaned) - lookup.keys())
        if not cleaned or invalid:
            print(f'{word!r} → Từ trống hoặc chứa ký tự ngoài bảng chữ: {invalid}')
            continue
        indices = torch.tensor([lookup[char] for char in cleaned])
        x = torch.nn.functional.one_hot(indices, num_classes=model.num_chars).float().unsqueeze(1)
        probabilities = model(x).softmax(dim=1)[0]
        best = probabilities.argmax().item()
        print(f'{word} → {model.languages[best]} (điểm softmax {probabilities[best]:.1%})')


if __name__ == '__main__':
    main()
