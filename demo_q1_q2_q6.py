"""Chạy minh họa Q1, Q2, Q6 và lưu kết quả để đối chiếu khi vấn đáp."""
import argparse
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache_matplotlib'))
from cong_cu import backend
from models import Attention, PerceptronModel, RegressionModel
from train import train_perceptron, train_regression


def chay_mot_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    perceptron = PerceptronModel(3)
    points = backend.PerceptronDataset(perceptron)
    train_perceptron(perceptron, points)
    with torch.no_grad():
        predictions = [perceptron.get_prediction(sample['x']) for sample in points]
        new_point = torch.tensor([2., 1., 1.])  # Thành phần cuối là bias.
        q1 = {'luot_duyet': len(perceptron.training_history),
              'so_dung': int(np.count_nonzero(predictions == points.y.ravel())),
              'so_mau': len(points), 'diem_minh_hoa': perceptron(new_point).item(),
              'nhan_minh_hoa': perceptron.get_prediction(new_point)}

    q2 = thu_q2(seed, 0.003)
    return {'seed': seed, 'q1': q1, 'q2': q2}


def thu_q2(seed, learning_rate):
    torch.manual_seed(seed)
    model = RegressionModel()
    samples = backend.RegressionDataset(model)
    start = perf_counter()
    train_regression(model, samples, learning_rate=learning_rate)
    seconds = perf_counter() - start
    with torch.no_grad():
        x = torch.tensor([[-1.0], [0.0], [1.0]])
        grid = np.sort(samples.x[:, 0])
        midpoint = torch.tensor((grid[:-1] + grid[1:]) / 2, dtype=torch.float32)[:, None]
        mse_midpoint = (model(midpoint) - midpoint.sin()).square().mean().item()
        return {'learning_rate': learning_rate, 'epoch': len(model.training_history),
                'seconds': seconds, 'mse_train': model.training_history[-1],
                'mse_noi_suy': mse_midpoint,
                'minh_hoa': [{'x': a, 'sin_x': b, 'du_doan': c} for a, b, c in
                             zip(x.flatten().tolist(), x.sin().flatten().tolist(),
                                 model(x).flatten().tolist())]}


@torch.no_grad()
def minh_hoa_attention():
    model = Attention(layer_size=2, block_size=3)
    for layer in (model.q_layer, model.k_layer, model.v_layer):
        layer.weight.copy_(torch.eye(2))
        layer.bias.zero_()
    x = torch.tensor([[[1., 0.], [0., 1.], [1., 1.]]])
    changed = x.clone()
    changed[:, -1] += 10
    result = model(x)
    modified = model(changed)
    assert torch.allclose(result[:, :-1], modified[:, :-1])
    assert not torch.allclose(result[:, -1], modified[:, -1])
    return {'dau_vao': x[0].tolist(), 'dau_ra': result[0].tolist(),
            'dau_ra_sau_khi_doi_token_cuoi': modified[0].tolist(),
            'hai_vi_tri_dau_giu_nguyen': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 42, 73])
    args = parser.parse_args()
    backend.use_graphics = False
    torch.set_num_threads(2)
    runs = []
    for seed in args.seeds:
        result = chay_mot_seed(seed)
        runs.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    reference_seed = 42
    baseline = next((run['q2'] for run in runs if run['seed'] == reference_seed), None)
    comparison = [thu_q2(reference_seed, 0.001),
                  baseline if baseline is not None else thu_q2(reference_seed, 0.003),
                  thu_q2(reference_seed, 0.005)]
    result = {'lan_chay': runs, 'q2_so_sanh_seed': reference_seed,
              'q2_so_sanh_learning_rate': comparison, 'q6': minh_hoa_attention()}
    out = ROOT / 'ket_qua'
    out.mkdir(exist_ok=True)
    path = out / 'demo_q1_q2_q6.json'
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    for row in comparison:
        print(f"Q2 LR={row['learning_rate']} | epoch={row['epoch']} | "
              f"MSE train={row['mse_train']:.6f} | MSE nội suy={row['mse_noi_suy']:.6f}")
    print(json.dumps(result['q6'], ensure_ascii=False))
    print(f'Đã lưu {path}')


if __name__ == '__main__':
    main()
