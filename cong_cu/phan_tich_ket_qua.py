"""Lưu đánh giá sau khi đã chốt trọng số bằng tập dev."""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from backend import get_data_path
from ve_bieu_do import tao_bieu_do


@torch.no_grad()
def luu_phan_tich(question, model, dataset, output, seed):
    model.eval()
    prefix = output / f'{question}_seed{seed}'
    images = None
    if question == 'q4':
        _, predicted, labels = dataset._predict(split='test')
        predictions = np.array([item.item() for item in predicted])
        names = model.languages
        split = 'Test Q4'
        torch.save(model.state_dict(), prefix.with_name(prefix.name + '_weights.pt'))
    else:
        names = [str(i) for i in range(10)]
        if question == 'q3':
            images, labels = dataset.test_images, dataset.test_labels
            split = 'Test Q3 (MNIST)'
        else:
            # Đánh giá ngoài benchmark: Q5 chỉ học 200 ảnh từ train_images.
            with np.load(get_data_path('mnist.npz')) as data:
                images, labels = data['test_images'], data['test_labels']
            split = 'Tập test MNIST chính thức (Q5)'
        batches = torch.as_tensor(images, dtype=torch.float32).split(100)
        predictions = torch.cat([model(x).argmax(dim=1) for x in batches]).numpy()

    matrix = np.zeros((len(names), len(names)), dtype=int)
    np.add.at(matrix, (labels, predictions), 1)
    metrics = {'split': split, 'seed': seed, 'samples': len(labels),
               'correct': int(np.trace(matrix)), 'accuracy': float(np.trace(matrix) / len(labels)),
               'labels': names, 'confusion_matrix': matrix.tolist()}
    if question == 'q5':
        official = model(torch.as_tensor(dataset.test_images, dtype=torch.float32)).argmax(dim=1)
        metrics['berkeley_test_samples'] = len(dataset.test_labels)
        metrics['berkeley_test_accuracy'] = float(np.mean(official.numpy() == dataset.test_labels))
    prefix.with_suffix('.json').write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')

    fig, ax = plt.subplots(figsize=(7.5, 5.5), layout='constrained')
    palette = ax.imshow(matrix, cmap='Blues')
    fig.colorbar(palette, ax=ax, shrink=0.75)
    for (row, column), count in np.ndenumerate(matrix):
        ax.text(column, row, str(count), ha='center', va='center', fontsize=8,
                color='white' if count > matrix.max() / 2 else '#132f49')
    ax.set(xticks=range(len(names)), xticklabels=names,
           yticks=range(len(names)), yticklabels=names,
           xlabel='Dự đoán', ylabel='Nhãn thật',
           title=f'{split} • {metrics["accuracy"]:.2%} • {len(labels):,} mẫu')
    fig.savefig(prefix.with_name(prefix.name + '_ma_tran.png'), dpi=150)
    plt.close(fig)

    if images is not None:
        correct = np.flatnonzero(predictions == labels)[:6]
        incorrect = np.flatnonzero(predictions != labels)[:6]
        fig, axes = plt.subplots(2, 6, figsize=(9, 3.4), layout='constrained')
        for row, indices, category in zip(axes, [correct, incorrect], ['Đúng', 'Sai']):
            for ax in row:
                ax.axis('off')
            for ax, index in zip(row, indices):
                ax.imshow(images[index].reshape(28, 28), cmap='gray')
                ax.set_title(f'{category} · {labels[index]} → {predictions[index]}', fontsize=9)
        fig.suptitle(f'{split} • Nhãn thật → dự đoán')
        fig.savefig(prefix.with_name(prefix.name + '_vi_du.png'), dpi=150)
        plt.close(fig)

    tao_bieu_do(question, model, dataset, output, seed)
    print(f'{split}: {metrics["correct"]}/{len(labels)} = {metrics["accuracy"]:.2%}')
