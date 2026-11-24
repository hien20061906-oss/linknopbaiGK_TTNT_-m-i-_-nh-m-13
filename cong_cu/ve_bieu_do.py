"""Vẽ kết quả sau khi kiểm thử, không thay đổi cách huấn luyện."""
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def tao_bieu_do(question, model, dataset, output, seed):
    plt.rcParams['font.family'] = 'Arial'
    history = model.training_history
    if question == 'q2':
        x = torch.tensor(dataset.x, dtype=torch.float32)
        model.eval()
        with torch.no_grad():
            predicted = model(x).detach().cpu().numpy().reshape(-1)
        order = np.argsort(dataset.x[:, 0])
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        axes[0].plot(dataset.x[order, 0], dataset.y[order, 0], label='Hàm sin thật', linewidth=2)
        axes[0].plot(dataset.x[order, 0], predicted[order], '--', label='Mô hình dự đoán', linewidth=2)
        axes[0].set(title='Q2: So sánh dự đoán và giá trị đúng', xlabel='x', ylabel='y')
        axes[0].legend()
        axes[1].semilogy(range(1, len(history)+1), history, label='MSE toàn tập')
        axes[1].axhline(0.02, color='red', linestyle='--', label='Ngưỡng đề: 0,02')
        axes[1].set(title=f'MSE cuối: {history[-1]:.6f}', xlabel='Epoch', ylabel='MSE')
        axes[1].legend()
        fig.suptitle(f'Kết quả Q2 vừa chạy, seed {seed} — dữ liệu huấn luyện')
    elif question == 'q1':
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.plot(range(1, len(history)+1), history, marker='o')
        ax.set(title=f'Q1: Số mẫu sai mỗi lượt — seed {seed}', xlabel='Lượt duyệt', ylabel='Số mẫu sai')
        ax.grid(alpha=0.2)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        epochs = [r['epoch'] for r in history]
        axes[0].plot(epochs, [r['loss'] for r in history], marker='o')
        axes[0].set(title='Loss huấn luyện trung bình', xlabel='Epoch', ylabel='Loss')
        axes[1].plot(epochs, [100*r['dev_accuracy'] for r in history], marker='o')
        axes[1].set(title='Độ chính xác trên dev', xlabel='Epoch', ylabel='Độ chính xác (%)')
        note = ' — dev trùng một phần train theo đề' if question == 'q5' else ''
        fig.suptitle(f'{question.upper()}: Kết quả vừa chạy, seed {seed}{note}')
    fig.tight_layout()
    path = output / f'{question}_seed{seed}_bieu_do.png'
    fig.savefig(path, dpi=160, bbox_inches='tight')
    plt.close(fig)
    return path
