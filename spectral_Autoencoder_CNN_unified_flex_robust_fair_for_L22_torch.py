#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CNN频域自编码器 - PyTorch版本，支持断点续训功能
"""

import numpy as np, pickle, time, pandas as pd, matplotlib
import os, argparse
matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from matplotlib.colors import ListedColormap
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import random
import math


# ─── 模型 ─────────────────────────────────────────────────────────────────────

class FlexibleCNNAE(nn.Module):
    """
    通用自适应 CNN 自编码器
    input_shape: TF 格式 (spatial_dim, freq_dim, in_channels)
    前向传播期望输入: (B, in_channels, spatial_dim, freq_dim)
    """
    def __init__(self, input_shape, spatial_target=8, freq_target=None,
                 filters_base=64, bottleneck_channels=None,
                 use_bilinear_upsample=True):
        super().__init__()
        spatial_dim, freq_dim, in_channels = input_shape

        assert 0 < spatial_target <= spatial_dim
        if freq_target is not None:
            assert 0 < freq_target <= freq_dim

        spatial_steps = (0 if spatial_dim <= spatial_target
                         else math.ceil(math.log2(spatial_dim / spatial_target)))
        freq_steps = (0 if (freq_target is None or freq_dim <= freq_target)
                      else math.ceil(math.log2(freq_dim / freq_target)))
        total_steps = max(spatial_steps, freq_steps)

        print(f"[INFO] 空间维度步数: {spatial_steps} | 频率维度步数: {freq_steps} | 总步数: {total_steps}")

        self.spatial_dim   = spatial_dim
        self.freq_dim      = freq_dim
        self.in_channels   = in_channels
        self.spatial_steps = spatial_steps
        self.freq_steps    = freq_steps
        self.total_steps   = total_steps
        self.use_bilinear  = use_bilinear_upsample

        if bottleneck_channels is None:
            bottleneck_channels = in_channels
        self.bottleneck_channels = bottleneck_channels

        # 预计算每步编码器下采样后的空间尺寸
        # PyTorch Conv2d(padding=1, kernel=3, stride=s) → ceil(H/s)
        h, w = spatial_dim, freq_dim
        enc_shapes = []
        for step in range(total_steps):
            ss = 2 if step < spatial_steps else 1
            fs = 2 if step < freq_steps    else 1
            h = math.ceil(h / ss)
            w = math.ceil(w / fs)
            enc_shapes.append((h, w))
        self.enc_shapes = enc_shapes

        # 编码器自适应 resize 目标尺寸
        if total_steps > 0:
            t_h = spatial_target if spatial_steps > 0 else enc_shapes[-1][0]
            t_w = (freq_target
                   if (freq_steps > 0 and freq_target is not None)
                   else enc_shapes[-1][1])
        else:
            t_h, t_w = spatial_dim, freq_dim
        self.enc_target_hw = (t_h, t_w)

        def cbr(in_ch, out_ch, stride=(1, 1)):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        # ── 编码器 ──────────────────────────────────────────────────────────
        self.enc_stem  = cbr(in_channels, filters_base)
        self.enc_drop0 = nn.Dropout(0.1)

        enc_downs, enc_drops = [], []
        prev_ch = filters_base
        for step in range(total_steps):
            filters = filters_base * (2 ** step)
            ss = 2 if step < spatial_steps else 1
            fs = 2 if step < freq_steps    else 1
            enc_downs.append(cbr(prev_ch, filters, stride=(ss, fs)))
            enc_drops.append(nn.Dropout(0.1))
            prev_ch = filters
        self.enc_downs = nn.ModuleList(enc_downs)
        self.enc_drops = nn.ModuleList(enc_drops)

        # 瓶颈层（线性激活）
        self.encoded_conv = nn.Conv2d(prev_ch, bottleneck_channels, 3, padding=1, bias=True)

        # ── 解码器 ──────────────────────────────────────────────────────────
        dec_blocks, dec_drops = [], []
        prev_ch = bottleneck_channels
        for step in range(total_steps - 1, -1, -1):
            filters = filters_base * (2 ** step) if step > 0 else filters_base
            dec_blocks.append(cbr(prev_ch, filters))
            dec_drops.append(nn.Dropout(0.1))
            prev_ch = filters
        self.dec_blocks = nn.ModuleList(dec_blocks)
        self.dec_drops  = nn.ModuleList(dec_drops)

        self.dec_head    = cbr(prev_ch, filters_base // 2)
        self.output_conv = nn.Conv2d(filters_base // 2, in_channels, 3, padding=1)

        print(f"[INFO] 编码器输出形状: ({t_h}, {t_w}, {bottleneck_channels})")

    def _resize(self, x, hw):
        if x.shape[2] == hw[0] and x.shape[3] == hw[1]:
            return x
        mode = 'bilinear' if self.use_bilinear else 'nearest'
        kw = {'align_corners': False} if mode == 'bilinear' else {}
        return F.interpolate(x, size=(int(hw[0]), int(hw[1])), mode=mode, **kw)

    def encode(self, x):
        x = self.enc_drop0(self.enc_stem(x))
        for i in range(self.total_steps):
            x = self.enc_drops[i](self.enc_downs[i](x))
        x = self._resize(x, self.enc_target_hw)
        return self.encoded_conv(x)

    def decode(self, x):
        x = F.relu(x)
        if self.total_steps > 0:
            x = self._resize(x, self.enc_shapes[-1])

        mode = 'bilinear' if self.use_bilinear else 'nearest'
        kw = {'align_corners': False} if mode == 'bilinear' else {}

        for i, step in enumerate(range(self.total_steps - 1, -1, -1)):
            ss = 2 if step < self.spatial_steps else 1
            fs = 2 if step < self.freq_steps    else 1
            if ss > 1 or fs > 1:
                x = F.interpolate(
                    x, size=(x.shape[2] * ss, x.shape[3] * fs),
                    mode=mode, **kw
                )
            x = self.dec_drops[i](self.dec_blocks[i](x))
            if step < len(self.enc_shapes):
                x = self._resize(x, self.enc_shapes[step])

        x = self._resize(x, (self.spatial_dim, self.freq_dim))
        x = self.dec_head(x)
        return self.output_conv(x)

    def forward(self, x):
        return self.decode(self.encode(x))


def count_params(model):
    return sum(p.numel() for p in model.parameters())


# ─── 数据格式转换 ──────────────────────────────────────────────────────────────

def to_tensor(data_numpy):
    """(N, H, W, C) numpy → (N, C, H, W) float32 tensor"""
    return torch.from_numpy(data_numpy.astype(np.float32)).permute(0, 3, 1, 2)

def from_tensor(tensor):
    """(N, C, H, W) tensor → (N, H, W, C) numpy"""
    return tensor.detach().cpu().permute(0, 2, 3, 1).numpy()


# ─── 训练工具 ─────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total = 0.0
    for (xb,) in loader:
        xb = xb.to(device)
        loss = F.mse_loss(model(xb), xb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item() * len(xb)
    return total / len(loader.dataset)

def eval_loss(model, loader, device):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device)
            total += F.mse_loss(model(xb), xb).item() * len(xb)
    return total / len(loader.dataset)

def predict_model(model, tensor, batch_size=32, device='cpu'):
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(tensor), batch_size):
            outs.append(model(tensor[i:i + batch_size].to(device)).cpu())
    return torch.cat(outs, dim=0)


# ─── Plot utils ───────────────────────────────────────────────────────────────

def plot_difference(t_true, u_true, t_pred, u_pred, fn):
    N = u_true.shape[0]
    colors  = sns.diverging_palette(240, 10, n=9,  as_cmap=True)
    colors2 = sns.diverging_palette(240, 10, n=41)
    colors2 = ListedColormap(colors2[20:])
    fig, ax = plt.subplots(4, 1, figsize=(8, 8), dpi=400,
                           sharex=True, sharey=False,
                           gridspec_kw=dict(hspace=0.05))
    ax[0].pcolormesh(t_true, np.linspace(-11, 11, N), u_true,
                     shading="gouraud", cmap=colors, vmin=-3, vmax=3)
    ax[1].pcolormesh(t_pred, np.linspace(-11, 11, N), u_pred,
                     shading="gouraud", cmap=colors, vmin=-3, vmax=3)
    ax[2].pcolormesh(t_pred, np.linspace(-11, 11, N), np.abs(u_pred - u_true),
                     shading="gouraud", cmap=colors2, vmin=0, vmax=3)
    ax[3].plot(t_pred, np.linalg.norm(u_pred - u_true, axis=0))
    ax[0].set_ylabel(r'$x$'); ax[1].set_ylabel(r'$x$'); ax[2].set_ylabel(r'$x$')
    ax[3].set(ylabel=r'$||u-\tilde{u}||$', xlabel=r'$t$')
    plt.savefig(fn, bbox_inches="tight"); plt.close()

def plot_history(hist, fn, resume_epoch=0):
    plt.figure(figsize=(10, 6))
    epochs = hist["epoch"].to_numpy() + resume_epoch + 1
    plt.semilogy(epochs, hist["loss"].to_numpy(), label="train")
    if "val_loss" in hist:
        plt.semilogy(epochs, hist["val_loss"].to_numpy(), label="val")
    plt.legend()
    plt.xlabel("epoch"); plt.ylabel("loss")
    plt.title("Training History")
    plt.grid(True, alpha=0.3)
    if resume_epoch > 0:
        plt.axvline(x=resume_epoch, color='red', linestyle='--', alpha=0.7,
                    label=f'Resume from epoch {resume_epoch}')
        plt.legend()
    plt.tight_layout()
    plt.savefig(fn, dpi=150); plt.close()


# ─── FFT 工具 ─────────────────────────────────────────────────────────────────

def complex_to_real(complex_data):
    return np.concatenate([np.real(complex_data), np.imag(complex_data)], axis=-1)

def real_to_complex(real_data):
    mid = real_data.shape[-1] // 2
    return real_data[..., :mid] + 1j * real_data[..., mid:]

def fft_analysis_blocks(signal, Nblock, overlap):
    step = int(Nblock * (1 - overlap))
    window = np.hanning(Nblock)
    if signal.ndim == 2:
        n_space, n_time = signal.shape
    else:
        n_space, n_time = 1, len(signal)
        signal = signal.reshape(1, -1)
    pad_start  = Nblock // 2
    pad_end    = Nblock // 2
    signal_pad = np.pad(signal, ((0, 0), (pad_start, pad_end)), mode='edge')
    first_center = pad_start
    last_center  = pad_start + n_time - 1
    n_blocks = int(np.ceil((last_center - first_center) / step)) + 1
    fft_blocks, block_info = [], []
    for i in range(n_blocks):
        center = first_center + i * step
        start  = center - Nblock // 2
        end    = start + Nblock
        if start >= 0 and end <= signal_pad.shape[1]:
            block = signal_pad[:, start:end] * window[None, :]
            fft_blocks.append(np.fft.rfft(block, axis=1))
            block_info.append((start, end, center))
    return (np.array(fft_blocks), block_info,
            signal_pad.shape[1], (pad_start, pad_end, n_time))

def fft_synthesis_blocks(fft_blocks, block_info, padded_length, padding_info,
                         Nblock, overlap):
    window = np.hanning(Nblock)
    pad_start, pad_end, original_length = padding_info
    n_blocks, n_space, _ = fft_blocks.shape
    output     = np.zeros((n_space, padded_length))
    window_sum = np.zeros(padded_length)
    for fft_block, (start, end, _) in zip(fft_blocks, block_info):
        time_block = np.fft.irfft(fft_block, n=Nblock, axis=1)
        output[:, start:end]  += time_block
        window_sum[start:end] += window
    output = output / np.maximum(window_sum[None, :], 1e-15)
    return output[:, pad_start:pad_start + original_length]

def prepare_cnn_data(fft_blocks_real, spatial_dim, freq_dim):
    n_blocks, n_space, freq_features = fft_blocks_real.shape
    return fft_blocks_real.reshape(n_blocks, spatial_dim, freq_dim, 2)

def reshape_cnn_output(cnn_output):
    n_blocks, spatial_dim, freq_dim, channels = cnn_output.shape
    return cnn_output.reshape(n_blocks, spatial_dim, freq_dim * channels)


# ─── 主流程 ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    t_init = time.time()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    parser = argparse.ArgumentParser(description='CNN频域自编码器训练')
    parser.add_argument('--trunc',          type=int,  default=8)
    parser.add_argument('--epochs',         type=int,  default=100)
    parser.add_argument('--filters_base',   type=int,  default=64)
    parser.add_argument('--checkpoint_dir', type=str,
                        default='./checkpoints_cnn_spatial_a1_20_robust_L22')
    parser.add_argument('--model_type',     type=str,  default='basic',
                        choices=['basic', 'balanced', 'progressive'])
    parser.add_argument('--freq_reduction', default=False)
    parser.add_argument('--freq_target',    type=int,  default=11)
    args = parser.parse_args()

    eps    = 1e-15
    EPOCHS = args.epochs

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    trunc_list  = [8]
    Nblock_list = [120]

    print("=== 调参设置 ===")
    print(f"trunc 列表: {trunc_list}")
    print(f"Nblock 列表: {Nblock_list}")
    print(f"共 {len(trunc_list) * len(Nblock_list)} 组组合。")

    # ── 数据加载 ──────────────────────────────────────────────────────────────
    print("\n=== 数据加载 ===")
    u_true = pickle.load(open("kse_solution_L22_N64_steps400000_warm5000_h0.25.pkl", "rb"))
    num1, num2 = 100_00, 11_000
    u_train, u_test = u_true[:num1], u_true[num1:num2]
    u_train, u_test = u_train.T, u_test.T

    u_mean = u_train.mean(1, keepdims=True)
    u_std  = u_train.std(1,  keepdims=True) + eps
    u_train_norm = (u_train - u_mean) / u_std
    u_test_norm  = (u_test  - u_mean) / u_std

    print(f"数据形状: 训练集 {u_train.shape}, 测试集 {u_test.shape}")

    overlap     = 0.5
    all_results = []

    for trunc in trunc_list:
        for Nblock in Nblock_list:
            freq_target = Nblock // 2 + 1

            print("\n" + "=" * 60)
            print(f"开始实验: trunc={trunc}, Nblock={Nblock}")
            print("=" * 60)

            # ── FFT 分析 ───────────────────────────────────────────────────
            print("生成训练频域数据...")
            fft_train, train_block_info, train_padded_len, train_padding_info = \
                fft_analysis_blocks(u_train_norm, Nblock, overlap)
            print("生成测试频域数据...")
            fft_test, test_block_info, test_padded_len, test_padding_info = \
                fft_analysis_blocks(u_test_norm, Nblock, overlap)

            n_blocks_train, n_space, n_freq = fft_train.shape
            print(f"频域训练数据: {fft_train.shape}")
            print(f"频域测试数据:  {fft_test.shape}")

            # ── CNN 格式 ───────────────────────────────────────────────────
            train_data_cnn = prepare_cnn_data(complex_to_real(fft_train), n_space, n_freq)
            test_data_cnn  = prepare_cnn_data(complex_to_real(fft_test),  n_space, n_freq)
            print(f"CNN 输入形状: 训练 {train_data_cnn.shape}, 测试 {test_data_cnn.shape}")

            train_tensor = to_tensor(train_data_cnn)
            test_tensor  = to_tensor(test_data_cnn)

            train_loader = DataLoader(TensorDataset(train_tensor),
                                      batch_size=32, shuffle=True,  num_workers=0)
            val_loader   = DataLoader(TensorDataset(test_tensor),
                                      batch_size=32, shuffle=False, num_workers=0)

            # ── 构建模型 ───────────────────────────────────────────────────
            input_shape = train_data_cnn.shape[1:]   # (spatial_dim, freq_dim, 2)
            model = FlexibleCNNAE(
                input_shape=input_shape,
                spatial_target=trunc,
                freq_target=freq_target,
                filters_base=args.filters_base,
            ).to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            # 每 500 epoch LR 降一个数量级（与原 TF scheduler 等价）
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=lambda epoch: 0.1 ** (epoch // 500)
            )

            print(f"\n=== 开始训练 ===")
            print(f"目标 epoch: {EPOCHS}")

            history = {'loss': [], 'val_loss': [], 'epoch': []}
            t0 = time.time()

            for epoch in range(EPOCHS):
                tr_loss  = train_one_epoch(model, train_loader, optimizer, device)
                val_loss = eval_loss(model, val_loader, device)
                scheduler.step()

                history['loss'].append(tr_loss)
                history['val_loss'].append(val_loss)
                history['epoch'].append(epoch)

                print(f"Epoch {epoch+1:04d} | loss={tr_loss:.3e} | val_loss={val_loss:.3e}")

            train_time = (time.time() - t0) / 60
            print(f"训练完成，耗时 {train_time:.1f} 分钟")

            # ── 保存 ───────────────────────────────────────────────────────
            checkpoint_dir = os.path.join(
                args.checkpoint_dir,
                f"trunc{trunc}_Nblock{Nblock}_freq{freq_target}"
            )
            os.makedirs(checkpoint_dir, exist_ok=True)

            hist_df = pd.DataFrame(history)
            hist_df.to_csv(os.path.join(checkpoint_dir, "training_history.csv"), index=False)
            plot_history(hist_df,
                         os.path.join(checkpoint_dir,
                                      f"training_curve_cnn_trunc{trunc}_Nblock{Nblock}.png"))

            final_model_path = os.path.join(
                checkpoint_dir, f"final_cnn_model_trunc{trunc}_Nblock{Nblock}.pt"
            )
            torch.save(model.state_dict(), final_model_path)
            print(f"✓ 最终模型已保存: {final_model_path}")

            # ── 预测与重构 ─────────────────────────────────────────────────
            print("\n=== 预测和重构 ===")
            recon_tensor   = predict_model(model, test_tensor, batch_size=32, device=device)
            recon_data_cnn = from_tensor(recon_tensor)        # (N, H, W, C)

            recon_complex = real_to_complex(reshape_cnn_output(recon_data_cnn))

            print("执行 FFT 合成...")
            pred_u_norm = fft_synthesis_blocks(
                recon_complex, test_block_info, test_padded_len,
                test_padding_info, Nblock, overlap
            )
            pred_u = pred_u_norm * u_std + u_mean

            mse       = np.mean((u_test - pred_u) ** 2)
            max_error = np.max(np.abs(u_test - pred_u))
            rel_error = np.sqrt(mse) / np.std(u_test)

            print(f"\n=== 结果统计 (trunc={trunc}, Nblock={Nblock}) ===")
            print(f"最大绝对误差 = {max_error:.3e}")
            print(f"MSE = {mse:.3e}")
            print(f"相对误差 = {rel_error:.3%}")

            # ── FFT 重构精度验证 ───────────────────────────────────────────
            print(f"\n=== FFT 重构精度验证 ===")
            perfect_recon_norm = fft_synthesis_blocks(
                fft_test, test_block_info, test_padded_len,
                test_padding_info, Nblock, overlap
            )
            perfect_recon = perfect_recon_norm * u_std + u_mean
            fft_error = np.max(np.abs(u_test - perfect_recon))
            print(f"FFT 完美重构误差 = {fft_error:.3e}")
            if fft_error < 1e-12:
                print("✓ FFT 重构算法正确，达到机器精度")
            else:
                print("⚠ FFT 重构算法可能有问题")

            ae_only_error = np.max(np.abs(perfect_recon - pred_u))
            print(f"CNN 自编码器纯重构误差 = {ae_only_error:.3e}")

            # ── 保存结果 ───────────────────────────────────────────────────
            np.savez(os.path.join(checkpoint_dir, "normalization_params.npz"),
                     u_mean=u_mean, u_std=u_std)

            dt     = 0.25
            t      = np.arange(0, len(u_true)) * dt
            t_test = t[num1:num2]

            plot_start, plot_end = 0, min(400, len(t_test))
            plot_difference(
                t_test[plot_start:plot_end], u_test[:, plot_start:plot_end],
                t_test[plot_start:plot_end], pred_u[:, plot_start:plot_end],
                os.path.join(checkpoint_dir, "cnn_spectral_ae_comparison.png")
            )

            final_time = (time.time() - t_init) / 60
            results = {
                'max_error':             max_error,
                'mse':                   mse,
                'rel_error':             rel_error,
                'fft_error':             fft_error,
                'ae_only_error':         ae_only_error,
                'trunc':                 trunc,
                'Nblock':                Nblock,
                'filters_base':          args.filters_base,
                'final_epoch':           EPOCHS,
                'total_epochs':          EPOCHS,
                'training_time_minutes': train_time,
                'final_time_minutes':    final_time,
                'model_count_params':    count_params(model),
            }
            pd.DataFrame([results]).to_csv(
                os.path.join(checkpoint_dir, "final_results.csv"), index=False
            )
            all_results.append(results)

            print(f"模型参数量: {count_params(model):,}")
            print(f"\n✓ 当前实验完成: trunc={trunc}, Nblock={Nblock}")

    if all_results:
        pd.DataFrame(all_results).to_csv(
            os.path.join(args.checkpoint_dir, "sweep_summary.csv"), index=False
        )
        print("\n=== 所有调参实验已完成，汇总结果已保存为 sweep_summary.csv ===")

    print("\n✓ 频域 CNN 自编码器处理完成！")
    print("Done!")
