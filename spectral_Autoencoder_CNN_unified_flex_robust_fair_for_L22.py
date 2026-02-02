import numpy as np, pickle, time, pandas as pd, matplotlib
import os, argparse, json
matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from matplotlib.colors import ListedColormap
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.utils.extmath import randomized_svd
import random
import math
from tensorflow.keras import layers, backend as K

# ──────────────────────────────────────────────── Callbacks and utilities ──────────────────
class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"Epoch {epoch+1:04d} | loss={logs['loss']:.3e}")

def scheduler(epoch, lr):
    """
    Every 500 epochs, reduce learning rate by one order of magnitude.
    """
    initial_lr = 1e-3  # Define initial learning rate

    # Calculate number of decay stages
    decay_stage = epoch // 500

    # Calculate new learning rate
    new_lr = initial_lr * math.pow(0.1, decay_stage)

    return new_lr


def prepare_cnn_data(fft_blocks_real, spatial_dim, freq_dim):
    """Prepare data format for CNN"""
    n_blocks, n_space, freq_features = fft_blocks_real.shape
    cnn_data = fft_blocks_real.reshape(n_blocks, spatial_dim, freq_dim, 2)
    return cnn_data

def reshape_cnn_output(cnn_output):
    """Reshape CNN output back to original format for FFT synthesis"""
    n_blocks, spatial_dim, freq_dim, channels = cnn_output.shape
    reshaped = cnn_output.reshape(n_blocks, spatial_dim, freq_dim * channels)
    return reshaped

# ───────────────────────────────────────── Plot utilities ──────────────
def plot_difference(t_true, u_true, t_pred, u_pred, fn):
    N = u_true.shape[0]
    colors  = sns.diverging_palette(240, 10, n=9,  as_cmap=True)
    colors2 = sns.diverging_palette(240, 10, n=41); colors2 = ListedColormap(colors2[20:])
    fig, ax = plt.subplots(4, 1, figsize=(8, 8), dpi=400,
                           sharex=True, sharey=False,
                           gridspec_kw=dict(hspace=0.05))
    ax[0].pcolormesh(t_true, np.linspace(-11, 11, N), u_true,
                     shading="gouraud", cmap=colors, vmin=-3, vmax=3)
    ax[1].pcolormesh(t_pred, np.linspace(-11, 11, N), u_pred,
                     shading="gouraud", cmap=colors, vmin=-3, vmax=3)
    ax[2].pcolormesh(t_pred, np.linspace(-11, 11, N), np.abs(u_pred-u_true),
                     shading="gouraud", cmap=colors2, vmin=0,  vmax=3)
    ax[3].plot(t_pred, np.linalg.norm(u_pred-u_true, axis=0))
    ax[0].set_ylabel(r'$x$'); ax[1].set_ylabel(r'$x$'); ax[2].set_ylabel(r'$x$')
    ax[3].set(ylabel=r'$||u-\tilde{u}||$', xlabel=r'$t$')
    plt.savefig(fn, bbox_inches="tight"); plt.close()

def plot_history(hist, fn, resume_epoch=0):
    """Plot training history, supports epoch offset for resumed training"""
    plt.figure(figsize=(10, 6))

    # Adjust epoch display
    epochs = hist["epoch"].to_numpy() + resume_epoch + 1

    plt.semilogy(epochs, hist["loss"].to_numpy(), label="train")
    if "val_loss" in hist:
        plt.semilogy(epochs, hist["val_loss"].to_numpy(), label="val")

    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Training History")
    plt.grid(True, alpha=0.3)

    # If resumed training, add marker line
    if resume_epoch > 0:
        plt.axvline(x=resume_epoch, color='red', linestyle='--', alpha=0.7,
                   label=f'Resume from epoch {resume_epoch}')
        plt.legend()

    plt.tight_layout()
    plt.savefig(fn, dpi=150)
    plt.close()

def complex_to_real(complex_data):
    """Convert complex data to real data [real, imag]"""
    real_part = np.real(complex_data)
    imag_part = np.imag(complex_data)
    return np.concatenate([real_part, imag_part], axis=-1)

def real_to_complex(real_data):
    """Convert real data back to complex data"""
    mid = real_data.shape[-1] // 2
    real_part = real_data[..., :mid]
    imag_part = real_data[..., mid:]
    return real_part + 1j * imag_part

def build_flexible_cnn_ae(
        input_shape,
        spatial_target=8,
        freq_target=None,   
        filters_base=64,
        bottleneck_channels=None,  # None means automatic calculation
        use_bilinear_upsample=True  # True: bilinear upsampling; False: nearest neighbor
):

    print(f"[INFO] Decoder output shape: ")

    return model

def fft_analysis_blocks(signal, Nblock, overlap):
    """Correct FFT analysis phase: blocking + window function + FFT"""
    step = int(Nblock * (1 - overlap))
    window = np.hanning(Nblock)

    if signal.ndim == 2:
        n_space, n_time = signal.shape
    else:
        n_space, n_time = 1, len(signal)
        signal = signal.reshape(1, -1)

    # Add boundary padding to ensure complete coverage
    pad_start = Nblock // 2
    pad_end = Nblock // 2
    signal_pad = np.pad(signal, ((0, 0), (pad_start, pad_end)), mode='edge')

    # Calculate block positions
    first_center = pad_start
    last_center = pad_start + n_time - 1
    n_blocks = int(np.ceil((last_center - first_center) / step)) + 1

    # Analysis phase: blocking + window function + FFT
    fft_blocks = []
    block_info = []  # Save block info for reconstruction

    for i in range(n_blocks):
        center = first_center + i * step
        start = center - Nblock // 2
        end = start + Nblock

        if start >= 0 and end <= signal_pad.shape[1]:
            # Apply window function and FFT
            block = signal_pad[:, start:end] * window[None, :]
            fft_block = np.fft.rfft(block, axis=1)
            fft_blocks.append(fft_block)
            block_info.append((start, end, center))

    fft_blocks = np.array(fft_blocks)  # (n_blocks, n_space, n_freq)

    return fft_blocks, block_info, signal_pad.shape[1], (pad_start, pad_end, n_time)

def fft_synthesis_blocks(fft_blocks, block_info, padded_length, padding_info, Nblock, overlap):
    """Correct FFT synthesis phase: iFFT + Overlap-Add"""
    step = int(Nblock * (1 - overlap))
    window = np.hanning(Nblock)
    pad_start, pad_end, original_length = padding_info

    n_blocks, n_space, n_freq = fft_blocks.shape

    # Synthesis phase: iFFT + Overlap-Add
    output = np.zeros((n_space, padded_length))
    window_sum = np.zeros(padded_length)

    for i, (fft_block, (start, end, center)) in enumerate(zip(fft_blocks, block_info)):
        # iFFT (Note: no longer apply window function!)
        time_block = np.fft.irfft(fft_block, n=Nblock, axis=1)

        # Overlap-Add
        output[:, start:end] += time_block
        window_sum[start:end] += window

    # Normalization
    eps = 1e-15
    window_sum = np.maximum(window_sum, eps)
    output = output / window_sum[None, :]

    # Remove padding, restore original length
    output = output[:, pad_start:pad_start + original_length]

    return output

# ──────────────────────────────────────────────── Main process ──────────────────────
if __name__ == "__main__":
    t_init = time.time()

    # Set Python random seed
    random.seed(42)
    # Set NumPy random seed
    np.random.seed(42)
    # Set TensorFlow random seed
    tf.random.set_seed(42)
    # Command-line argument parsing
    parser = argparse.ArgumentParser(description='CNN frequency-domain autoencoder training')
    parser.add_argument('--trunc', type=int, default=8,
                        help='Truncation parameter (will be overridden by trunc list in tuning mode)')
    parser.add_argument('--epochs', type=int, default=1500,
                        help='Number of training epochs for frequency AE')
    parser.add_argument('--filters_base', type=int, default=64,
                        help='Number of base CNN filters')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints_cnn_spatial_a1_20_robust_L22',
                        help='Checkpoint save directory (will be further subdivided by trunc and Nblock internally)')
    parser.add_argument('--model_type', type=str, default='basic',
                        choices=['basic', 'balanced', 'progressive'], help='Model type')
    parser.add_argument('--freq_reduction', default=False,
                    help='Whether to perform frequency dimension reduction')
    parser.add_argument('--freq_target', type=int, default=11,
                        help='Target frequency dimension')

    args = parser.parse_args()

    eps = 1e-15
    trunc = args.trunc
    EPOCHS = args.epochs
    model_type = args.model_type
    freq_reduction = args.freq_reduction

    # Tuning lists: candidate values for trunc and Nblock
    trunc_list = [5, 6, 7, 8, 9, 10]
    Nblock_list = [20, 40, 60, 80, 100, 120, 140, 160]

    print("=== Tuning Settings ===")
    print(f"trunc candidates: {trunc_list}")
    print(f"Nblock candidates: {Nblock_list}")
    print(f"Total {len(trunc_list) * len(Nblock_list)} combinations.")

    # ① Load & normalize original snapshots (only once)
    print("\n=== Data Loading ===")
    u_true = pickle.load(open("kse_solution_L22_N64_steps400000_warm5000_h0.25.pkl", "rb"))
    u_train, u_test = u_true[:100_000], u_true[100_000:108_000]
    u_train, u_test = u_train.T, u_test.T

    # Data normalization
    u_mean, u_std = u_train.mean(1, keepdims=True), u_train.std(1, keepdims=True) + eps
    u_train_norm = (u_train - u_mean) / u_std
    u_test_norm = (u_test - u_mean) / u_std

    print(f"Data shape: train {u_train.shape}, test {u_test.shape}")

    # ② Frequency domain blocking parameters (overlap fixed, Nblock traversed in loop)
    overlap = 0.5

    # Used to save summary of results for all combinations
    all_results = []

    # Traverse trunc and Nblock combinations for tuning
    for trunc in trunc_list:
        for Nblock in Nblock_list:

            # freq_target = (Nblock + 1) / 2
            freq_target = Nblock//2 +1
            print("\n" + "=" * 60)
            print(f"Starting experiment: trunc={trunc}, Nblock={Nblock}")
            print("=" * 60)
            print(f"FFT parameters: Nblock={Nblock}, overlap={overlap}")

            # ③ FFT analysis to obtain frequency domain data
            print("Generating training frequency domain data...")
            fft_train, train_block_info, train_padded_len, train_padding_info = fft_analysis_blocks(u_train_norm, Nblock, overlap)

            print("Generating test frequency domain data...")
            fft_test, test_block_info, test_padded_len, test_padding_info = fft_analysis_blocks(u_test_norm, Nblock, overlap)

            n_blocks_train, n_space, n_freq = fft_train.shape
            n_blocks_test = fft_test.shape[0]

            print(f"Frequency domain training data: {fft_train.shape}")
            print(f"Frequency domain test data: {fft_test.shape}")

            # ④ Data preprocessing: convert to CNN format
            fft_train_real = complex_to_real(fft_train)
            fft_test_real = complex_to_real(fft_test)

            train_data_cnn = prepare_cnn_data(fft_train_real, n_space, n_freq)
            test_data_cnn = prepare_cnn_data(fft_test_real, n_space, n_freq)

            print(f"CNN input shape: train {train_data_cnn.shape}, test {test_data_cnn.shape}")

            # ⑤ Train dimensionality reduction model
            input_shape = train_data_cnn.shape[1:]
            model = build_flexible_cnn_ae(
                    input_shape=input_shape,
                    spatial_target=trunc,
                    freq_target=freq_target
                )
            model.compile(loss="mse", optimizer=tf.keras.optimizers.Adam())

            print(f"\n=== Starting Training ===")
            print(f"Target epochs: {EPOCHS}")
            print(f"Current experiment parameters: trunc={trunc}, Nblock={Nblock}")

            t0 = time.time()

            # Start training
            hist = model.fit(
                train_data_cnn, train_data_cnn,
                epochs=EPOCHS,
                validation_data=(test_data_cnn, test_data_cnn),
                verbose=0,
                callbacks=[PrintDot(), keras.callbacks.LearningRateScheduler(scheduler)],
                batch_size=32,
                initial_epoch=0  # Keras internal epoch counting starts from 0
            )

            train_time = (time.time()-t0)/60
            print(f"Training completed, time elapsed {train_time:.1f} minutes")

            # Create checkpoint directory: store by trunc and Nblock combinations
            if freq_reduction:
                checkpoint_dir = os.path.join(args.checkpoint_dir, f"trunc{trunc}_Nblock{Nblock}_freq{freq_target}_{model_type}")
            else:
                checkpoint_dir = os.path.join(args.checkpoint_dir, f"trunc{trunc}_Nblock{Nblock}_freq{freq_target}")

            os.makedirs(checkpoint_dir, exist_ok=True)

            # Merge training history
            new_history_df = pd.DataFrame(hist.history)
            new_history_df["epoch"] = new_history_df.index
            combined_history = new_history_df

            # Save complete training history
            combined_history.to_csv(os.path.join(checkpoint_dir, "training_history.csv"), index=False)
            plot_history(combined_history,
                                os.path.join(checkpoint_dir, f"training_curve_cnn_trunc{trunc}_Nblock{Nblock}.png"),
                                resume_epoch=0)  # Since epoch has been adjusted, use 0 here

            # Save final model
            final_model_path = os.path.join(checkpoint_dir, f"final_cnn_model_trunc{trunc}_Nblock{Nblock}.h5")
            model.save_weights(final_model_path)
            print(f"✅ Final model saved: {final_model_path}")

            # Prediction
            print("\n=== Prediction and Reconstruction ===")
            recon_data_cnn = model.predict(test_data_cnn, verbose=0)

            # ⑥ Reconstruction and evaluation
            print("Reconstructing frequency domain data format...")
            recon_data_reshaped = reshape_cnn_output(recon_data_cnn)
            recon_complex = real_to_complex(recon_data_reshaped)

            print("Performing FFT synthesis...")
            pred_u_norm = fft_synthesis_blocks(
                recon_complex,
                test_block_info,
                test_padded_len,
                test_padding_info,
                Nblock,
                overlap
            )

            # Denormalize
            pred_u = pred_u_norm * u_std + u_mean

            # Calculate error
            mse = np.mean((u_test - pred_u)**2)
            max_error = np.max(np.abs(u_test - pred_u))
            rel_error = np.sqrt(mse) / np.std(u_test)

            print(f"\n=== Final Results Statistics (trunc={trunc}, Nblock={Nblock}) ===")
            print(f"Maximum absolute error = {max_error:.3e}")
            print(f"MSE = {mse:.3e}")
            print(f"Relative error = {rel_error:.3%}")

            # ⑦ Verify FFT reconstruction accuracy
            print(f"\n=== FFT Reconstruction Accuracy Verification ===")
            perfect_recon_norm = fft_synthesis_blocks(
                fft_test,
                test_block_info,
                test_padded_len,
                test_padding_info,
                Nblock,
                overlap
            )
            perfect_recon = perfect_recon_norm * u_std + u_mean

            fft_error = np.max(np.abs(u_test - perfect_recon))
            print(f"FFT perfect reconstruction error = {fft_error:.3e}")

            if fft_error < 1e-12:
                print("FFT reconstruction algorithm correct, achieved machine precision")
            else:
                print("FFT reconstruction algorithm may have issues")

            # Calculate pure reconstruction error of neural network (excluding FFT error)
            ae_only_error = np.max(np.abs(perfect_recon - pred_u))
            print(f"CNN autoencoder pure reconstruction error = {ae_only_error:.3e}")

            # ⑧ Save results and plots
            print(f"\n=== Saving Results ===")

            # Save normalization parameters
            np.savez(os.path.join(checkpoint_dir, "normalization_params.npz"),
                     u_mean=u_mean, u_std=u_std)

            # Plot comparison figure
            dt = 0.25
            t = np.arange(0, len(u_true)) * dt
            t_test = t[100_000:108_000]

            # Select a portion of data for visualization
            plot_start, plot_end = 0, min(400, len(t_test))
            t_plot = t_test[plot_start:plot_end]
            u_plot = u_test[:, plot_start:plot_end]
            pred_plot = pred_u[:, plot_start:plot_end]

            plot_difference(t_plot, u_plot, t_plot, pred_plot,
                            os.path.join(checkpoint_dir, "cnn_spectral_ae_comparison.png"))

            # Save detailed results
            final_time = (time.time()-t_init)/60
            final_epoch = EPOCHS
            results = {
                'max_error': max_error,
                'mse': mse,
                'rel_error': rel_error,
                'fft_error': fft_error,
                'ae_only_error': ae_only_error,
                'trunc': trunc,
                'Nblock': Nblock,
                'filters_base': args.filters_base,
                'final_epoch': final_epoch,
                'total_epochs': EPOCHS,
                'training_time_minutes': train_time,
                'final_time_minutes': final_time,
                'model_count_params': model.count_params()
            }

            results_df = pd.DataFrame([results])
            results_df.to_csv(os.path.join(checkpoint_dir, "final_results.csv"), index=False)

            all_results.append(results)

            print(f"Model parameter count: {model.count_params():,}")
            print(f"\nCurrent experiment completed: trunc={trunc}, Nblock={Nblock}")

    # ⑨ After all tuning experiments, save summary CSV
    if len(all_results) > 0:
        summary_df = pd.DataFrame(all_results)
        summary_df.to_csv(os.path.join(args.checkpoint_dir, "sweep_summary.csv"), index=False)
        print("\n=== All tuning experiments completed, summary results saved as sweep_summary.csv ===")

    print("\nDual dimensionality reduction CNN autoencoder processing completed!")
    print("Done!")
