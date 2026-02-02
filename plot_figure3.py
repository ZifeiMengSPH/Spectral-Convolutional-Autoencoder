
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from matplotlib.colors import LogNorm, ListedColormap
import os
import argparse
import seaborn as sns

# ========== Global plotting style settings ==========
try:
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["mathtext.fontset"] = "custom"
    plt.rcParams["mathtext.rm"] = "Times New Roman"
    plt.rcParams["mathtext.it"] = "Times New Roman:italic"
    plt.rcParams["axes.linewidth"] = 0.9
except:
    print("Warning: Times New Roman font setup failed, using default font")
    pass

sns.set_context("paper", font_scale=1.5)
# Change to white style, remove default grid
sns.set_style("white")
# Force font settings again to prevent being overridden by seaborn
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["mathtext.rm"] = "Times New Roman"
plt.rcParams["mathtext.it"] = "Times New Roman:italic"



def plot_combined_energy_decay(file_paths, labels, save_path, figsize=(10, 7)):
    """Plot Energy Decay curves from multiple files in one figure"""
    print(f"Generating combined Energy Decay plot: {save_path}")
    
    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(file_paths)))
    markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'X']
    max_latent_dim = 0
    
    for i, (npz_path, label) in enumerate(zip(file_paths, labels)):
        if not os.path.exists(npz_path):
            continue
        try:
            data = np.load(npz_path, allow_pickle=True)
            if 'h_data_complex' in data:
                h_data_complex = data['h_data_complex']
                magnitude = np.abs(h_data_complex)
                energy_per_latent = np.sum(np.square(magnitude), axis=(0, 2))
            elif 'energy_per_latent' in data:
                energy_per_latent = data['energy_per_latent']
            else:
                continue
            
            sorted_energy = np.sort(energy_per_latent)[::-1]
            max_latent_dim = max(max_latent_dim, len(sorted_energy))
            latent_indices = np.arange(1, len(sorted_energy) + 1)
            
            mark_every = 1
            if len(sorted_energy) > 50:
                mark_every = max(1, len(sorted_energy) // 20)
            
            ax.plot(latent_indices, sorted_energy, color=colors[i], marker=markers[i % len(markers)],
                   markersize=6, markevery=mark_every, linewidth=2.0, label=label, alpha=0.9)
        except Exception as e:
            print(f"    Processing failed {label}: {e}")
            continue

    ax.set_xlabel('Latent Index (Sorted by Energy)', fontsize=20)
    ax.set_ylabel(r'Total Energy $\sum |h|^2$', fontsize=20)
    ax.set_title('Energy Decay Comparison', fontsize=18, loc='center', pad=15)
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax.yaxis.get_offset_text().set_fontsize(14)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_xlim(0, max_latent_dim + 2)
    
    if max_latent_dim > 0:
        step = max(5, int(max_latent_dim / 15 / 5) * 5)
        if step == 0: step = 1
        xticks = np.arange(0, max_latent_dim + step, step)
        xticks[0] = 1
        ax.set_xticks(xticks)

    ax.legend(fontsize=14, frameon=True, framealpha=0.9, edgecolor='gray', loc='best')
    ax.grid(True, which="both", ls="--", alpha=0.4, color='gray', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    eps_path = save_path.replace('.png', '.eps')
    plt.savefig(eps_path, format='eps', bbox_inches='tight')
    plt.close()
    print(f"  -> Chart saved: {save_path}")


def get_optimal_xticks(num_latent):
    """Return appropriate ticks based on Latent dimension count"""
    if num_latent <= 10:
        return np.arange(1, num_latent + 1)
    elif num_latent <= 50:
        ticks = np.arange(0, num_latent + 1, 5)
        ticks[0] = 1
        return ticks
    else:
        # For larger dimensions (L66, L88, L110), set step to 10
        ticks = np.arange(0, num_latent + 1, 10)
        ticks[0] = 1
        return ticks

def draw_heatmap(ax, mean_spectrum):
    """Plot (a) Spectral Decomposition Heatmap"""
    num_latent = mean_spectrum.shape[0]
    num_freq = mean_spectrum.shape[1]
     #inferno
    im = ax.imshow(mean_spectrum.T, aspect='auto', origin='lower', 
                   cmap='inferno', norm=LogNorm(vmin=np.max(mean_spectrum)*1e-3, vmax=np.max(mean_spectrum)),
                   extent=[0.5, num_latent + 0.5, 0.5, num_freq + 0.5])
    
    ax.set_ylabel(r'Temporal Frequency Bin $k$', fontsize=20)
    ax.set_xlabel(r'Latent Index (Sorted by Energy)', fontsize=20)
    ax.set_xlim(0.5, num_latent + 0.5)
    
    # Optimize X-axis ticks
    ax.set_xticks(get_optimal_xticks(num_latent))
    
    # Y-axis tick handling
    if num_freq <= 20:
        y_ticks = np.arange(1, num_freq + 1, 1)
    elif num_freq <= 62:
        y_ticks = np.arange(0, num_freq + 1, 10)
        y_ticks[0] = 1
    else:
        y_ticks = np.arange(0, num_freq + 1, 20)
        y_ticks[0] = 1
    ax.set_yticks(y_ticks)
    ax.tick_params(bottom=True, left=True, direction='out', width=0.9, length=4)
    ax.set_title(r'(a) Spectral Decomposition in Latent Space', loc='left', fontsize=18, pad=19)
    return im

def draw_energy_decay(ax, energy_per_latent, sorted_indices, num_latent):
    """Plot (b) Energy Decay"""
    latent_indices_1based = np.arange(1, num_latent + 1)
    ax.plot(latent_indices_1based, energy_per_latent[sorted_indices], color='k', marker='o', markersize=4, lw=1)
    
    # Decide whether to use logarithmic or linear scale based on energy range
    e_max = np.max(energy_per_latent)
    e_min = np.min(energy_per_latent[energy_per_latent > 0])
    if e_max / e_min > 100:
        ax.set_yscale('log')
        ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=6))
    else:
        ax.set_yscale('linear')
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))

    ax.set_xlabel('Latent Index', fontsize=20)
    ax.set_ylabel(r'Total Energy $E_m = \sum_k \langle |h_{m,k}|^2 \rangle$', fontsize=20)
    ax.tick_params(bottom=True, left=True, direction='out', width=0.9, length=4)
    ax.set_title(r'(b) Energy Decay', loc='left', fontsize=20, pad=19)
    ax.grid(True, which="major", ls="-", alpha=0.3)
    ax.grid(True, which="minor", ls=":", alpha=0.1)
    
    # Optimize X-axis ticks
    ax.set_xticks(get_optimal_xticks(num_latent))
    ax.set_xlim(0.5, num_latent + 0.5)

def draw_latent_mode(ax, h_data_complex, u_data, block_info, mode_idx, sorted_indices, mean_spectrum, title_prefix, n_samples_plot=500, npz_path=None):
    """Plot Dynamics for a single Latent Mode (c) or (d)"""
    # Get data for specific mode
    idx = sorted_indices[mode_idx]
    peak_freq_idx = np.argmax(mean_spectrum[mode_idx, :])
    
    n_samples_plot = min(n_samples_plot, h_data_complex.shape[0])
    if block_info is not None:
        n_samples_plot = min(n_samples_plot, len(block_info))
        
    z_series = h_data_complex[:n_samples_plot, idx, peak_freq_idx]
    
    # Plot trajectory
    colors_time = np.linspace(0, 1, n_samples_plot)
    sc = ax.scatter(z_series.real, z_series.imag, c=colors_time, cmap='viridis', s=15, alpha=0.8, zorder=2)
    ax.grid(False)
    ax.axhline(0, color='gray', lw=0.5, ls='--')
    ax.axvline(0, color='gray', lw=0.5, ls='--')
    ax.set_xlabel(r'Re($h$)', fontsize=20)
    ax.set_ylabel(r'Im($h$)', fontsize=20)
    ax.tick_params(bottom=True, left=True, direction='out', width=0.9, length=4)
    
    real_k = peak_freq_idx + 1
    # ax.set_title(f'{title_prefix} Latent Mode #{mode_idx+1} ($k={real_k}$)', fontsize=12, loc='left')
    ax.set_title(f'{title_prefix} Latent Mode #{mode_idx+1}', fontsize=20, loc='left')
    ax.axis('equal')
    
    # --- Adjust Y-axis range for L22 ---
    if npz_path and "L22" in os.path.basename(npz_path):
        if mode_idx == 0:   # Mode 1
            ax.set_ylim(-100, 150)
        elif mode_idx == 1: # Mode 2
            ax.set_ylim(-100, 150)

    # Physical embedding
    if u_data is not None and block_info is not None:
        mag_series = np.abs(z_series)
        t_max = np.argmax(mag_series)
        t_min = np.argmin(mag_series)
        
        cmap_div = sns.diverging_palette(240, 10, n=9, as_cmap=True)
        
        ax.scatter(z_series.real[t_min], z_series.imag[t_min], c='blue', s=40, marker='D', edgecolors='k', label='Min', zorder=10)
        ax.scatter(z_series.real[t_max], z_series.imag[t_max], c='red', s=60, marker='*', edgecolors='k', label='Max', zorder=10)
        ax.legend(loc='upper right', fontsize=7, framealpha=0.8)
        
        # Insets
        # Inset 1: Min Energy (Top Left usually)
        ax_ins1 = ax.inset_axes([0.05, 0.65, 0.35, 0.3])
        start_min, end_min, _ = block_info[t_min]
        if start_min < u_data.shape[1]:
            u_block = u_data[:, start_min:end_min]
            ax_ins1.imshow(u_block, aspect='auto', cmap=cmap_div, origin='lower', vmin=-3, vmax=3)
        ax_ins1.set_xticks([])
        ax_ins1.set_yticks([])
        for spine in ax_ins1.spines.values(): spine.set_edgecolor('k'); spine.set_linewidth(1.0)
        
        # Inset 2: Max Energy (Bottom Right usually)
        # For L22, to avoid blocking, move y coordinate up slightly, or keep at 0.05 since smaller ymin gives more space
        ax_ins2 = ax.inset_axes([0.6, 0.05, 0.35, 0.3])
        start_max, end_max, _ = block_info[t_max]
        if start_max < u_data.shape[1]:
            u_block = u_data[:, start_max:end_max]
            ax_ins2.imshow(u_block, aspect='auto', cmap=cmap_div, origin='lower', vmin=-3, vmax=3)
        ax_ins2.set_xticks([])
        ax_ins2.set_yticks([])
        for spine in ax_ins2.spines.values(): spine.set_edgecolor('k'); spine.set_linewidth(1.0)
    
    return sc

def plot_comprehensive_analysis(npz_path, save_path=None):
    """
    Generate large plot containing (a) Heatmap, (b) Energy Decay, (c) Mode 1, (d) Mode 2
    """
    if not os.path.exists(npz_path):
        return

    print(f"Plotting comprehensive analysis: {os.path.basename(npz_path)}")
    try:
        data = np.load(npz_path, allow_pickle=True)
        h_data_complex = data['h_data_complex']
        u_data = data.get('u_data', None)
        block_info = data.get('block_info', None)
    except Exception:
        return

    # Data calculation
    magnitude = np.abs(h_data_complex)
    energy_per_latent = np.sum(np.square(magnitude), axis=(0, 2))
    sorted_indices = np.argsort(energy_per_latent)[::-1]
    h_sorted_mag = magnitude[:, sorted_indices, :]
    mean_spectrum = np.mean(h_sorted_mag, axis=0)
    
    num_latent = mean_spectrum.shape[0]

    if save_path is None:
        save_path = npz_path.replace('.npz', '_comprehensive.png')

    fig = plt.figure(figsize=(14, 10))
    # 2x2 layout - increase wspace to avoid colorbar labels overlapping with y-axis labels of right subplots
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1], wspace=0.4, hspace=0.3)
    
    # (a) Heatmap
    ax_a = fig.add_subplot(gs[0, 0])
    im = draw_heatmap(ax_a, mean_spectrum)
    plt.colorbar(im, ax=ax_a, fraction=0.046, pad=0.04).set_label(r'Magnitude $\langle |h_{m,k}| \rangle$ (log scale)', rotation=270, labelpad=15)
    
    # (b) Energy Decay
    ax_b = fig.add_subplot(gs[0, 1])
    draw_energy_decay(ax_b, energy_per_latent, sorted_indices, num_latent)
    
    # (c) Latent Mode 1
    if num_latent >= 1:
        ax_c = fig.add_subplot(gs[1, 0])
        sc = draw_latent_mode(ax_c, h_data_complex, u_data, block_info, 0, sorted_indices, mean_spectrum, "(c)", npz_path=npz_path)
        # Colorbar for time
        cbar = plt.colorbar(sc, ax=ax_c, fraction=0.046, pad=0.04)
        cbar.set_label('Time (Normalized)')
        cbar.ax.tick_params(labelsize=8)

    # (d) Latent Mode 2
    if num_latent >= 2:
        ax_d = fig.add_subplot(gs[1, 1])
        sc = draw_latent_mode(ax_d, h_data_complex, u_data, block_info, 1, sorted_indices, mean_spectrum, "(d)", npz_path=npz_path)
        cbar = plt.colorbar(sc, ax=ax_d, fraction=0.046, pad=0.04)
        cbar.set_label('Time (Normalized)')
        cbar.ax.tick_params(labelsize=8)

    # plt.suptitle(f"Latent Space Analysis ({os.path.basename(npz_path)})", fontsize=20, y=0.98)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace('.png', '.eps'), format='eps', bbox_inches='tight')
    plt.close()
    print(f"  -> Comprehensive analysis plot saved: {save_path}")

def plot_latent_dynamics_single(npz_path, save_path=None):

    """
    Plot Latent Dynamics for a single file (complex plane trajectory + physical embedding)
    Strictly follows the style of plot_latent_dynamics function in plot_paper_figures.py
    """
    if not os.path.exists(npz_path):
        print(f"Error: File does not exist {npz_path}")
        return

    print(f"Plotting Latent Dynamics: {os.path.basename(npz_path)}")
    
    try:
        data = np.load(npz_path, allow_pickle=True)
        h_data_complex = data['h_data_complex']
        
        # Try to read physical data
        u_data = data['u_data'] if 'u_data' in data else None
        block_info = data['block_info'] if 'block_info' in data else None
        
        if u_data is None or block_info is None:
            print(f"Warning: {npz_path} missing u_data or block_info, will only plot trajectory without physical embedding")
            
    except Exception as e:
        print(f"Failed to load npz: {e}")
        return

    # Data preprocessing
    magnitude = np.abs(h_data_complex)
    energy_per_latent = np.sum(np.square(magnitude), axis=(0, 2))
    sorted_indices = np.argsort(energy_per_latent)[::-1]
    
    # Calculate mean spectrum to determine peak frequency
    h_sorted_mag = magnitude[:, sorted_indices, :]
    mean_spectrum = np.mean(h_sorted_mag, axis=0)

    if save_path is None:
        save_path = npz_path.replace('.npz', '_latent_dynamics.png')

    print(f"Plotting Latent dynamics: {save_path}")
    fig = plt.figure(figsize=(18, 6)) # Wider canvas
    gs = gridspec.GridSpec(1, 3, wspace=0.3)
    
    # Only use top 3 strongest modes
    n_samples_plot = 500 # Plot only first 500 steps
    n_samples_plot = min(n_samples_plot, h_data_complex.shape[0])
    
    if block_info is not None:
        n_samples_plot = min(n_samples_plot, len(block_info))
    
    for i in range(3):
        if i >= len(sorted_indices):
            break

        idx = sorted_indices[i] # Original latent index
        peak_freq_idx = np.argmax(mean_spectrum[i, :]) # Peak frequency of this mode
        
        # Extract complex time series (N, )
        z_series = h_data_complex[:n_samples_plot, idx, peak_freq_idx]
        
        ax = fig.add_subplot(gs[i])
        
        # Plot complex plane trajectory (Re vs Im)
        # Use color to encode time, remove misleading connecting lines
        colors_time = np.linspace(0, 1, n_samples_plot)
        sc = ax.scatter(z_series.real, z_series.imag, c=colors_time, cmap='viridis', s=15, alpha=0.8, zorder=2)
        
        # Ensure grid is off
        ax.grid(False)
        
        # Add origin lines
        ax.axhline(0, color='gray', lw=0.5, ls='--')
        ax.axvline(0, color='gray', lw=0.5, ls='--')
        
        ax.set_xlabel(r'Re($h$)', fontsize=20)
        ax.set_ylabel(r'Im($h$)', fontsize=20)
        
        real_k = peak_freq_idx + 1
        ax.set_title(f'Latent Mode #{i+1}', fontsize=20)
        ax.axis('equal')
        
        # --- Rigorous physical validation ---
        if u_data is not None and block_info is not None:
            # Find energy maximum point (Burst) and minimum point (Quiescent)
            mag_series = np.abs(z_series)
            t_max = np.argmax(mag_series)
            t_min = np.argmin(mag_series)

            # Colormap (use diverging_palette to ensure consistency with reference figure)
            cmap_div = sns.diverging_palette(240, 10, n=9, as_cmap=True)
            
            # Mark extreme points on trajectory
            ax.scatter(z_series.real[t_min], z_series.imag[t_min], c='blue', s=80, marker='D', edgecolors='k', label='Min Energy', zorder=10)
            ax.scatter(z_series.real[t_max], z_series.imag[t_max], c='red', s=80, marker='*', edgecolors='k', label='Max Energy', zorder=10)
            ax.legend(loc='upper right', fontsize=8)

            # Inset 1: Low Energy Physical State
            ax_ins1 = ax.inset_axes([0.05, 0.65, 0.35, 0.3])
            start_min, end_min, _ = block_info[t_min]
            # u_data shape is (Space, Time), slice time
            if start_min < u_data.shape[1] and end_min <= u_data.shape[1]:
                u_block_min = u_data[:, start_min:end_min]
                # Use cmap_div to ensure red-blue color scheme
                ax_ins1.imshow(u_block_min, aspect='auto', cmap=cmap_div, origin='lower', vmin=-3, vmax=3)
            ax_ins1.set_xticks([])
            ax_ins1.set_yticks([])
            ax_ins1.set_title(f'u(x,t) @ Min', fontsize=7, color='blue')
            # Set border color
            for spine in ax_ins1.spines.values():
                spine.set_edgecolor('k')
                spine.set_linewidth(1.0)

            # Inset 2: High Energy Physical State
            ax_ins2 = ax.inset_axes([0.6, 0.05, 0.35, 0.3])
            start_max, end_max, _ = block_info[t_max]
            if start_max < u_data.shape[1] and end_max <= u_data.shape[1]:
                u_block_max = u_data[:, start_max:end_max]
                ax_ins2.imshow(u_block_max, aspect='auto', cmap=cmap_div, origin='lower', vmin=-3, vmax=3)
            ax_ins2.set_xticks([])
            ax_ins2.set_yticks([])
            ax_ins2.set_title(f'u(x,t) @ Max', fontsize=7, color='red')
            # Set border color
            for spine in ax_ins2.spines.values():
                spine.set_edgecolor('k')
                spine.set_linewidth(1.0)
        else:
            # Fallback to magnitude plot if no physical data
            ins_ax = ax.inset_axes([0.65, 0.65, 0.3, 0.3])
            ins_ax.plot(np.abs(z_series)[:min(200, n_samples_plot)], 'k-', lw=0.8)
            ins_ax.set_xticks([])
            ins_ax.set_yticks([])
            ins_ax.set_title(r'$|h(t)|$', fontsize=8)

    # Add Colorbar to indicate time evolution
    cbar_ax = fig.add_axes([0.92, 0.15, 0.01, 0.7])
    cbar = fig.colorbar(sc, cax=cbar_ax, label='Time Evolution (Normalized)')
    
    plt.suptitle("Correspondence between Latent Trajectories and Physical Spatiotemporal Structures", fontsize=14, y=1.05)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Also save as EPS vector graphics
    eps_path = save_path.replace('.png', '.eps')
    plt.savefig(eps_path, format='eps', bbox_inches='tight')
    
    plt.close()
    print(f"  -> Latent Dynamics plot saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Batch plotting script')
    parser.add_argument('--mode', type=str, default='all', choices=['energy', 'latent', 'all'],
                       help='Plotting mode: energy (energy decay), latent (dynamics), all (all)')
    args = parser.parse_args()

    # 1. Determine data directory
    base_dirs = ['kse_out/fig4', 'fig4']
    target_dir = None
    for d in base_dirs:
        if os.path.exists(d):
            target_dir = d
            break
    
    if target_dir is None:
        if os.path.exists('L22.npz'):
            target_dir = '.'
        else:
            print("Error: Cannot find data folder")
            return

    files_info = [
        ('L22.npz', 'L=22'), ('L44.npz', 'L=44'), ('L66.npz', 'L=66'), 
        ('L88.npz', 'L=88'), ('L110.npz', 'L=110')
    ]
    
    file_paths = []
    labels = []
    for fname, label in files_info:
        npz_path = os.path.join(target_dir, fname)
        file_paths.append(npz_path)
        labels.append(label)

    print(f"{'='*60}")
    print(f"Starting batch plotting (mode: {args.mode})")
    print(f"Data directory: {os.path.abspath(target_dir)}")
    print(f"{'='*60}")

    # 2. Plot Latent Dynamics (plot separately)
    if args.mode in ['latent', 'all']:
        print("\n[Task 1] Plotting Latent Dynamics...")
        for npz_path in file_paths:
            if os.path.exists(npz_path):
                # Plot single latent dynamics figure (keep unchanged)
                plot_latent_dynamics_single(npz_path)
                # Plot comprehensive analysis figure (a,b,c,d)
                plot_comprehensive_analysis(npz_path)
            else:
                print(f"  Skipping missing file: {npz_path}")

    # 3. Plot Energy Decay (combined figure)
    if args.mode in ['energy', 'all']:
        print("\n[Task 2] Plotting Energy Decay Comparison...")
        save_path = os.path.join(target_dir, 'combined_energy_decay.png')
        plot_combined_energy_decay(file_paths, labels, save_path)

    print(f"\n{'='*60}")
    print("All completed!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
