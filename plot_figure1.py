
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm, ListedColormap
import seaborn as sns
import os
import argparse

# Set plotting style (optional)
try:
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["mathtext.fontset"] = "custom"
    plt.rcParams["mathtext.rm"] = "Times New Roman"
    plt.rcParams["mathtext.it"] = "Times New Roman:italic"
    plt.rcParams["axes.linewidth"] = 0.9
    # plt.rcParams["xtick.direction"] = "in"
    # plt.rcParams["ytick.direction"] = "in"
except:
    pass

# ========== Joint probability density plot utility functions (refer to joint_PDF_use3.py) ==========
def _gaussian_smooth2d(H, sigma_bins=0.0):
    if sigma_bins is None or sigma_bins <= 0:
        return H
    try:
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(H, sigma=sigma_bins, mode="nearest")
    except Exception:
        return H

def hist2d_prob(x, y, xedges, yedges, sigma_bins=0.0, eps=1e-12):
    """2D histogram → probability P"""
    H, _, _ = np.histogram2d(x, y, bins=[xedges, yedges], density=False)
    H = _gaussian_smooth2d(H.astype(float), sigma_bins)
    P = H / (H.sum() + eps)
    P[~np.isfinite(P)] = 0.0
    return P

def central_derivative_nonperiodic(u, dx=1.0, axis=0):
    """Non-periodic central difference; endpoints use one-sided difference/three-point approximation. Returns ux, uxx."""
    up = np.roll(u, -1, axis=axis)
    um = np.roll(u,  1, axis=axis)
    ux  = (up - um) / (2.0 * dx)
    uxx = (up - 2.0 * u + um) / (dx * dx)

    slicer = [slice(None)] * u.ndim
    slicer[axis] = 0;   i0  = tuple(slicer)
    slicer[axis] = 1;   i1  = tuple(slicer)
    slicer[axis] = -1;  im1 = tuple(slicer)
    slicer[axis] = -2;  im2 = tuple(slicer)

    # First-order endpoints
    ux[i0]  = (u.take(1, axis=axis)  - u.take(0, axis=axis))  / dx
    ux[im1] = (u.take(-1, axis=axis) - u.take(-2, axis=axis)) / dx
    # Second-order endpoints
    uxx[i0]  = (u.take(2, axis=axis)  - 2.0*u.take(1, axis=axis)  + u.take(0, axis=axis))   / (dx*dx)
    uxx[im1] = (u.take(-1, axis=axis) - 2.0*u.take(-2, axis=axis) + u.take(-3, axis=axis)) / (dx*dx)
    return ux, uxx


def derivatives_blockwise_seamless(u, block_size, overlap, dx=1.0, axis=0):
    """
    Seamless block-wise derivative computation + extract only center segments for stitching.
    Returns: ux_full, uxx_full, mask (True for valid center segments)
    """
    assert 0 <= overlap < 1.0
    stride = int(round(block_size * (1.0 - overlap)))
    assert stride > 0, "stride must be positive; check block_size/overlap"

    N = u.shape[axis]
    starts = list(range(0, N - block_size + 1, stride))
    margin = max(0, (block_size - stride) // 2)  # ~= block_size*overlap/2

    ux_full  = np.full_like(u, np.nan, dtype=float)
    uxx_full = np.full_like(u, np.nan, dtype=float)
    mask     = np.zeros_like(u, dtype=bool)

    for s in starts:
        e = s + block_size
        sl_block = [slice(None)] * u.ndim
        sl_block[axis] = slice(s, e)
        ub = u[tuple(sl_block)]

        ux_b, uxx_b = central_derivative_nonperiodic(ub, dx=dx, axis=axis)

        inner = slice(margin, block_size - margin) if margin > 0 else slice(None)
        sl_inner = [slice(None)] * u.ndim
        sl_inner[axis] = inner

        s_out = s + (margin if margin > 0 else 0)
        e_out = e - (margin if margin > 0 else 0)
        sl_out = [slice(None)] * u.ndim
        sl_out[axis] = slice(s_out, e_out)

        ux_full[tuple(sl_out)]  = ux_b[tuple(sl_inner)]
        uxx_full[tuple(sl_out)] = uxx_b[tuple(sl_inner)]
        mask[tuple(sl_out)]     = True

    return ux_full, uxx_full, mask


def flatten_masked_pairs(ux, uxx, mask):
    """Only extract (ux, uxx) pairs where mask=True for statistics"""
    return ux[mask].ravel(), uxx[mask].ravel()
# =========================================================================

def plot_latent_dynamics(h_data_complex, sorted_indices, mean_spectrum, save_path, u_data=None, block_info=None):
    """
    Plot temporal evolution of top 3 principal Latent variables (rigorous physics version)
    Display:
    1. Complex plane trajectories
    2. Physical tracing: corresponding physical spatiotemporal blocks at maximum/minimum energy moments
    """
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
        colors = np.linspace(0, 1, n_samples_plot)
        sc = ax.scatter(z_series.real, z_series.imag, c=colors, cmap='viridis', s=15, alpha=0.8, zorder=2)
        # ax.plot(z_series.real, z_series.imag, 'k-', lw=0.3, alpha=0.3, zorder=1) # Remove connecting lines
        
        # Add origin lines
        ax.axhline(0, color='gray', lw=0.5, ls='--')
        ax.axvline(0, color='gray', lw=0.5, ls='--')
        
        ax.set_xlabel(r'Re($h$)')
        ax.set_ylabel(r'Im($h$)')
        
        # Fix title: display wavenumber k instead of index, assuming k = index + 1
        real_k = peak_freq_idx + 1
        ax.set_title(f'Latent Mode #{i+1} (k={real_k})\nLatent Trajectory', fontsize=11)
        ax.axis('equal')
        
        # --- Rigorous physical validation ---
        if u_data is not None and block_info is not None:
            # Find energy maximum point (Burst) and minimum point (Quiescent)
            mag_series = np.abs(z_series)
            t_max = np.argmax(mag_series)
            t_min = np.argmin(mag_series)

            cmap = "inferno"
            cmap_base = plt.get_cmap(cmap)
            cmap_nan = ListedColormap(cmap_base(np.linspace(0, 1, 256)))
            cmap_nan.set_bad("white")
            
            
            colors  = sns.diverging_palette(240, 10, n=9,  as_cmap=True)
            colors2 = sns.diverging_palette(240, 10, n=41)
            colors2 = ListedColormap(colors2[20:])
            
            # Mark on trajectory
            ax.scatter(z_series.real[t_min], z_series.imag[t_min], c='blue', s=80, marker='D', edgecolors='k', label='Min Energy', zorder=10)
            ax.scatter(z_series.real[t_max], z_series.imag[t_max], c='red', s=80, marker='*', edgecolors='k', label='Max Energy', zorder=10)
            ax.legend(loc='upper right', fontsize=8)

            # Inset 1: Low Energy Physical State
            ax_ins1 = ax.inset_axes([0.05, 0.65, 0.35, 0.3])
            start_min, end_min, _ = block_info[t_min]
            # u_data shape is (Space, Time), slice time
            u_block_min = u_data[:, start_min:end_min]
            # 'RdBu_r'
            ax_ins1.imshow(u_block_min, aspect='auto', cmap=colors, origin='lower', vmin=-3, vmax=3)
            ax_ins1.set_xticks([])
            ax_ins1.set_yticks([])
            ax_ins1.set_title(f'u(x,t) @ Min', fontsize=7, color='blue')

            # Inset 2: High Energy Physical State
            ax_ins2 = ax.inset_axes([0.6, 0.05, 0.35, 0.3])
            start_max, end_max, _ = block_info[t_max]
            u_block_max = u_data[:, start_max:end_max]
            
            ax_ins2.imshow(u_block_max, aspect='auto', cmap=colors, origin='lower', vmin=-3, vmax=3)
            ax_ins2.set_xticks([])
            ax_ins2.set_yticks([])
            ax_ins2.set_title(f'u(x,t) @ Max', fontsize=7, color='red')
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
    plt.close()
    print(f"Latent dynamics evolution plot saved: {save_path}")

def plot_timefreq_learning_from_npz(npz_path, save_path=None):
    """
    Read npz data and plot frequency-Latent energy spectrum and phase distribution
    """
    if not os.path.exists(npz_path):
        print(f"Error: File does not exist {npz_path}")
        return

    print(f"Processing time-frequency analysis data: {npz_path}")
    try:
        data = np.load(npz_path, allow_pickle=True)
        h_data_complex = data['h_data_complex']
        # Try to read physical data if it exists (if user decided to save it)
        u_data = data['u_data'] if 'u_data' in data else None
        block_info = data['block_info'] if 'block_info' in data else None
    except Exception as e:
        print(f"Failed to load npz: {e}")
        return

    # Data preprocessing (recalculate to ensure consistency)
    # 1. Calculate magnitude and phase
    magnitude = np.abs(h_data_complex) # (N, Latent, Freq)
    phase = np.angle(h_data_complex)   # (N, Latent, Freq)
    
    # 2. Calculate total energy for each Latent dimension (sum over samples and frequencies)
    energy_per_latent = np.sum(np.square(magnitude), axis=(0, 2))
    
    # 3. Get sorting indices (descending order)
    sorted_indices = np.argsort(energy_per_latent)[::-1]
    
    # 4. Rearrange data
    h_sorted_mag = magnitude[:, sorted_indices, :]
    h_sorted_phase = phase[:, sorted_indices, :]
    
    # 5. Calculate mean amplitude spectrum for heatmap (Latent, Freq)
    mean_spectrum = np.mean(h_sorted_mag, axis=0)
    
    if save_path is None:
        save_path = npz_path.replace('.npz', '_reproduced.png')

    # --- Start plotting logic ---
    fig = plt.figure(figsize=(12, 10))
    gs = gridspec.GridSpec(3, 2, height_ratios=[3, 1, 1.5], width_ratios=[4, 1])
    
    # --- Panel A: Frequency vs Latent Dimension (Heatmap) ---
    ax_heatmap = fig.add_subplot(gs[0, 0])
    
    num_latent = mean_spectrum.shape[0]
    num_freq = mean_spectrum.shape[1]
    
    # Use logarithmic scale to display magnitude
    im = ax_heatmap.imshow(mean_spectrum.T, aspect='auto', origin='lower', 
                           cmap='inferno', norm=LogNorm(vmin=np.max(mean_spectrum)*1e-3, vmax=np.max(mean_spectrum)),
                           extent=[0.5, num_latent + 0.5, 0.5, num_freq + 0.5])
    
    ax_heatmap.set_ylabel(r'Frequency $k$ (wavenumber)', fontsize=12)
    ax_heatmap.set_xlabel(r'Latent Index (Sorted by Energy)', fontsize=12)
    
    ax_heatmap.set_xlim(0.5, num_latent + 0.5)
    ax_heatmap.set_xticks(np.arange(1, num_latent + 1))
    
    # Y-axis tick handling
    if num_freq <= 20:
        y_ticks = np.arange(1, num_freq + 1, 1)
    elif num_freq <= 60:
        y_ticks = np.arange(0, num_freq + 1, 10)
        y_ticks[0] = 1
    else:
        y_ticks = np.arange(0, num_freq + 1, 10)
        y_ticks[0] = 1
    ax_heatmap.set_yticks(y_ticks)
    
    ax_heatmap.set_title(r'(a) Spectral Decomposition in Latent Space', loc='left', fontsize=14)
    
    # --- Panel B: Energy Decay (Scree Plot) ---
    ax_energy = fig.add_subplot(gs[0, 1], sharey=None)
    latent_indices_1based = np.arange(1, num_latent + 1)
    ax_energy.plot(latent_indices_1based, energy_per_latent[sorted_indices], color='k', marker='o', markersize=3, lw=1)
    ax_energy.set_yscale('log')
    ax_energy.set_xlabel('Latent Index')
    ax_energy.set_ylabel(r'Total Energy $\Sigma |h|^2$')
    ax_energy.set_title(r'(b) Energy Decay', loc='left', fontsize=12)
    ax_energy.grid(True, which="both", ls="-", alpha=0.2)
    ax_energy.set_xticks(latent_indices_1based)
    ax_energy.set_xlim(0.5, num_latent + 0.5)
    
    # Add Colorbar
    cbar = plt.colorbar(im, ax=ax_heatmap, fraction=0.046, pad=0.04)
    cbar.set_label(r'Magnitude $|h|$ (log scale)', rotation=270, labelpad=15)

    # --- Panel C: Phase Distributions for Top 3 Modes ---
    ax_phase_row = fig.add_subplot(gs[2, :])
    ax_phase_row.axis('off')
    ax_phase_row.set_title(r'(c) Phase Distributions of Top 3 Energetic Modes (at peak freq)', loc='left', fontsize=14)
    
    gs_inner = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[2, :])
    
    for i in range(3):
        if i >= num_latent:
            break
            
        peak_freq_idx = np.argmax(mean_spectrum[i, :])
        phase_samples = h_sorted_phase[:, i, peak_freq_idx]
        
        ax_polar = fig.add_subplot(gs_inner[i], projection='polar')
        n_bins = 30
        hist, bin_edges = np.histogram(phase_samples, bins=n_bins, range=(-np.pi, np.pi))
        
        width = (2 * np.pi) / n_bins
        ax_polar.bar(bin_edges[:-1], hist, width=width, bottom=0.0, color=f'C{i}', alpha=0.7, edgecolor='k')
        
        ax_polar.set_title(f'Mode #{i+1}\nFreq={peak_freq_idx}', fontsize=10, y=1.05)
        ax_polar.set_yticks([])
        
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"PRL-level physical analysis plot saved: {save_path}")
    
    # Plot supplementary Latent dynamics figure
    plot_latent_dynamics(h_data_complex, sorted_indices, mean_spectrum, 
                         save_path.replace(".png", "_dynamics.png"),
                         u_data=u_data, block_info=block_info)


def plot_difference_and_joint_pdf_combined(npz_path, save_path=None, pdf_steps=None):
    """
    Read npz data and plot on the same figure:
    Row 1: True Spatiotemporal
    Row 2: Reconstructed Spatiotemporal
    Row 3: True Joint PDF | Reconstructed Joint PDF | Colorbar
    
    Args:
        npz_path: Data file path
        save_path: Save path
        pdf_steps: Number of time steps for calculating Joint PDF (None means use all data)
    """
    if not os.path.exists(npz_path):
        print(f"Error: File does not exist {npz_path}")
        return

    print(f"Processing Combined Plot: {npz_path}")
    try:
        data = np.load(npz_path)
        t_true = data['t_true']
        u_true = data['u_true']
        t_pred = data['t_pred']
        u_pred = data['u_pred']
    except Exception as e:
        print(f"Failed to load npz: {e}")
        return
        
    if save_path is None:
        save_path = npz_path.replace('.npz', '_combined_figure.png')
    
    # --- Extract only first plot_steps for spatiotemporal plot ---
    plot_steps = 1000
    if u_true.shape[1] > plot_steps:
        u_true_plot = u_true[:, :plot_steps]
        t_true_plot = t_true[:plot_steps]
    else:
        u_true_plot = u_true
        t_true_plot = t_true
        
    if u_pred.shape[1] > plot_steps:
        u_pred_plot = u_pred[:, :plot_steps]
        t_pred_plot = t_pred[:plot_steps]
    else:
        u_pred_plot = u_pred
        t_pred_plot = t_pred
    
    # --- Joint PDF calculation preparation ---
    # Decide how much data to use based on pdf_steps parameter
    if pdf_steps is not None and pdf_steps > 0:
        max_steps = min(u_true.shape[1], u_pred.shape[1])
        if pdf_steps < max_steps:
            u_true_pdf = u_true[:, :pdf_steps]
            u_pred_pdf = u_pred[:, :pdf_steps]
            print(f"Joint PDF using first {pdf_steps} steps")
        else:
            u_true_pdf = u_true
            u_pred_pdf = u_pred
            print(f"Joint PDF using all {max_steps} steps (pdf_steps >= data length)")
    else:
        u_true_pdf = u_true
        u_pred_pdf = u_pred
        print(f"Joint PDF using all {min(u_true.shape[1], u_pred.shape[1])} steps")
    
    # Calculate seamless derivative parameters
    N = u_true.shape[0]
    dx = 22.0 / 64.0
    bins = 360 # Slightly reduce resolution for faster plotting
    sigma_bins = 0.8
    block_size = 20 
    overlap = 0.5
    space_axis = 0 
    
    # Calculate derivatives (True)
    ux_true_full, uxx_true_full, mask_true = derivatives_blockwise_seamless(u_true_pdf, block_size, overlap, dx=dx, axis=space_axis)
    ux_t_flat, uxx_t_flat = flatten_masked_pairs(ux_true_full, uxx_true_full, mask_true)
    
    # Calculate derivatives (Pred)
    ux_pred_full, uxx_pred_full, mask_pred = derivatives_blockwise_seamless(u_pred_pdf, block_size, overlap, dx=dx, axis=space_axis)
    ux_p_flat, uxx_p_flat = flatten_masked_pairs(ux_pred_full, uxx_pred_full, mask_pred)

    # Unify coordinate range
    all_x = np.concatenate([ux_t_flat, ux_p_flat])
    all_y = np.concatenate([uxx_t_flat, uxx_p_flat])
    x_lo, x_hi = np.percentile(all_x, [0.5, 99.5])
    y_lo, y_hi = np.percentile(all_y, [0.5, 99.5])
    R = float(max(abs(x_lo), abs(x_hi), abs(y_lo), abs(y_hi))) * 1.1
    
    # Fix display range to [-3, 3], consistent with reference figure
    R_display = R  # Adjust this value to control PDF plot display range
    
    xedges = np.linspace(-R, R, bins)
    yedges = np.linspace(-R, R, bins)
    X, Y = np.meshgrid(xedges, yedges, indexing="xy")
    
    # Calculate probability P
    P_true = hist2d_prob(ux_t_flat, uxx_t_flat, xedges, yedges, sigma_bins=sigma_bins)
    P_pred = hist2d_prob(ux_p_flat, uxx_p_flat, xedges, yedges, sigma_bins=sigma_bins)
    
    # Colorbar Range
    allP = np.concatenate([P_true.ravel(), P_pred.ravel()])
    allP = allP[allP > 0]
    if allP.size == 0:
        pmin, pmax = 1e-7, 1e-3
    else:
        P_VMIN_VMAX = (1e-6, 1e-3)
        # P_VMIN_VMAX = (1e-3, 1) # Based on previous user modifications
        pmin, pmax = P_VMIN_VMAX

    norm = LogNorm(vmin=pmin, vmax=pmax)
    cmap = "inferno"
    cmap_base = plt.get_cmap(cmap)
    cmap_nan = ListedColormap(cmap_base(np.linspace(0, 1, 256)))
    cmap_nan.set_bad("white")

    # --- Start plotting layout ---
    # 5 row layout: 
    # Row 0: True Spatiotemporal (col 0-1) | Colorbar (col 2)
    # Row 1: Pred Spatiotemporal (col 0-1) | Colorbar (col 2)
    # Row 2: Error Spatiotemporal (col 0-1) | Colorbar (col 2)
    # Row 3: Error Curve (col 0-1)
    # Row 4: Joint PDF True (col 0) | Joint PDF Pred (col 1) | Colorbar (col 2)
    
    fig = plt.figure(figsize=(10, 14), dpi=300)
    # Height ratios: spatiotemporal plots slightly higher, PDF plots square
    gs = gridspec.GridSpec(5, 3, height_ratios=[1, 1, 1, 0.6, 1.2], width_ratios=[1, 1, 0.05], wspace=0.15, hspace=0.4)
    
    colors  = sns.diverging_palette(240, 10, n=9,  as_cmap=True)
    colors2 = sns.diverging_palette(240, 10, n=41)
    colors2 = ListedColormap(colors2[20:])

    # 1. True Spatiotemporal
    ax_st_true = fig.add_subplot(gs[0, :-1])    
    cax1 = fig.add_subplot(gs[0, 2])       # Colorbar in the last column
    im_st1 = ax_st_true.pcolormesh(t_true_plot, np.linspace(-11, 11, N), u_true_plot,
                                   shading="gouraud", cmap=colors, vmin=-3, vmax=3)
    ax_st_true.set_ylabel(r'$x$')
    ax_st_true.set_title(r'(a) True Spatiotemporal Evolution', loc='left', fontsize=12)
    ax_st_true.set_xticklabels([]) # Hide x-axis labels
    
    cbar1 = fig.colorbar(im_st1, cax=cax1)
    cbar1.set_label(r'$u$', fontsize=12)
    
    # 2. Pred Spatiotemporal
    ax_st_pred = fig.add_subplot(gs[1, :-1]) # Cross two columns
    cax2 = fig.add_subplot(gs[1, 2])       # Colorbar
    im_st2 = ax_st_pred.pcolormesh(t_pred_plot, np.linspace(-11, 11, N), u_pred_plot,
                                   shading="gouraud", cmap=colors, vmin=-3, vmax=3)
    ax_st_pred.set_ylabel(r'$x$')
    ax_st_pred.set_title(r'(b) Reconstructed Spatiotemporal Evolution', loc='left', fontsize=12)
    ax_st_pred.set_xticklabels([]) # Hide x-axis labels

    cbar2 = fig.colorbar(im_st2, cax=cax2)
    cbar2.set_label(r'$\tilde{u}$', fontsize=12)

    # 3. Error Spatiotemporal
    ax_st_err = fig.add_subplot(gs[2, :-1]) # Cross two columns
    cax3 = fig.add_subplot(gs[2, 2])      # Colorbar
    im_st3 = ax_st_err.pcolormesh(t_pred_plot, np.linspace(-11, 11, N), np.abs(u_pred_plot-u_true_plot),
                                   shading="gouraud", cmap=colors2, vmin=0, vmax=3)
    ax_st_err.set_ylabel(r'$x$')
    ax_st_err.set_title(r'(c) Absolute Error $|u - \tilde{u}|$', loc='left', fontsize=12)
    ax_st_err.set_xticklabels([]) # Hide x-axis labels

    cbar3 = fig.colorbar(im_st3, cax=cax3)
    cbar3.set_label(r'$|u - \tilde{u}|$', fontsize=12)

    # 4. Error Curve
    ax_curve = fig.add_subplot(gs[3, :-1]) # Span first two columns, aligned with above
    ax_curve.plot(t_pred_plot, np.linalg.norm(u_pred_plot-u_true_plot, axis=0), 'k-', lw=1)
    ax_curve.set_ylabel(r'$||u-\tilde{u}||$')
    ax_curve.set_xlabel(r'$t$')
    ax_curve.set_title(r'(d) L2 Error Norm', loc='left', fontsize=12)
    ax_curve.set_xlim(t_pred_plot[0], t_pred_plot[-1])

    # 5. Joint PDF (use nested GridSpec for centered layout: left margin | PDF True | gap | PDF Pred | Colorbar | right margin)
    pdf_gs = gridspec.GridSpecFromSubplotSpec(
        1, 6,
        subplot_spec=gs[4, :],
        width_ratios=[0.15, 1.0, 0.08, 1.0, 0.06, 0.15],  # Symmetric left/right margins
        wspace=0.25
    )

    # 5.1 Joint PDF True
    ax_pdf_true = fig.add_subplot(pdf_gs[0, 1])
    im_pdf_true = ax_pdf_true.pcolormesh(X, Y, P_true.T, cmap=cmap_nan, norm=norm, shading="auto", rasterized=True)
    ax_pdf_true.set_title(r"(e) True Joint PDF", loc='left', fontsize=12)
    ax_pdf_true.set_xlabel(r"$u_x$")
    ax_pdf_true.set_ylabel(r"$u_{xx}$")
    ax_pdf_true.set_aspect("equal")
    ax_pdf_true.set_xlim(-R_display, R_display)
    ax_pdf_true.set_ylim(-R_display, R_display)
    
    # 5.2 Joint PDF Pred
    ax_pdf_pred = fig.add_subplot(pdf_gs[0, 3])
    im_pdf_pred = ax_pdf_pred.pcolormesh(X, Y, P_pred.T, cmap=cmap_nan, norm=norm, shading="auto", rasterized=True)
    ax_pdf_pred.set_title(r"(f) Reconstructed Joint PDF", loc='left', fontsize=12)
    ax_pdf_pred.set_xlabel(r"$u_x$")
    ax_pdf_pred.set_yticklabels([])
    ax_pdf_pred.set_aspect("equal")
    ax_pdf_pred.set_xlim(-R_display, R_display)
    ax_pdf_pred.set_ylim(-R_display, R_display)
    
    # 5.3 PDF Colorbar (adjacent to right of PDF Pred)
    cax = fig.add_subplot(pdf_gs[0, 4])
    cbar = fig.colorbar(im_pdf_true, cax=cax)
    cbar.set_label(r"$P(u_x, u_{xx})$", fontsize=12)
    try:
        from matplotlib.ticker import LogLocator, LogFormatterMathtext
        cbar.locator = LogLocator(base=10)
        cbar.formatter = LogFormatterMathtext(base=10)
        cbar.update_ticks()
    except:
        pass
        
    plt.savefig(save_path, bbox_inches="tight")
    
    # Save as EPS vector graphics
    if save_path.endswith('.png'):
        eps_path = save_path.replace('.png', '.eps')
        plt.savefig(eps_path, format='eps', bbox_inches="tight")
        print(f"Combined figure saved (EPS): {eps_path}")
        
    plt.close()
    print(f"Combined figure saved (PNG): {save_path}")

def plot_difference_from_npz(npz_path, save_path=None, pdf_steps=None):
    """
    Old compatibility interface, now redirects to new combined plotting function
    
    Args:
        pdf_steps: Number of time steps for calculating Joint PDF (None means use all data)
    """
    plot_difference_and_joint_pdf_combined(npz_path, save_path, pdf_steps=pdf_steps)
    


def main():
    parser = argparse.ArgumentParser(description='Regenerate paper figures from saved .npz data')
    parser.add_argument('--target', type=str, default='./checkpoints_cnn_spatial_a1_20_robust_L22/trunc8_Nblock120_freq61', 
                        help='Directory path containing .npz files or single .npz file path')
    parser.add_argument('--pdf_steps', type=int, default=None,
                        help='Number of time steps for plotting Joint PDF (default: None means all data)')
    args = parser.parse_args()
    
    target_path = args.target
    pdf_steps = args.pdf_steps
    
    if os.path.isfile(target_path) and target_path.endswith('.npz'):
        # Process single file
        if 'timefreq' in os.path.basename(target_path) or 'latent' in os.path.basename(target_path):
            plot_timefreq_learning_from_npz(target_path)
        elif 'comparison' in os.path.basename(target_path) or 'diff' in os.path.basename(target_path):
            plot_difference_from_npz(target_path, pdf_steps=pdf_steps)
        else:
            print(f"Unknown file type: {target_path}, please check file name contains 'timefreq' or 'comparison'")
            
    elif os.path.isdir(target_path):
        # Scan directory
        print(f"Scanning directory: {target_path}")
        files = [f for f in os.listdir(target_path) if f.endswith('.npz')]
        count = 0
        for f in files:
            full_path = os.path.join(target_path, f)
            
            # Determine file type based on filename features
            if 'timefreq' in f or 'latent_representation' in f:
                plot_timefreq_learning_from_npz(full_path)
                count += 1
            elif 'cnn_spectral_ae_comparison' in f or 'difference' in f:
                plot_difference_from_npz(full_path, pdf_steps=pdf_steps)
                count += 1
        
        if count == 0:
            print("No matching .npz data files found. Please ensure filenames contain 'timefreq' or 'comparison'.")
    else:
        print(f"Invalid path: {target_path}")

if __name__ == "__main__":
    main()
