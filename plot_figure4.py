
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from matplotlib.colors import Normalize
import os
from scipy.interpolate import griddata

# ========== Global plotting style settings ==========
try:
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['mathtext.fontset'] = 'custom'
    plt.rcParams['mathtext.rm'] = 'Times New Roman'
    plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
    plt.rcParams['axes.unicode_minus'] = False 
except:
    pass

def calculate_isi(d_vals, mse_vals):
    """Calculate local logarithmic decay rate gamma(d)"""
    gamma_vals = []
    d_targets = []
    sorted_indices = np.argsort(d_vals)
    d_vals = d_vals[sorted_indices]
    mse_vals = mse_vals[sorted_indices]
    
    for i in range(1, len(d_vals)):
        d_curr, d_prev = d_vals[i], d_vals[i-1]
        mse_curr, mse_prev = mse_vals[i], mse_vals[i-1]
        if mse_curr <= 0 or mse_prev <= 0:
            gamma_vals.append(np.nan)
        else:
            numerator = np.log(mse_curr) - np.log(mse_prev)
            denominator = np.log(d_curr) - np.log(d_prev)
            gamma = -numerator / denominator if abs(denominator) > 1e-10 else np.nan
            gamma_vals.append(gamma)
        d_targets.append(d_curr)
    return np.array(d_targets), np.array(gamma_vals)

def get_phase_data(file_path):
    """Load and process single phase diagram data"""
    if not os.path.exists(file_path):
        return None
    df = pd.read_excel(file_path)
    df.columns = [str(c).lower().strip() for c in df.columns]
    trunc_col = next((c for c in df.columns if 'trunc' in c), None)
    # Try to match two possible column name formats: 'mse_ae=' or simply 'block=', enhance robustness
    block_cols = [c for c in df.columns if 'mse_ae=' in c or 'block=' in c]
    
    if not trunc_col or not block_cols:
        print(f"Skipping {file_path}: Could not identify 'trunc' or data columns.")
        return None

    def get_block_val(col_name):
        try: 
            # Try to parse block=X or mse_ae=X
            if 'block=' in col_name:
                return float(col_name.split('block=')[1].split(' ')[0])
            if 'mse_ae=' in col_name:
                # Extract float number, e.g. "mse_ae=0.9" -> 0.9
                part = col_name.split('mse_ae=')[1]
                import re
                # Match floating point number
                match = re.search(r'\d+\.?\d*', part)
                return float(match.group()) if match else 0.0
        except: return 0.0
    block_cols.sort(key=get_block_val)
    
    X_list, Y_list, Z_list = [], [], []
    df = df.sort_values(trunc_col)
    d_vals_all = df[trunc_col].values
    
    # Modification: Y-axis directly uses energy percentage (0.5 -> 50)
    for col in block_cols:
        val = get_block_val(col)
        T_val = val * 100  # 0.5 -> 50
        
        d_gamma, gamma = calculate_isi(d_vals_all, df[col].values)
        for d, g in zip(d_gamma, gamma):
            if not np.isnan(g):
                X_list.append(d); Y_list.append(T_val); Z_list.append(g)
    
    # Convert to arrays
    X, Y, Z = np.array(X_list), np.array(Y_list), np.array(Z_list)
    
    # Key: remove duplicates to prevent griddata errors
    # Combine (X, Y) as complex numbers for uniqueness check, or use lexsort
    coords = np.vstack((X, Y)).T
    unique_coords, idx = np.unique(coords, axis=0, return_index=True)
    
    if len(unique_coords) < len(coords):
        # print(f"Removed {len(coords) - len(unique_coords)} duplicate points.")
        X = X[idx]
        Y = Y[idx]
        Z = Z[idx]
        
    return X, Y, Z

def plot_combined_figure():
    configs = [
        (22, 8), (44, 16), (58, 22), (66, 24), (88, 32), (96, 35)
    ]
    base_dir = "figures"
    labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
    
    # PRL two-column width is about 7 inches, adjust height to make top and bottom rows more compact
    fig = plt.figure(figsize=(15, 8.5), dpi=300)
    # Use GridSpec with fine spacing control, reduce hspace
    gs = gridspec.GridSpec(2, 3, wspace=0.35, hspace=0.2)
    
    # --- (a)-(e) Phase Diagrams ---
    for i, (L_val, d_star) in enumerate(configs):
        ax = fig.add_subplot(gs[i // 3, i % 3])
        # Core: set subplot box to square
        ax.set_box_aspect(1)
        
        # Note: user-provided filename contains space "cut freq.xlsx"
        file_path = os.path.join(base_dir, f"sweep_summary-L{L_val}-cut_freq.xlsx")
        data = get_phase_data(file_path)
        
        if data is None:
            ax.text(0.5, 0.5, f"L={L_val} Data Missing", ha='center')
            continue
            
        X, Y, Z = data
        xi = np.linspace(X.min(), X.max(), 200)
        yi = np.linspace(Y.min(), Y.max(), 200)
        XI, YI = np.meshgrid(xi, yi)
        ZI = griddata((X, Y), Z, (XI, YI), method='cubic')
        
        vmax = np.percentile(np.abs(Z), 99)
        heatmap = ax.pcolormesh(XI, YI, ZI, cmap='RdBu_r', vmin=-vmax, vmax=vmax, shading='auto')
        
        # Adjust colorbar height to align with square
        cbar = plt.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)
        cbar.set_label(r'$\gamma(d, \tau)$', fontsize=9)
        
        ax.contour(XI, YI, ZI, levels=[0], colors='black', linewidths=1.2)
        ax.axvline(x=d_star, color='forestgreen', linestyle='--', linewidth=1.5, 
                   label=f'$d^*={d_star}$', alpha=0.9)
        
        ax.set_xlabel(r'Latent Dimension $d$', fontsize=10)
        ax.set_ylabel(r'Energy Percentage (%)', fontsize=10)
        # Fine-tune title position, remove bold
        ax.set_title(f'{labels[i]} $L={L_val}$', loc='left', fontsize=12, pad=8)
        
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.tick_params(axis='both', which='major', labelsize=9, direction='out', width=0.8, length=4)
        ax.legend(loc='upper left', fontsize=8, framealpha=0.4, handlelength=1.5)

    # # --- (f) Extensivity Scaling Law ---
    # ax_f = fig.add_subplot(gs[1, 2])
    # ax_f.set_box_aspect(1) # Also set to square
    
    # L_vals = np.array([22, 44, 66, 88, 110])
    # d_stars = np.array([8, 16, 24, 32, 40])
    
    # k_zero = np.sum(L_vals * d_stars) / np.sum(L_vals**2)
    # y_pred = k_zero * L_vals
    # r_squared = 1 - np.sum((d_stars - y_pred)**2) / np.sum((d_stars - np.mean(d_stars))**2)
    
    # L_range = np.linspace(0, 125, 100)
    # ax_f.plot(L_range, k_zero * L_range, 'k--', alpha=0.6, lw=1.2, label=f'$d^* \\approx {k_zero:.3f} L$')
    # ax_f.errorbar(L_vals, d_stars, yerr=1.2, fmt='o', markersize=6, capsize=3, 
    #               color='#D62728', ecolor='gray', elinewidth=1, label='Identified $d^*$')
    
    # for L, d in zip(L_vals, d_stars):
    #     ax_f.text(L + 4, d - 1.5, f'({L},{d})', fontsize=8)
        
    # ax_f.set_xlabel('System Size $L$', fontsize=10)
    # ax_f.set_ylabel('Manifold Dimension $d^*$', fontsize=10)
    # # 移除加粗
    # ax_f.set_title(f'{labels[5]} Scaling Law', loc='left', fontsize=12, pad=8)
    # ax_f.set_xlim(0, 130); ax_f.set_ylim(0, 50)
    # ax_f.grid(True, linestyle=':', alpha=0.4)
    # ax_f.tick_params(axis='both', which='major', labelsize=9, direction='out', width=0.8, length=4)
    
    # # --- 关键：为了让图(f)的大小与带colorbar的相图对齐，添加一个不可见的colorbar占位 ---
    # # 创建一个不可见的映射对象
    # sm = plt.cm.ScalarMappable(cmap='RdBu_r', norm=plt.Normalize(vmin=0, vmax=1))
    # sm.set_array([])
    # dummy_cbar = plt.colorbar(sm, ax=ax_f, fraction=0.046, pad=0.04)
    # dummy_cbar.ax.set_visible(False) # 隐藏颜色条内容
    
    # # R^2 标注位置优化
    # ax_f.text(0.05, 0.85, f'$R^2 = {r_squared:.4f}$', transform=ax_f.transAxes, 
    #           fontsize=9, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
    # ax_f.legend(loc='lower right', fontsize=8, framealpha=0.4)

    # 移除自动 tight_layout，使用 GridSpec 参数手动控制
    # plt.tight_layout() 
    save_path = "Phase_Extensivity_Combined_cut_freq.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=400)
    plt.savefig(save_path.replace('.png', '.eps'), format='eps', bbox_inches='tight')
    plt.close()
    print(f"Combined figure saved to {save_path} and EPS format.")

if __name__ == "__main__":
    plot_combined_figure()

