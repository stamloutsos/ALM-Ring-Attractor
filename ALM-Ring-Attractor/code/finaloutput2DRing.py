import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, circmean, circstd, wilcoxon, mannwhitneyu
from sklearn.decomposition import PCA
from pathlib import Path
import pandas as pd
from scipy.special import i0
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION - RING ATTRACTOR + CORRECT T_THETA
# ============================================================================
OUTPUT_DIR = Path(r"E:\expiramentsclinic\000060\Outputresults_RING_LANGEVIN_FINAL")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"
DATA_DIR = OUTPUT_DIR / "data"

for d in [FIGURES_DIR, TABLES_DIR, DATA_DIR]:
    d.mkdir(exist_ok=True)

MIN_SELECTIVE_UNITS = 4
MIN_TRIALS = 20
MIN_ERROR_TRIALS = 10
RAYLEIGH_R_THRESHOLD = 0.3
RADIAL_CV_THRESHOLD = 0.5

# ============================================================================
# FUNCTIONS
# ============================================================================

def rayleigh_test(theta):
    if len(theta) < 10:
        return 1.0, 1.0
    C = np.mean(np.cos(theta))
    S = np.mean(np.sin(theta))
    R = np.sqrt(C**2 + S**2)
    n = len(theta)
    z = n * R**2
    p = np.exp(-z) * (1 + (2*z - z**2) / (4*n) - (24*z - 132*z**2 + 76*z**3 - 9*z**4) / (288*n**2))
    return R, min(p, 1.0)

def vonmises_fit(theta):
    mu = circmean(theta)
    R = np.sqrt(np.mean(np.cos(theta))**2 + np.mean(np.sin(theta))**2)
    if R < 0.53:
        kappa = 2*R + R**3 + (5*R**5)/6
    elif R < 0.85:
        kappa = -0.4 + 1.39*R + 0.43/(1-R)
    else:
        kappa = 1/(R**3 - 4*R**2 + 3*R + 1e-9)
    return mu, kappa, R

def compute_T_theta(X_2d):
    """
    Σωστός υπολογισμός T_θ: MSD των γωνιών μεταξύ διαδοχικών trials
    """
    if len(X_2d) < 3:
        return np.nan
    theta = np.arctan2(X_2d[:,1], X_2d[:,0])
    theta_unwrap = np.unwrap(theta)
    dtheta = np.diff(theta_unwrap)
    MSD = np.mean(dtheta**2)
    T_theta = MSD / 2.0 # dt=1 trial
    return T_theta

def test_ring_attractor(X_2d, labels):
    if len(X_2d) < MIN_TRIALS:
        return None

    r = np.linalg.norm(X_2d, axis=1)
    theta = np.arctan2(X_2d[:,1], X_2d[:,0])

    R_rayleigh, p_rayleigh = rayleigh_test(theta)
    mu_vm, kappa_vm, R_vm = vonmises_fit(theta)

    r_mean, r_std = np.mean(r), np.std(r)
    r_cv = r_std / (r_mean + 1e-9)
    is_r_tight = r_cv < RADIAL_CV_THRESHOLD

    theta_left = theta[labels == 0]
    theta_right = theta[labels == 1]
    if len(theta_left) > 5 and len(theta_right) > 5:
        mu_left = circmean(theta_left)
        mu_right = circmean(theta_right)
        angular_sep = np.abs(np.arctan2(np.sin(mu_right - mu_left), np.cos(mu_right - mu_left)))
    else:
        angular_sep = 0

    total_var = np.sum(np.var(X_2d, axis=0))
    pc1_var = np.var(X_2d[:,0])
    pc2_var = np.var(X_2d[:,1])
    var_ratio = (pc1_var + pc2_var) / (total_var + 1e-9)

    theta_range = np.max(theta) - np.min(theta)
    is_arc = theta_range < np.pi * 1.5

    T_theta = compute_T_theta(X_2d)

    return {
        'R_rayleigh': R_rayleigh,
        'p_rayleigh': p_rayleigh,
        'is_uniform': R_rayleigh < RAYLEIGH_R_THRESHOLD,
        'kappa_vm': kappa_vm,
        'mu_vm': mu_vm,
        'r_mean': r_mean,
        'r_std': r_std,
        'r_cv': r_cv,
        'is_r_tight': is_r_tight,
        'angular_sep': angular_sep,
        'var_ratio': var_ratio,
        'theta_range': theta_range,
        'is_arc': is_arc,
        'n_trials': len(X_2d),
        'T_theta': T_theta
    }

def analyze_session_ring(nwb_path):
    try:
        with h5py.File(nwb_path, 'r') as h5:
            trials = h5['intervals/trials']
            instruction = trials['trial_instruction'][:].astype(str)
            outcomes = trials['outcome'][:].astype(str)

            hit_mask = outcomes == 'hit'
            error_mask = (outcomes == 'miss') | (outcomes == 'ignore')

            if np.sum(hit_mask) < MIN_TRIALS:
                return None

            events_data = h5['acquisition/LabeledEvents/data'][:]
            events_ts = h5['acquisition/LabeledEvents/timestamps'][:]
            labels = list(h5['acquisition/LabeledEvents/data'].attrs['labels'])
            go_times = events_ts[events_data == labels.index('go_start_times')]

            units = h5['units']
            spike_times = units['spike_times'][:]
            spike_times_idx = units['spike_times_index'][:]
            n_units = len(units['id'][:])

            n_trials = len(trials['start_time'][:])
            delay_rates = np.full((n_trials, n_units), np.nan)

            for j in range(min(n_trials, len(go_times))):
                t_go = go_times[j]
                t0, t1 = t_go - 1.0, t_go - 0.1
                if t0 < 0: continue
                for k in range(n_units):
                    st_start = spike_times_idx[k-1] if k > 0 else 0
                    st_end = spike_times_idx[k]
                    st = spike_times[st_start:st_end]
                    count = np.sum((st >= t0) & (st < t1))
                    delay_rates[j, k] = count / 0.9

            valid = ~np.isnan(delay_rates).any(axis=1)
            if np.sum(valid) < 30: return None

            left_hit = (instruction == 'left') & hit_mask & valid
            right_hit = (instruction == 'right') & hit_mask & valid

            if np.sum(left_hit) < 5 or np.sum(right_hit) < 5: return None

            left_rates = delay_rates[left_hit]
            right_rates = delay_rates[right_hit]

            selective = []
            for k in range(n_units):
                if np.std(left_rates[:, k]) < 1e-6 or np.std(right_rates[:, k]) < 1e-6: continue
                _, p = ttest_ind(left_rates[:, k], right_rates[:, k], equal_var=False)
                if p < 0.05: selective.append(k)

            if len(selective) < MIN_SELECTIVE_UNITS: return None

            # PCA σε 2D για hit trials
            X_hit = delay_rates[hit_mask & valid][:, selective]
            y_hit = (instruction == 'right')[hit_mask & valid].astype(int)

            pca = PCA(n_components=2)
            X_hit_2d = pca.fit_transform(X_hit)

            ring_stats_hit = test_ring_attractor(X_hit_2d, y_hit)
            if ring_stats_hit is None: return None

            # Error trials - μόνο αν υπάρχουν αρκετά
            X_err_2d = np.array([])
            ring_stats_err = {'T_theta': np.nan}
            if np.sum(error_mask & valid) >= MIN_ERROR_TRIALS:
                X_err = delay_rates[error_mask & valid][:, selective]
                X_err_2d = pca.transform(X_err)
                ring_stats_err = test_ring_attractor(X_err_2d, np.zeros(len(X_err_2d)))

            return {
                'session': nwb_path.stem.split('_')[1],
                'n_units': len(selective),
                'n_trials_hit': np.sum(hit_mask & valid),
                'n_trials_err': np.sum(error_mask & valid),
                'pc1_var_exp': pca.explained_variance_ratio_[0],
                'pc2_var_exp': pca.explained_variance_ratio_[1],
                **{f'hit_{k}': v for k, v in ring_stats_hit.items()},
                **{f'err_{k}': v for k, v in ring_stats_err.items()},
                'X_hit_2d': X_hit_2d,
                'X_err_2d': X_err_2d,
                'y_hit': y_hit
            }

    except Exception as e:
        print(f" ERROR {nwb_path.stem}: {e}")
        return None

def main():
    print("\n" + "="*80)
    print("RING ATTRACTOR LANGEVIN TEST - CORRECTED T_THETA")
    print("="*80)
    print(f"Output: {OUTPUT_DIR}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    base = Path(r"E:\expiramentsclinic\000060")
    files = sorted(list(base.glob("sub-*/**/*.nwb")))
    print(f"\nFound {len(files)} NWB files")

    results = []

    for i, f in enumerate(files):
        print(f"\n[{i+1}/{len(files)}] {f.stem}", end="... ")
        res = analyze_session_ring(f)
        if res:
            results.append(res)
            print(f"R={res['hit_R_rayleigh']:.2f}, κ={res['hit_kappa_vm']:.2f}, r_cv={res['hit_r_cv']:.2f}, "
                  f"T_h={res['hit_T_theta']:.3f}, T_e={res['err_T_theta']:.3f}")
        else:
            print("rejected")

    df = pd.DataFrame([{k:v for k,v in r.items() if k not in ['X_hit_2d', 'X_err_2d', 'y_hit']} for r in results])

    if df.empty:
        print("\n" + "="*80)
        print("ΑΠΟΤΕΛΕΣΜΑ: 0/98 sessions με 2D δομή")
        print("="*80)
        return

    df.to_csv(DATA_DIR / "ring_results_final.csv", index=False)

    # Συγκεντρωτικά
    T_h = df['hit_T_theta'].values
    T_e = df['err_T_theta'].dropna().values

    print("\n" + "="*80)
    print(f"RESULTS: {len(df)}/98 sessions analyzed")
    print("="*80)
    print(f"Mean R_rayleigh = {df['hit_R_rayleigh'].mean():.3f} ± {df['hit_R_rayleigh'].std():.3f}")
    print(f"Uniform θ (<0.3): {np.sum(df['hit_R_rayleigh'] < 0.3)}/{len(df)} sessions")
    print(f"Mean kappa_vm = {df['hit_kappa_vm'].mean():.3f}")
    print(f"Mean r_cv = {df['hit_r_cv'].mean():.3f}")
    print(f"Tight radius (CV<0.5): {np.sum(df['hit_r_cv'] < 0.5)}/{len(df)} sessions")
    print(f"Mean angular_sep = {df['hit_angular_sep'].mean():.3f} rad = {df['hit_angular_sep'].mean()*180/np.pi:.1f}°")
    print(f"Sessions passing RING criteria: {np.sum((df['hit_R_rayleigh'] < 0.3) & (df['hit_r_cv'] < 0.5))}/{len(df)}")

    print(f"\n--- LANGEVIN TEMPERATURE ---")
    print(f"T_θ(hit) = {np.mean(T_h):.4f} ± {np.std(T_h):.4f}, N={len(T_h)}")
    if len(T_e) > 0:
        print(f"T_θ(error) = {np.mean(T_e):.4f} ± {np.std(T_e):.4f}, N={len(T_e)}")
        # Mann-Whitney U test - δεν απαιτεί paired samples
        stat, p = mannwhitneyu(T_e, T_h, alternative='greater')
        print(f"Mann-Whitney U test T_err > T_hit: p = {p:.3g}")
        if p < 0.01:
            print(">>> ΣΗΜΑΝΤΙΚΟ: T_θ(error) > T_θ(hit). Langevin επιβεβαιώνεται.")
        else:
            print(">>> ΠΡΟΣΟΧΗ: T_θ(error) <= T_θ(hit). Το μοντέλο θέλει επανεξέταση.")
    else:
        print("Δεν βρέθηκαν sessions με αρκετά error trials για σύγκριση T_θ")

    # LaTeX Table
    print("\n=== LATEX TABLE ===")
    print(r"\begin{table}[ht]")
    print(r"\centering")
    print(r"\caption{Ring attractor metrics. R: Rayleigh (0=uniform). κ: von Mises concentration. r\_cv: radial CV. T_θ: angular diffusion.}")
    print(r"\begin{tabular}{lccccccc}")
    print(r"\toprule")
    print(r"Session & N & R & κ & r\_cv & Sep(rad) & T_θ & Ring? \\")
    print(r"\midrule")
    for _, row in df.iterrows():
        ring_str = "Yes" if (row['hit_R_rayleigh'] < 0.3 and row['hit_r_cv'] < 0.5) else "No"
        print(f"{row['session']} & {int(row['n_trials_hit'])} & {row['hit_R_rayleigh']:.2f} & "
              f"{row['hit_kappa_vm']:.2f} & {row['hit_r_cv']:.2f} & {row['hit_angular_sep']:.2f} & "
              f"{row['hit_T_theta']:.3f} & {ring_str} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")

    # Figures
    fig, axes = plt.subplots(4, 5, figsize=(20, 16))
    axes = axes.flatten()
    for i, res in enumerate(results[:20]):
        ax = axes[i]
        X_2d = res['X_hit_2d']
        y = res['y_hit']
        ax.scatter(X_2d[y==0, 0], X_2d[y==0, 1], c='blue', alpha=0.5, s=10, label='Left')
        ax.scatter(X_2d[y==1, 0], X_2d[y==1, 1], c='red', alpha=0.5, s=10, label='Right')
        ax.set_title(f"{res['session']}\nR={res['hit_R_rayleigh']:.2f}")
        ax.set_aspect('equal')
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "ring_gallery.png", dpi=200)
    plt.close()

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes[0,0].hist(df['hit_R_rayleigh'], bins=20, alpha=0.7)
    axes[0,0].axvline(0.3, color='r', linestyle='--', label='Threshold')
    axes[0,0].set_xlabel('Rayleigh R')
    axes[0,0].set_title('Angular Uniformity')
    axes[0,0].legend()

    axes[0,1].hist(df['hit_kappa_vm'], bins=20, alpha=0.7)
    axes[0,1].set_xlabel('von Mises κ')
    axes[0,1].set_title('Angular Concentration')

    axes[0,2].hist(df['hit_r_cv'], bins=20, alpha=0.7)
    axes[0,2].axvline(0.5, color='r', linestyle='--', label='Threshold')
    axes[0,2].set_xlabel('Radial CV')
    axes[0,2].set_title('Radial Tightness')
    axes[0,2].legend()

    axes[1,0].scatter(df['hit_R_rayleigh'], df['hit_r_cv'], alpha=0.6)
    axes[1,0].axvline(0.3, color='r', linestyle='--')
    axes[1,0].axhline(0.5, color='r', linestyle='--')
    axes[1,0].set_xlabel('Rayleigh R')
    axes[1,0].set_ylabel('Radial CV')
    axes[1,0].set_title('Ring Criteria')

    axes[1,1].hist(df['hit_angular_sep'], bins=20, alpha=0.7)
    axes[1,1].set_xlabel('Angular Separation (rad)')
    axes[1,1].set_title('Left vs Right')

    axes[1,2].hist(df['pc1_var_exp'] + df['pc2_var_exp'], bins=20, alpha=0.7)
    axes[1,2].set_xlabel('PC1+PC2 Variance Explained')
    axes[1,2].set_title('2D Capture')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "ring_stats.png", dpi=300)
    plt.close()

    # Figure 3: T_theta comparison - μόνο για sessions με errors
    df_paired = df.dropna(subset=['err_T_theta'])
    if len(df_paired) > 0:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.scatter(df_paired['hit_T_theta'], df_paired['err_T_theta'], alpha=0.6, s=50)
        max_T = max(df_paired['hit_T_theta'].max(), df_paired['err_T_theta'].max())
        ax.plot([0, max_T], [0, max_T], 'k--', alpha=0.5, label='y=x')
        ax.set_xlabel('T_θ (Hit)')
        ax.set_ylabel('T_θ (Error)')
        ax.set_title(f'T_θ: p={p:.3g}, N={len(df_paired)}')
        ax.legend()
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "T_theta_comparison.png", dpi=300)
        plt.close()

    print(f"\nAll results saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()