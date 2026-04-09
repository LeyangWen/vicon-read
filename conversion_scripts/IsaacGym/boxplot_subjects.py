import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('always', message='.*LMM FALLBACK.*')

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['hatch.color'] = 'gray'

# ============================================================
# CONFIGURATION FLAGS
# ============================================================
SHOW_INDIVIDUAL_DOTS = True
SHOW_OUTLIERS = True
SHOW_MEAN_DIAMONDS = True
USE_HATCH = 'both'  # True = hatch only, False = color only, 'both' = light color + hatch

# ============================================================
# DATA - Load from CSV
# ============================================================
CSV_PATH = '/Users/leyangwen/Library/CloudStorage/OneDrive-Umich/isaac_3dsspp/intervention_eval_data/Claude_formatted_data_corrected.csv'
DATA_COLS = ['L4L5_Compression', 'Wrist', 'Elbow', 'Shoulder', 'Torso', 'Neck', 'Hip', 'Knee', 'Ankle', 'RF', 'LF', 'LI', 'RWL']

_df = pd.read_csv(CSV_PATH)

def _extract(source, terrain):
    sub = _df[(_df['source'] == source) & (_df['terrain'] == terrain)].sort_values('scenario')
    return sub[DATA_COLS].values.tolist()

gen_flat = _extract('Generated', 'flat')
gen_terrain = _extract('Generated', 'terrain')
s1_flat = _extract('S1', 'flat')
s1_terrain = _extract('S1', 'terrain')
s2_flat = _extract('S2', 'flat')
s2_terrain = _extract('S2', 'terrain')
s3_flat = _extract('S3', 'flat')
s3_terrain = _extract('S3', 'terrain')

COL_L4L5 = 0; COL_WRIST = 1; COL_ELBOW = 2
COL_SHOULDER = 3; COL_TORSO = 4; COL_NECK = 5; COL_HIP = 6
COL_KNEE = 7; COL_ANKLE = 8; COL_RF = 9; COL_LF = 10
COL_LI = 11; COL_RWL = 12



# ============================================================
# HELPERS
# ============================================================
def build_paired_data(s1_data, s2_data, s3_data, gen_data, col):
    """Build arrays for descriptive stats: each worker's value paired with the gen value for that scenario."""
    mocap_vals, gen_vals = [], []
    for i in range(len(gen_data)):
        for subj_data in [s1_data, s2_data, s3_data]:
            mocap_vals.append(subj_data[i][col])
            gen_vals.append(gen_data[i][col])
    return np.array(mocap_vals), np.array(gen_vals)

def build_lmm_dataframe(s1_data, s2_data, s3_data, gen_data, col):
    """Build a DataFrame for LMM: value ~ condition + (1|scenario).

    4 observations per scenario (S1, S2, S3 captured + 1 synthesized).
    No triplication of the generated value.
    """
    rows = []
    n_scenarios = len(gen_data)
    subjects = ['S1', 'S2', 'S3']
    subj_data_list = [s1_data, s2_data, s3_data]

    for i in range(n_scenarios):
        scenario_id = f'Sc{i+1}'
        for s_idx, subj in enumerate(subjects):
            rows.append({
                'value': subj_data_list[s_idx][i][col],
                'condition': 0,  # captured
                'scenario': scenario_id,
            })
        rows.append({
            'value': gen_data[i][col],
            'condition': 1,  # synthesized
            'scenario': scenario_id,
        })
    return pd.DataFrame(rows)

def run_lmm(s1_data, s2_data, s3_data, gen_data, col):
    """Run LMM: value ~ condition + (1|scenario). Returns (beta, p, cohen_d)."""
    df = build_lmm_dataframe(s1_data, s2_data, s3_data, gen_data, col)

    # Check if there's any variance
    if df['value'].std() < 1e-10:
        return 0.0, 1.0, 0.0

    try:
        model = smf.mixedlm("value ~ condition", df, groups=df["scenario"])
        result = model.fit(reml=True)
        beta = result.params['condition']
        p = result.pvalues['condition']
    except Exception as e:
        warnings.warn(
            f"\n{'!'*60}\n"
            f"  LMM FALLBACK: Welch's t-test used instead of LMM\n"
            f"  Reason: {e}\n"
            f"{'!'*60}",
            stacklevel=2
        )
        captured = df[df['condition'] == 0]['value']
        synthesized = df[df['condition'] == 1]['value']
        _, p = stats.ttest_ind(captured, synthesized, equal_var=False)
        beta = synthesized.mean() - captured.mean()

    # Cohen's d from all individual observations
    captured = df[df['condition'] == 0]['value'].values
    synthesized = df[df['condition'] == 1]['value'].values
    pooled_std = np.sqrt((np.var(captured, ddof=1) * (len(captured)-1) +
                          np.var(synthesized, ddof=1) * (len(synthesized)-1)) /
                         (len(captured) + len(synthesized) - 2))
    d = (synthesized.mean() - captured.mean()) / pooled_std if pooled_std > 0 else 0

    return beta, p, d

def sig_label(p):
    if p > 0.05: return 'n.s.'
    elif p > 0.01: return '*'
    elif p > 0.001: return '**'
    else: return '***'

def get_outlier_mask(vals):
    """Return boolean mask: True for outliers (beyond 1.5*IQR)."""
    q1, q3 = np.percentile(vals, [25, 75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5*iqr, q3 + 1.5*iqr
    return np.array([(v < lower or v > upper) for v in vals])

# ============================================================
# PLOT FUNCTION
# ============================================================
def make_boxplot(s1_data, s2_data, s3_data, gen_data, title_str, filename):
    colors_mocap = '#C0392B'
    colors_gen = '#1A5276'
    fill_mocap = '#F2D7D5'   # light pink for 'both' mode
    fill_gen = '#D4E6F1'     # light blue for 'both' mode
    hatch_mocap = ''
    hatch_gen = '////////'
    
    # No neck
    percentile_names = ['Wrist', 'Elbow', 'Shoulder', 'Torso', 'Hip', 'Knee', 'Ankle']
    percentile_cols = [COL_WRIST, COL_ELBOW, COL_SHOULDER, COL_TORSO, COL_HIP, COL_KNEE, COL_ANKLE]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5),
                                    gridspec_kw={'width_ratios': [1.4, 4], 'wspace': 0.3})
    
    # Outlier style: always black hollow circles (same size as data dots)
    flier_props = dict(marker='o', markerfacecolor='none', markeredgecolor='black',
                       markersize=4, linewidth=1.0) if SHOW_OUTLIERS else dict(marker='None')
    
    mean_props = dict(marker='D', markerfacecolor='black', markeredgecolor='black', markersize=6, zorder=10)
    
    def draw_box_and_dots(ax, vals, pos, color, hatch, is_compression=False):
        """Draw one box with optional hatch/color and dots."""
        bp = ax.boxplot([vals], positions=[pos], widths=0.3,
                         patch_artist=True, showmeans=SHOW_MEAN_DIAMONDS,
                         meanprops=mean_props,
                         medianprops=dict(color='black', linewidth=1.5),
                         whiskerprops=dict(linewidth=1.2),
                         capprops=dict(linewidth=1.2),
                         flierprops=flier_props)
        if USE_HATCH == 'both':
            fc = fill_mocap if color == colors_mocap else fill_gen
            bp['boxes'][0].set_facecolor(fc)
            bp['boxes'][0].set_edgecolor('black')
            bp['boxes'][0].set_hatch(hatch)
            bp['boxes'][0].set_alpha(0.9)
        elif USE_HATCH:
            bp['boxes'][0].set_facecolor('white')
            bp['boxes'][0].set_edgecolor('black')
            bp['boxes'][0].set_hatch(hatch)
            bp['boxes'][0].set_alpha(0.8)
        else:
            bp['boxes'][0].set_facecolor(color)
            bp['boxes'][0].set_alpha(0.6)

        if SHOW_INDIVIDUAL_DOTS:
            outlier_mask = get_outlier_mask(vals) if SHOW_OUTLIERS else np.zeros(len(vals), dtype=bool)
            dot_size = 20 if is_compression else 15
            for idx, v in enumerate(vals):
                if outlier_mask[idx]:
                    ax.scatter(pos, v, color=color, alpha=0.7, s=dot_size,
                               zorder=5, edgecolors='none')
                else:
                    jit = np.random.uniform(-0.08, 0.08)
                    ax.scatter(pos + jit, v, color=color, alpha=0.4, s=dot_size,
                               zorder=2, edgecolors='none')
    
    # ---- Left: L4/L5 ----
    mocap_comp, gen_comp = build_paired_data(s1_data, s2_data, s3_data, gen_data, COL_L4L5)
    
    draw_box_and_dots(ax1, mocap_comp, 0.8, colors_mocap, hatch_mocap, is_compression=True)
    draw_box_and_dots(ax1, gen_comp, 1.2, colors_gen, hatch_gen, is_compression=True)
    
    beta_comp, p_comp, d_comp = run_lmm(s1_data, s2_data, s3_data, gen_data, COL_L4L5)
    # Place significance at fixed top of L4/L5 panel
    y_min_comp = min(mocap_comp.min(), gen_comp.min())
    y_max_comp = max(mocap_comp.max(), gen_comp.max())
    comp_label_y = y_max_comp + 150
    sl_comp = sig_label(p_comp)
    if sl_comp != 'n.s.':
        ax1.plot([0.8,0.8,1.2,1.2], [comp_label_y, comp_label_y+40, comp_label_y+40, comp_label_y], color='black', linewidth=1.2)
        ax1.text(1.0, comp_label_y+50, sl_comp, ha='center', va='bottom', fontsize=11, fontweight='bold')
    else:
        ax1.text(1.0, comp_label_y+30, 'n.s.', ha='center', va='bottom', fontsize=11, color='gray')
    ax1.set_ylim(y_min_comp - 100, comp_label_y + 200)

    ax1.set_ylabel('Compression Force (N)', fontsize=18)
    ax1.set_xticks([1.0])
    ax1.set_xticklabels(['L4/L5'], fontsize=18)
    ax1.set_title('3D Low Back Compression', fontsize=18, fontweight='bold')
    ax1.tick_params(axis='y', labelsize=15)
    ax1.grid(axis='y', alpha=0.3); ax1.set_axisbelow(True)
    
    # ---- Right: Percentiles ----
    positions_mocap, positions_gen = [], []
    mocap_data_list, gen_data_list = [], []
    
    for i, (joint, col) in enumerate(zip(percentile_names, percentile_cols)):
        pos_m = i*1.6 + 0.8; pos_g = i*1.6 + 1.2
        positions_mocap.append(pos_m); positions_gen.append(pos_g)
        m, g = build_paired_data(s1_data, s2_data, s3_data, gen_data, col)
        mocap_data_list.append(m); gen_data_list.append(g)
    
    for i in range(len(percentile_names)):
        draw_box_and_dots(ax2, mocap_data_list[i], positions_mocap[i], colors_mocap, hatch_mocap)
        draw_box_and_dots(ax2, gen_data_list[i], positions_gen[i], colors_gen, hatch_gen)
    
    # Significance brackets via LMM - aligned at fixed top position
    all_pct_for_top = np.concatenate(mocap_data_list + gen_data_list)
    fixed_bracket_y = 103
    for i, (joint, col) in enumerate(zip(percentile_names, percentile_cols)):
        beta, p, d = run_lmm(s1_data, s2_data, s3_data, gen_data, col)
        pos_m, pos_g = positions_mocap[i], positions_gen[i]

        sl = sig_label(p)
        if sl != 'n.s.':
            ax2.plot([pos_m,pos_m,pos_g,pos_g], [fixed_bracket_y, fixed_bracket_y+0.6, fixed_bracket_y+0.6, fixed_bracket_y],
                     color='black', linewidth=1.2)
            ax2.text((pos_m+pos_g)/2, fixed_bracket_y+0.7, sl, ha='center', va='bottom', fontsize=13, fontweight='bold')
        else:
            ax2.text((pos_m+pos_g)/2, fixed_bracket_y+0.3, 'n.s.', ha='center', va='bottom', fontsize=11, color='gray')
    
    ax2.set_xticks([i*1.6+1.0 for i in range(len(percentile_names))])
    ax2.set_xticklabels(percentile_names, fontsize=18)
    ax2.set_ylabel('Strength Percent Capable (%)', fontsize=18)
    ax2.set_title('Joint Strength Percent Capable', fontsize=18, fontweight='bold')
    ax2.tick_params(axis='y', labelsize=15)
    ax2.grid(axis='y', alpha=0.3); ax2.set_axisbelow(True)
    
    all_pct = np.concatenate(mocap_data_list + gen_data_list)
    ax2.set_ylim(max(0, all_pct.min()-5), 110)

    # Legend
    if USE_HATCH == 'both':
        handles = [
            mpatches.Patch(facecolor=fill_mocap, edgecolor='black', hatch=hatch_mocap, label='Captured motion'),
            mpatches.Patch(facecolor=fill_gen, edgecolor='black', hatch=hatch_gen, label='Recommended motion'),
        ]
    elif USE_HATCH:
        handles = [
            mpatches.Patch(facecolor='white', edgecolor='black', hatch=hatch_mocap, label='Captured motion'),
            mpatches.Patch(facecolor='white', edgecolor='black', hatch=hatch_gen, label='Recommended motion'),
        ]
    else:
        handles = [
            mpatches.Patch(facecolor=colors_mocap, alpha=0.6, edgecolor='black', label='Captured motion'),
            mpatches.Patch(facecolor=colors_gen, alpha=0.6, edgecolor='black', label='Recommended motion'),
        ]
    if SHOW_MEAN_DIAMONDS:
        handles.append(plt.Line2D([0],[0], marker='D', color='w', markerfacecolor='black',
                                   markeredgecolor='black', markersize=6, label='Mean'))
    if SHOW_OUTLIERS:
        handles.append(plt.Line2D([0],[0], marker='o', color='w', markerfacecolor='none',
                                   markeredgecolor='black', markersize=6, label='Outlier (>1.5x IQR)'))
    if SHOW_INDIVIDUAL_DOTS:
        handles.append(plt.Line2D([0],[0], marker='o', color='w', markerfacecolor='gray',
                                   alpha=0.4, markersize=6, label='Individual observation'))
    ax2.legend(handles=handles, fontsize=13, loc='lower left', framealpha=0.9,
               bbox_to_anchor=(0.08, 0.0))
    
    n_scenarios = len(gen_data)
    n_obs = n_scenarios * 4  # 3 captured + 1 synthesized per scenario
    fig.text(0.9, -0.01,
             '* p<0.05  ** p<0.01  *** p<0.001  n.s. not significant',
             ha='right', va='bottom', fontsize=14, color='gray', style='italic')

    plt.suptitle(title_str, fontsize=20, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'/Users/leyangwen/Downloads/{filename}', dpi=200, bbox_inches='tight')
    plt.close()
    
    # Stats table
    print(f"\n{'='*100}")
    print(f"  {title_str}")
    print(f"  n = {n_scenarios} scenarios x 4 observations (3 captured + 1 synthesized) = {n_obs} total")
    print(f"  LMM: value ~ condition + (1|scenario)")
    print(f"{'='*100}")
    print(f"{'Metric':<15} {'Mocap Mean+-SD':<22} {'Gen Mean+-SD':<22} {'Beta':<12} {'p-value':<12} {'Cohen d':<10} {'Sig':<5}")
    print("-"*98)
    
    print(f"{'L4/L5 (N)':<15} {mocap_comp.mean():.0f}+-{mocap_comp.std(ddof=1):.0f}{'':<10} {gen_comp.mean():.0f}+-{gen_comp.std(ddof=1):.0f}{'':<10} {beta_comp:+.1f}      {p_comp:.4f}      {d_comp:+.2f}      {sig_label(p_comp)}")
    
    for i, (joint, col) in enumerate(zip(percentile_names, percentile_cols)):
        m, g = mocap_data_list[i], gen_data_list[i]
        beta, p, d = run_lmm(s1_data, s2_data, s3_data, gen_data, col)
        print(f"{joint:<15} {m.mean():.1f}+-{m.std(ddof=1):.1f}{'':<12} {g.mean():.1f}+-{g.std(ddof=1):.1f}{'':<12} {beta:+.2f}       {p:.4f}      {d:+.2f}      {sig_label(p)}")


# ============================================================
# NIOSH PLOT FUNCTION
# ============================================================
def make_niosh_boxplot(s1_data, s2_data, s3_data, gen_data, title_str, filename):
    colors_mocap = '#C0392B'
    colors_gen = '#1A5276'
    fill_mocap = '#F2D7D5'
    fill_gen = '#D4E6F1'
    hatch_mocap = ''
    hatch_gen = '////////'

    niosh_names = ['Lifting Index', 'RWL']
    niosh_cols = [COL_LI, COL_RWL]
    niosh_units = ['Index', 'Weight (kg)']

    fig, axes = plt.subplots(1, 2, figsize=(10, 6.5), gridspec_kw={'wspace': 0.4})

    flier_props = dict(marker='o', markerfacecolor='none', markeredgecolor='black',
                       markersize=4, linewidth=1.0) if SHOW_OUTLIERS else dict(marker='None')
    mean_props = dict(marker='D', markerfacecolor='black', markeredgecolor='black', markersize=6, zorder=10)

    def draw_box_and_dots(ax, vals, pos, color, hatch):
        bp = ax.boxplot([vals], positions=[pos], widths=0.3,
                         patch_artist=True, showmeans=SHOW_MEAN_DIAMONDS,
                         meanprops=mean_props,
                         medianprops=dict(color='black', linewidth=1.5),
                         whiskerprops=dict(linewidth=1.2),
                         capprops=dict(linewidth=1.2),
                         flierprops=flier_props)
        if USE_HATCH == 'both':
            fc = fill_mocap if color == colors_mocap else fill_gen
            bp['boxes'][0].set_facecolor(fc)
            bp['boxes'][0].set_edgecolor('black')
            bp['boxes'][0].set_hatch(hatch)
            bp['boxes'][0].set_alpha(0.9)
        elif USE_HATCH:
            bp['boxes'][0].set_facecolor('white')
            bp['boxes'][0].set_edgecolor('black')
            bp['boxes'][0].set_hatch(hatch)
            bp['boxes'][0].set_alpha(0.8)
        else:
            bp['boxes'][0].set_facecolor(color)
            bp['boxes'][0].set_alpha(0.6)

        if SHOW_INDIVIDUAL_DOTS:
            outlier_mask = get_outlier_mask(vals) if SHOW_OUTLIERS else np.zeros(len(vals), dtype=bool)
            for idx, v in enumerate(vals):
                if outlier_mask[idx]:
                    ax.scatter(pos, v, color=color, alpha=0.7, s=20,
                               zorder=5, edgecolors='none')
                else:
                    jit = np.random.uniform(-0.08, 0.08)
                    ax.scatter(pos + jit, v, color=color, alpha=0.4, s=20,
                               zorder=2, edgecolors='none')

    # Stats table header
    print(f"\n{'='*100}")
    print(f"  {title_str}")
    n_scenarios = len(gen_data)
    n_obs = n_scenarios * 4
    print(f"  n = {n_scenarios} scenarios x 4 observations (3 captured + 1 synthesized) = {n_obs} total")
    print(f"  LMM: value ~ condition + (1|scenario)")
    print(f"{'='*100}")
    print(f"{'Metric':<15} {'Mocap Mean+-SD':<22} {'Gen Mean+-SD':<22} {'Beta':<12} {'p-value':<12} {'Cohen d':<10} {'Sig':<5}")
    print("-"*98)

    for ax_idx, (name, col, unit) in enumerate(zip(niosh_names, niosh_cols, niosh_units)):
        ax = axes[ax_idx]
        mocap_vals, gen_vals = build_paired_data(s1_data, s2_data, s3_data, gen_data, col)

        draw_box_and_dots(ax, mocap_vals, 0.8, colors_mocap, hatch_mocap)
        draw_box_and_dots(ax, gen_vals, 1.2, colors_gen, hatch_gen)

        beta, p, d = run_lmm(s1_data, s2_data, s3_data, gen_data, col)

        # Significance bracket
        y_min = min(mocap_vals.min(), gen_vals.min())
        y_max = max(mocap_vals.max(), gen_vals.max())
        y_range = y_max - y_min if y_max > y_min else 1
        bracket_y = y_max + y_range * 0.08
        sl = sig_label(p)
        if sl != 'n.s.':
            bar_h = y_range * 0.02
            ax.plot([0.8, 0.8, 1.2, 1.2],
                    [bracket_y, bracket_y + bar_h, bracket_y + bar_h, bracket_y],
                    color='black', linewidth=1.2)
            ax.text(1.0, bracket_y + bar_h * 1.5, sl, ha='center', va='bottom', fontsize=13, fontweight='bold')
        else:
            ax.text(1.0, bracket_y, 'n.s.', ha='center', va='bottom', fontsize=11, color='gray')

        ax.set_ylim(y_min - y_range * 0.1, bracket_y + y_range * 0.25)
        ax.set_ylabel(unit, fontsize=18)
        ax.set_xticks([1.0])
        ax.set_xticklabels([name], fontsize=18)
        ax.set_title(name, fontsize=18, fontweight='bold')
        ax.tick_params(axis='y', labelsize=15)
        ax.grid(axis='y', alpha=0.3)
        ax.set_axisbelow(True)

        # Print stats row
        print(f"{name:<15} {mocap_vals.mean():.2f}+-{mocap_vals.std(ddof=1):.2f}{'':<10} {gen_vals.mean():.2f}+-{gen_vals.std(ddof=1):.2f}{'':<10} {beta:+.3f}       {p:.4f}      {d:+.2f}      {sl}")

    # Legend
    if USE_HATCH == 'both':
        handles = [
            mpatches.Patch(facecolor=fill_mocap, edgecolor='black', hatch=hatch_mocap, label='Captured motion'),
            mpatches.Patch(facecolor=fill_gen, edgecolor='black', hatch=hatch_gen, label='Recommended motion'),
        ]
    elif USE_HATCH:
        handles = [
            mpatches.Patch(facecolor='white', edgecolor='black', hatch=hatch_mocap, label='Captured motion'),
            mpatches.Patch(facecolor='white', edgecolor='black', hatch=hatch_gen, label='Recommended motion'),
        ]
    else:
        handles = [
            mpatches.Patch(facecolor=colors_mocap, alpha=0.6, edgecolor='black', label='Captured motion'),
            mpatches.Patch(facecolor=colors_gen, alpha=0.6, edgecolor='black', label='Recommended motion'),
        ]
    if SHOW_MEAN_DIAMONDS:
        handles.append(plt.Line2D([0],[0], marker='D', color='w', markerfacecolor='black',
                                   markeredgecolor='black', markersize=6, label='Mean'))
    if SHOW_OUTLIERS:
        handles.append(plt.Line2D([0],[0], marker='o', color='w', markerfacecolor='none',
                                   markeredgecolor='black', markersize=6, label='Outlier (>1.5x IQR)'))
    if SHOW_INDIVIDUAL_DOTS:
        handles.append(plt.Line2D([0],[0], marker='o', color='w', markerfacecolor='gray',
                                   alpha=0.4, markersize=6, label='Individual observation'))
    fig.legend(handles=handles, fontsize=13, loc='lower center', framealpha=0.9,
               ncol=len(handles), bbox_to_anchor=(0.5, -0.02))

    fig.text(1, -0.045,
             '* p<0.05  ** p<0.01  *** p<0.001  n.s. not significant',
             ha='right', va='bottom', fontsize=14, color='gray', style='italic')

    plt.suptitle(title_str, fontsize=20, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'/Users/leyangwen/Downloads/{filename}', dpi=200, bbox_inches='tight')
    plt.close()


# ============================================================
# RUN
# ============================================================
np.random.seed(42)

make_boxplot(s1_flat, s2_flat, s3_flat, gen_flat,
             '3DSSPP Results: Worker Motions vs. Recommended Motions Under Different Task Scenarios',
             'boxplot_flat_ground.png')

make_boxplot(s1_terrain, s2_terrain, s3_terrain, gen_terrain,
             '3DSSPP Results: Worker Motions vs. Recommended Motions Under Different Task and Site Scenarios',
             'boxplot_terrain.png')

make_niosh_boxplot(s1_flat, s2_flat, s3_flat, gen_flat,
                   'NIOSH Lifting Analysis: Worker Motions vs. Recommended Motions Under Different Task Scenarios',
                   'boxplot_niosh_flat_ground.png')

make_niosh_boxplot(s1_terrain, s2_terrain, s3_terrain, gen_terrain,
                   'NIOSH Lifting Analysis: Worker Motions vs. Recommended Motions Under Different Task and Site Scenarios',
                   'boxplot_niosh_terrain.png')

print(f"\nFlags: SHOW_INDIVIDUAL_DOTS={SHOW_INDIVIDUAL_DOTS}, SHOW_OUTLIERS={SHOW_OUTLIERS}, SHOW_MEAN_DIAMONDS={SHOW_MEAN_DIAMONDS}")
print("Done.")
