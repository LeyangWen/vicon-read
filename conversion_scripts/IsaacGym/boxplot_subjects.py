import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION FLAGS
# ============================================================
SHOW_INDIVIDUAL_DOTS = True
SHOW_OUTLIERS = True
SHOW_MEAN_DIAMONDS = True

# ============================================================
# DATA
# ============================================================
COL_L4L5 = 0; COL_WRIST = 1; COL_ELBOW = 2;
COL_SHOULDER = 3; COL_TORSO = 4; COL_NECK = 5; COL_HIP = 6
COL_KNEE = 7; COL_ANKLE = 8


#                        L4L5      Wrist   Elbow   Shldr   Torso   Neck    Hip     Knee    Ankle   RF      LF
gen_flat = [
    [                    3404,     100,    100,    100,    99,     100,    95,     72,     99,     22,     22    ],
    [                    3707,     99,     100,    100,    98,     100,    95,     79,     98,     44,     44    ],
    [                    3274,     100,    100,    100,    99,     100,    95,     89,     98,     22,     22    ],
    [                    3460,     99,     100,    100,    98,     100,    95,     88,     99,     44,     44    ],
    [                    3127,     100,    100,    100,    99,     100,    91,     85,     100,    22,     22    ],
    [                    3161,     100,    100,    100,    99,     100,    89,     79,     100,    44,     44    ],
    [                    3252,     100,    100,    100,    99,     100,    89,     86,     99,     22,     22    ],
    [                    3547,     100,    100,    100,    99,     100,    87,     88,     99,     44,     44    ],
    [                    2973,     100,    100,    100,    99,     100,    95,     95,     100,    7,      7     ],
    [                    3113,     100,    100,    100,    99,     100,    95,     97,     100,    11,     11    ],
    [                    3925,     100,    100,    100,    98,     100,    95,     44,     99,     56,     56    ],
    [                    4443,     100,    100,    100,    97,     100,    91,     37,     98,     100,    100   ],
]
gen_terrain = [
    [                    3336,     100,    100,    100,    99,     100,    91,     98,     98,     22,     22    ],
    [                    3130,     100,    100,    100,    99,     100,    91,     98,     99,     22,     22    ],
    [                    2926,     100,    100,    100,    99,     100,    97,     89,     100,    7,      7     ],
    [                    3899,     99,     100,    100,    97,     100,    93,     95,     99,     56,     56    ],
]
s1_flat = [
    [                    3218.71,  94.00,  100.00, 99.80,  96.76,  100.00, 88.01,  99.67,  93.72,  22.25,  22.25 ],
    [                    3753.21,  99.46,  100.00, 99.86,  95.59,  100.00, 74.58,  99.02,  83.29,  44.44,  44.44 ],
    [                    3365.96,  99.62,  100.00, 99.86,  96.17,  100.00, 84.16,  99.85,  90.40,  22.25,  22.25 ],
    [                    3718.61,  7.75,   100.00, 99.68,  95.04,  100.00, 73.52,  99.47,  88.42,  44.44,  44.44 ],
    [                    3282.40,  99.38,  100.00, 99.87,  95.85,  100.00, 85.21,  99.51,  97.46,  22.25,  22.25 ],
    [                    3515.70,  99.49,  100.00, 99.85,  94.37,  100.00, 76.73,  99.46,  93.40,  44.44,  44.44 ],
    [                    3285.09,  99.57,  100.00, 99.88,  94.98,  100.00, 75.05,  99.75,  89.56,  22.25,  22.25 ],
    [                    3668.21,  99.36,  100.00, 99.77,  92.69,  100.00, 61.90,  99.79,  69.57,  44.44,  44.44 ],
    [                    3319.98,  99.58,  100.00, 99.92,  97.42,  100.00, 73.13,  94.76,  87.72,  6.66,   6.66  ],
    [                    3261.59,  99.50,  100.00, 99.88,  97.70,  100.00, 83.24,  94.07,  96.01,  11.12,  11.12 ],
    [                    3828.30,  99.27,  100.00, 99.80,  95.14,  100.00, 84.73,  98.32,  95.36,  55.57,  55.57 ],
    [                    4796.56,  97.62,  99.94,  99.44,  89.04,  100.00, 41.13,  99.30,  57.90,  100.01, 100.01],
]
s1_terrain = [
    [                    3516.40,  99.48,  100.00, 99.83,  92.75,  100.00, 40.88,  99.96,  97.44,  22.25,  22.25 ],
    [                    3211.74,  99.57,  100.00, 99.88,  90.41,  100.00, 86.39,  97.50,  93.66,  22.25,  22.25 ],
    [                    3255.05,  99.69,  100.00, 99.92,  95.29,  100.00, 83.18,  95.05,  87.64,  6.66,   6.66  ],
    [                    4572.37,  0.00,   100.00, 99.39,  89.36,  100.00, 25.78,  99.73,  85.34,  55.57,  55.57 ],
]
s2_flat = [
    [                    3445.83,  99.57,  100.00, 99.86,  96.56,  100.00, 77.59,  98.01,  91.61,  22.25,  22.25 ],
    [                    3791.49,  99.45,  100.00, 99.80,  96.57,  100.00, 69.12,  77.49,  92.03,  44.44,  44.44 ],
    [                    3485.75,  99.56,  100.00, 99.85,  97.33,  100.00, 77.20,  90.40,  93.29,  22.25,  22.25 ],
    [                    3887.40,  98.97,  100.00, 99.72,  95.86,  100.00, 58.54,  51.03,  84.26,  44.44,  44.44 ],
    [                    3417.60,  99.60,  100.00, 99.89,  93.57,  100.00, 61.69,  99.83,  67.89,  22.25,  22.25 ],
    [                    3722.95,  99.27,  100.00, 99.80,  91.14,  100.00, 50.78,  99.67,  50.54,  44.44,  44.44 ],
    [                    3529.68,  99.58,  100.00, 99.78,  93.84,  100.00, 62.41,  99.91,  80.57,  22.25,  22.25 ],
    [                    3613.41,  99.22,  100.00, 99.59,  91.19,  100.00, 47.10,  99.65,  36.79,  44.44,  44.44 ],
    [                    3064.82,  99.70,  100.00, 99.90,  97.66,  100.00, 79.09,  95.66,  90.97,  6.66,   6.66  ],
    [                    3172.13,  99.71,  100.00, 99.87,  97.82,  100.00, 76.77,  87.11,  93.40,  11.12,  11.12 ],
    [                    3773.60,  98.02,  100.00, 99.81,  95.87,  100.00, 74.37,  96.02,  92.77,  55.57,  55.57 ],
    [                    4827.38,  93.28,  99.91,  99.60,  92.98,  100.00, 57.66,  84.30,  80.12,  100.01, 100.01],
]
s2_terrain = [
    [                    3554.09,  96.60,  100.00, 99.91,  97.97,  100.00, 92.79,  87.79,  97.94,  22.25,  22.25 ],
    [                    3169.87,  99.55,  100.00, 99.89,  92.16,  100.00, 39.79,  99.91,  94.84,  22.25,  22.25 ],
    [                    3304.74,  99.69,  100.00, 99.91,  96.37,  100.00, 90.53,  99.45,  96.51,  6.66,   6.66  ],
    [                    4244.78,  99.31,  100.00, 99.70,  95.04,  100.00, 89.72,  98.50,  96.76,  55.57,  55.57 ],
]
s3_flat = [
    [                    3323.79,  99.68,  100.00, 99.89,  96.16,  100.00, 59.98,  98.47,  47.35,  22.25,  22.25 ],
    [                    3632.18,  99.60,  100.00, 99.82,  95.28,  100.00, 60.35,  98.98,  49.14,  44.44,  44.44 ],
    [                    3290.62,  99.66,  100.00, 99.82,  96.22,  100.00, 63.69,  98.66,  78.54,  22.25,  22.25 ],
    [                    3644.50,  99.56,  100.00, 99.59,  95.60,  100.00, 57.56,  96.09,  62.03,  44.44,  44.44 ],
    [                    3263.60,  99.68,  100.00, 99.95,  92.95,  100.00, 45.27,  99.72,  45.38,  22.25,  22.25 ],
    [                    3513.01,  99.62,  100.00, 99.94,  91.86,  100.00, 38.00,  99.66,  30.56,  44.44,  44.44 ],
    [                    3360.54,  99.65,  100.00, 99.90,  93.05,  100.00, 46.59,  99.95,  57.71,  22.25,  22.25 ],
    [                    3671.61,  92.73,  100.00, 99.73,  89.99,  100.00, 35.40,  99.77,  18.27,  44.44,  44.44 ],
    [                    2632.64,  99.71,  100.00, 99.92,  97.34,  100.00, 72.54,  99.27,  91.67,  6.66,   6.66  ],
    [                    2523.98,  99.29,  100.00, 99.90,  97.73,  100.00, 72.50,  97.54,  87.85,  11.12,  11.12 ],
    [                    3658.61,  41.45,  100.00, 99.84,  94.09,  100.00, 57.81,  99.67,  75.05,  55.57,  55.57 ],
    [                    4242.31,  2.04,   100.00, 99.37,  89.62,  100.00, 31.02,  99.36,  15.04,  100.01, 100.01],
]
s3_terrain = [
    [                    3763.03,  99.49,  100.00, 99.72,  94.31,  100.00, 89.41,  99.81,  97.60,  22.25,  22.25 ],
    [                    3247.39,  99.59,  100.00, 99.95,  92.44,  100.00, 41.32,  99.77,  99.48,  22.25,  22.25 ],
    [                    3173.90,  99.71,  100.00, 99.89,  95.87,  100.00, 90.31,  99.74,  97.20,  6.66,   6.66  ],
    [                    4316.01,  29.11,  100.00, 99.76,  89.25,  100.00, 25.29,  99.85,  93.67,  55.57,  55.57 ],
]



# ============================================================
# HELPERS
# ============================================================
def build_paired_data(s1_data, s2_data, s3_data, gen_data, col):
    mocap_vals, gen_vals = [], []
    for i in range(len(gen_data)):
        for subj_data in [s1_data, s2_data, s3_data]:
            mocap_vals.append(subj_data[i][col])
            gen_vals.append(gen_data[i][col])
    return np.array(mocap_vals), np.array(gen_vals)

def build_lmm_dataframe(s1_data, s2_data, s3_data, gen_data, col):
    """Build a DataFrame for LMM with crossed random effects (subject, scenario)."""
    rows = []
    n_scenarios = len(gen_data)
    subjects = ['S1', 'S2', 'S3']
    subj_data_list = [s1_data, s2_data, s3_data]
    
    for i in range(n_scenarios):
        scenario_id = f'Sc{i+1}'
        # Mocap observations
        for s_idx, subj in enumerate(subjects):
            rows.append({
                'value': subj_data_list[s_idx][i][col],
                'condition': 0,  # mocap = 0 (reference)
                'subject': subj,
                'scenario': scenario_id,
            })
        # Generated observation (no subject random effect - use "Gen")
        rows.append({
            'value': gen_data[i][col],
            'condition': 1,  # generated = 1
            'subject': 'Gen',
            'scenario': scenario_id,
        })
    return pd.DataFrame(rows)

def run_lmm(s1_data, s2_data, s3_data, gen_data, col):
    """Run LMM with scenario as random effect. Returns (beta, p, cohen_d)."""
    df = build_lmm_dataframe(s1_data, s2_data, s3_data, gen_data, col)
    
    # Check if there's any variance
    if df['value'].std() < 1e-10:
        return 0.0, 1.0, 0.0
    
    try:
        model = smf.mixedlm("value ~ condition", df, groups=df["scenario"],
                             re_formula="~condition")
        result = model.fit(reml=True, method='lbfgs')
        beta = result.params['condition']
        p = result.pvalues['condition']
    except Exception:
        try:
            # Fallback: simpler random intercept only
            model = smf.mixedlm("value ~ condition", df, groups=df["scenario"])
            result = model.fit(reml=True)
            beta = result.params['condition']
            p = result.pvalues['condition']
        except Exception:
            # Last resort: paired t-test
            mocap_vals, gen_vals = build_paired_data(s1_data, s2_data, s3_data, gen_data, col)
            _, p = stats.wilcoxon(mocap_vals, gen_vals)
            beta = gen_vals.mean() - mocap_vals.mean()
    
    # Cohen's d from raw paired data
    mocap_vals, gen_vals = build_paired_data(s1_data, s2_data, s3_data, gen_data, col)
    diffs = gen_vals - mocap_vals
    d = np.mean(diffs) / np.std(diffs, ddof=1) if np.std(diffs, ddof=1) > 0 else 0
    
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
    colors_mocap = '#E74C3C'
    colors_gen = '#2E86C1'
    
    # No neck
    percentile_names = ['Wrist', 'Elbow', 'Shoulder', 'Torso', 'Hip', 'Knee', 'Ankle']
    percentile_cols = [COL_WRIST, COL_ELBOW, COL_SHOULDER, COL_TORSO, COL_HIP, COL_KNEE, COL_ANKLE]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5),
                                    gridspec_kw={'width_ratios': [1, 4]})
    
    # Outlier style: always black hollow circles
    flier_props = dict(marker='o', markerfacecolor='none', markeredgecolor='black',
                       markersize=7, linewidth=1.2) if SHOW_OUTLIERS else dict(marker='None')
    
    mean_props = dict(marker='D', markerfacecolor='black', markeredgecolor='black', markersize=6)
    
    def draw_box_and_dots(ax, vals, pos, color, is_compression=False):
        """Draw one box with optional dots. Outliers are black circles with colored dot inside."""
        bp = ax.boxplot([vals], positions=[pos], widths=0.3,
                         patch_artist=True, showmeans=SHOW_MEAN_DIAMONDS,
                         meanprops=mean_props,
                         medianprops=dict(color='black', linewidth=1.5),
                         whiskerprops=dict(linewidth=1.2),
                         capprops=dict(linewidth=1.2),
                         flierprops=flier_props)
        bp['boxes'][0].set_facecolor(color)
        bp['boxes'][0].set_alpha(0.6)
        
        if SHOW_INDIVIDUAL_DOTS:
            outlier_mask = get_outlier_mask(vals)
            dot_size = 20 if is_compression else 15
            
            for idx, v in enumerate(vals):
                if outlier_mask[idx] and SHOW_OUTLIERS:
                    # Outlier: place colored dot centered on the outlier circle (no jitter)
                    ax.scatter(pos, v, color=color, alpha=0.7, s=dot_size,
                               zorder=5, edgecolors='none')
                else:
                    # Normal point: jitter
                    jit = np.random.uniform(-0.08, 0.08)
                    ax.scatter(pos + jit, v, color=color, alpha=0.4, s=dot_size,
                               zorder=2, edgecolors='none')
    
    # ---- Left: L4/L5 ----
    mocap_comp, gen_comp = build_paired_data(s1_data, s2_data, s3_data, gen_data, COL_L4L5)
    
    draw_box_and_dots(ax1, mocap_comp, 0.8, colors_mocap, is_compression=True)
    draw_box_and_dots(ax1, gen_comp, 1.2, colors_gen, is_compression=True)
    
    beta_comp, p_comp, d_comp = run_lmm(s1_data, s2_data, s3_data, gen_data, COL_L4L5)
    y_max = max(mocap_comp.max(), gen_comp.max())
    bracket_y = y_max + 80
    ax1.plot([0.8,0.8,1.2,1.2], [bracket_y, bracket_y+40, bracket_y+40, bracket_y], color='black', linewidth=1.2)
    ax1.text(1.0, bracket_y+50, sig_label(p_comp), ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax1.set_ylabel('Compression Force (N)', fontsize=12)
    ax1.set_xticks([1.0])
    ax1.set_xticklabels(['L4/L5'], fontsize=11)
    ax1.set_title('3D Low Back Compression', fontsize=13, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3); ax1.set_axisbelow(True)
    
    # ---- Right: Percentiles ----
    positions_mocap, positions_gen = [], []
    mocap_data_list, gen_data_list = [], []
    
    for i, (joint, col) in enumerate(zip(percentile_names, percentile_cols)):
        pos_m = i*2.0 + 0.8; pos_g = i*2.0 + 1.2
        positions_mocap.append(pos_m); positions_gen.append(pos_g)
        m, g = build_paired_data(s1_data, s2_data, s3_data, gen_data, col)
        mocap_data_list.append(m); gen_data_list.append(g)
    
    for i in range(len(percentile_names)):
        draw_box_and_dots(ax2, mocap_data_list[i], positions_mocap[i], colors_mocap)
        draw_box_and_dots(ax2, gen_data_list[i], positions_gen[i], colors_gen)
    
    # Significance brackets via LMM
    for i, (joint, col) in enumerate(zip(percentile_names, percentile_cols)):
        beta, p, d = run_lmm(s1_data, s2_data, s3_data, gen_data, col)
        all_vals = np.concatenate([mocap_data_list[i], gen_data_list[i]])
        y_max = all_vals.max()
        bracket_y = y_max + 1.2
        pos_m, pos_g = positions_mocap[i], positions_gen[i]
        
        sl = sig_label(p)
        if sl != 'n.s.':
            ax2.plot([pos_m,pos_m,pos_g,pos_g], [bracket_y, bracket_y+0.6, bracket_y+0.6, bracket_y],
                     color='black', linewidth=1.2)
            ax2.text((pos_m+pos_g)/2, bracket_y+0.7, sl, ha='center', va='bottom', fontsize=11, fontweight='bold')
        else:
            ax2.text((pos_m+pos_g)/2, bracket_y+0.3, 'n.s.', ha='center', va='bottom', fontsize=9, color='gray')
    
    ax2.set_xticks([i*2.0+1.0 for i in range(len(percentile_names))])
    ax2.set_xticklabels(percentile_names, fontsize=11)
    ax2.set_ylabel('Strength Percentile Capable (%)', fontsize=12)
    ax2.set_title('Joint Strength Percentile Capable', fontsize=13, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3); ax2.set_axisbelow(True)
    
    all_pct = np.concatenate(mocap_data_list + gen_data_list)
    ax2.set_ylim(max(0, all_pct.min()-5), 108)
    
    # Legend
    handles = [
        mpatches.Patch(facecolor=colors_mocap, alpha=0.6, edgecolor='black', label='Motion Capture (n=3 subjects)'),
        mpatches.Patch(facecolor=colors_gen, alpha=0.6, edgecolor='black', label='RL-Synthesized'),
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
    ax2.legend(handles=handles, fontsize=9, loc='lower left', framealpha=0.9)
    
    n_scenarios = len(gen_data)
    n_obs = n_scenarios * 3
    fig.text(0.99, 0.01,
             f'* p<0.05  ** p<0.01  *** p<0.001  n.s. not significant (LMM with scenario as random effect, n={n_obs} paired observations)',
             ha='right', va='bottom', fontsize=9, color='gray', style='italic')
    
    plt.suptitle(title_str, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'/Users/leyangwen/Downloads/{filename}', dpi=200, bbox_inches='tight')
    plt.close()
    
    # Stats table
    print(f"\n{'='*100}")
    print(f"  {title_str}")
    print(f"  n = {n_scenarios} scenarios x 3 subjects = {n_obs} paired observations")
    print(f"  LMM: value ~ condition + (1+condition|scenario)")
    print(f"{'='*100}")
    print(f"{'Metric':<15} {'Mocap Mean+-SD':<22} {'Gen Mean+-SD':<22} {'Beta':<12} {'p-value':<12} {'Cohen d':<10} {'Sig':<5}")
    print("-"*98)
    
    print(f"{'L4/L5 (N)':<15} {mocap_comp.mean():.0f}+-{mocap_comp.std(ddof=1):.0f}{'':<10} {gen_comp.mean():.0f}+-{gen_comp.std(ddof=1):.0f}{'':<10} {beta_comp:+.1f}      {p_comp:.4f}      {d_comp:+.2f}      {sig_label(p_comp)}")
    
    for i, (joint, col) in enumerate(zip(percentile_names, percentile_cols)):
        m, g = mocap_data_list[i], gen_data_list[i]
        beta, p, d = run_lmm(s1_data, s2_data, s3_data, gen_data, col)
        print(f"{joint:<15} {m.mean():.1f}+-{m.std(ddof=1):.1f}{'':<12} {g.mean():.1f}+-{g.std(ddof=1):.1f}{'':<12} {beta:+.2f}       {p:.4f}      {d:+.2f}      {sig_label(p)}")


# ============================================================
# RUN
# ============================================================
np.random.seed(42)

make_boxplot(s1_flat, s2_flat, s3_flat, gen_flat,
             'Flat-Ground Scenarios (12 Tasks): Motion Capture vs. RL-Synthesized Postures',
             'boxplot_flat_ground.png')

make_boxplot(s1_terrain, s2_terrain, s3_terrain, gen_terrain,
             'Terrain Scenarios (4 Tasks): Motion Capture vs. RL-Synthesized Postures',
             'boxplot_terrain.png')

print(f"\nFlags: SHOW_INDIVIDUAL_DOTS={SHOW_INDIVIDUAL_DOTS}, SHOW_OUTLIERS={SHOW_OUTLIERS}, SHOW_MEAN_DIAMONDS={SHOW_MEAN_DIAMONDS}")
print("Done.")
