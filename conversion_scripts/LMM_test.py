import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

# ── Raw string data ──────────────────────────────────────────────────

LI_mocap = """
MMH tasks	Scenarios				
		S01	S02	S03
Container	10 lbs - small	0.620049	0.707906899	0.713277251
	20 lbs - small	1.524060	1.352320886	1.460153949
	10 lbs - large	0.709814	0.751604573	0.856274347
	20 lbs - large	1.344163	1.501221576	1.548971543
Container w/ handle	10 lbs - small	0.528455	0.781004341	0.691379977
	20 lbs - small	1.083185	1.460060701	1.212836477
	10 lbs - large	0.649355	0.781105766	0.740427298
	20 lbs - large	1.327119	1.439224801	1.429339022
Irregular objects (lumber)	3 ft	0.222217	0.207229419	0.177623603
	5 ft	0.302269	0.312205722	0.290772679
Non-rigid bags	25 lbs	1.613177	1.539615293	1.653922727
	45 lbs	3.275656	2.966995167	3.375310664
"""

L4L5_mocap = """
MMH tasks	Scenarios				
		S01	S02	S03
Container	10 lbs - small	3371.740000	3501	3325.38
	20 lbs - small	3505.390000	3847.55	3630.18
	10 lbs - large	3770.420000	3892.27	3547.84
	20 lbs - large	3731.720000	3897.25	3631.86
Container w/ handle	10 lbs - small	3296.330000	3246.11	3301.88
	20 lbs - small	3631.940000	3833.27	3630.38
	10 lbs - large	3516.400000	3554.09	3763.03
	20 lbs - large	3680.170000	3982.25	3738.98
Irregular objects (lumber)	3 ft	3197.240000	3102.56	2481.28
	5 ft	2998.190000	2942.63	3036.25
Non-rigid bags	25 lbs	4025.040000	4028.94	3577.54
	45 lbs	4460.970000	5027.83	5531
"""

overall = """
MMH tasks	Scenarios	NIOSH Lifting Index (LI)			L4/L5 Compression Force (3D SSPP)		
		Captured motions	Synthesized motions	Difference (%)	Captured motions	Synthesized motions	Difference (%)
Container	10 lbs - small	0.68	0.42	-38%	3399	3225	-5%
	20 lbs - small	1.45	0.88	-39%	3661	3621	-1%
	10 lbs - large	0.77	0.47	-39%	3737	3237	-13%
	20 lbs - large	1.46	0.82	-44%	3754	3460	-8%
Container w/ handle	10 lbs - small	0.67	0.46	-30%	3281	3127	-5%
	20 lbs - small	1.25	0.77	-38%	3699	3161	-15%
	10 lbs - large	0.72	0.56	-22%	3611	3183	-12%
	20 lbs - large	1.40	0.99	-29%	3800	3559	-6%
Irregular objects (lumber)	3 ft	0.20	0.12	-41%	2927	2719	-7%
	5 ft	0.30	0.20	-33%	2992	2803	-6%
Non-rigid bags	25 lbs	1.60	0.86	-46%	3877	3609	-7%
	45 lbs	3.21	1.69	-47%	5007	4271	-15%
"""

# ── Parse helpers ────────────────────────────────────────────────────

def parse_subject_data(raw_str):
    lines = [l for l in raw_str.strip().split('\n') if l.strip()]
    lines = lines[2:]
    rows = []
    current_task = None
    for line in lines:
        parts = line.split('\t')
        parts = [p.strip() for p in parts]
        if parts[0]:
            current_task = parts[0]
        scenario = parts[1]
        vals = [float(v.strip()) for v in parts[2:] if v.strip()]
        if len(vals) == 3:
            rows.append({'task': current_task, 'scenario': scenario,
                         'S01': vals[0], 'S02': vals[1], 'S03': vals[2]})
    return pd.DataFrame(rows)

def parse_overall(raw_str):
    lines = [l for l in raw_str.strip().split('\n') if l.strip()]
    lines = lines[2:]
    rows = []
    current_task = None
    for line in lines:
        parts = line.split('\t')
        parts = [p.strip() for p in parts]
        if parts[0]:
            current_task = parts[0]
        scenario = parts[1]
        if scenario == '' or current_task == 'Average':
            continue
        vals = [p.strip().replace('%', '') for p in parts[2:]]
        vals = [v for v in vals if v]
        if len(vals) >= 6:
            rows.append({'task': current_task, 'scenario': scenario,
                         'LI_captured_mean': float(vals[0]), 'LI_synthesized': float(vals[1]),
                         'CF_captured_mean': float(vals[3]), 'CF_synthesized': float(vals[4])})
    return pd.DataFrame(rows)

# ── Parse ────────────────────────────────────────────────────────────

df_li = parse_subject_data(LI_mocap)
df_cf = parse_subject_data(L4L5_mocap)
df_overall = parse_overall(overall)

# ── Build long-format dataframe ──────────────────────────────────────

rows = []
for i in range(len(df_li)):
    task = df_li.iloc[i]['task']
    scenario = df_li.iloc[i]['scenario']
    scenario_id = f"{task}_{scenario}"
    li_syn = df_overall.iloc[i]['LI_synthesized']
    cf_syn = df_overall.iloc[i]['CF_synthesized']
    for subj in ['S01', 'S02', 'S03']:
        rows.append({'subject': subj, 'task': task, 'scenario': scenario_id,
                      'motion_type': 'captured', 'LI': df_li.iloc[i][subj], 'CF': df_cf.iloc[i][subj]})
        rows.append({'subject': subj, 'task': task, 'scenario': scenario_id,
                      'motion_type': 'synthesized', 'LI': li_syn, 'CF': cf_syn})

df_long = pd.DataFrame(rows)
df_long['motion_binary'] = (df_long['motion_type'] == 'synthesized').astype(int)

print("="*70)
print(f"DATA: Long format ({len(df_long)} observations)")
print("="*70)
print(df_long.head(12).to_string(index=False))

# ══════════════════════════════════════════════════════════════════════
# RUN ALL TESTS FOR A GIVEN METRIC
# ══════════════════════════════════════════════════════════════════════

def run_all_tests(df_long, metric_col, metric_name):
    print("\n" + "="*70)
    print(f"ANALYSIS FOR: {metric_name}")
    print("="*70)

    results = []

    # ── 1. LMM ───────────────────────────────────────────────────────
    print(f"\n--- LMM: {metric_col} ~ motion_type + (1|subject) + (1|scenario) ---")
    model = smf.mixedlm(f"{metric_col} ~ motion_binary", df_long,
                          groups=df_long["scenario"],
                          vc_formula={"subject": "0 + C(subject)"})
    result = model.fit()
    print(result.summary())

    beta = result.fe_params['motion_binary']
    se = result.bse_fe['motion_binary']
    z = beta / se
    p = result.pvalues['motion_binary']
    ci = result.conf_int().loc['motion_binary']

    # Random effect variances
    # groups (scenario) -> from summary, vcomp[0] is subject
    re_subject = result.vcomp[0] if len(result.vcomp) > 0 else np.nan
    residual = result.scale
    # scenario group variance from the random effects
    # Extract from the summary table
    summary_str = str(result.summary())

    print(f"\n>>> LMM KEY RESULTS ({metric_name}):")
    print(f"    Fixed effect (motion_type): beta={beta:.4f}, SE={se:.4f}, z={z:.4f}, p={p:.8f}")
    print(f"    95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")
    print(f"    Random effects: subject_var={re_subject:.4f}, residual_var={residual:.4f}")

    results.append({
        'metric': metric_name, 'test': 'LMM',
        'statistic_name': 'z', 'statistic_value': round(z, 4),
        'beta_or_mean_diff': round(beta, 4), 'SE': round(se, 4),
        'p_value_two_tail': round(p, 8), 'p_value_one_tail': round(p / 2, 8),
        'CI_95_lower': round(ci[0], 4), 'CI_95_upper': round(ci[1], 4),
        'effect_size_name': 'beta', 'effect_size_value': round(beta, 4),
        'n_observations': len(df_long), 'df': np.nan,
        'normality_W': np.nan, 'normality_p': np.nan
    })

    # ── 2. Paired t-test (scenario means, n=12) ─────────────────────
    cap = df_long[df_long['motion_type']=='captured'].groupby('scenario')[metric_col].mean()
    syn = df_long[df_long['motion_type']=='synthesized'].groupby('scenario')[metric_col].mean()
    diff = cap - syn

    t_stat, p_tt = stats.ttest_rel(cap, syn)
    d_cohen = diff.mean() / diff.std(ddof=1)
    se_diff = diff.std(ddof=1) / np.sqrt(len(diff))
    ci_diff = stats.t.interval(0.95, df=len(diff)-1, loc=diff.mean(), scale=se_diff)

    print(f"\n--- Paired t-test (scenario means, n=12) ---")
    print(f"    Mean diff={diff.mean():.4f}, SD={diff.std(ddof=1):.4f}")
    print(f"    t={t_stat:.4f}, df=11, p(two-tail)={p_tt:.8f}, p(one-tail)={p_tt/2:.8f}")
    print(f"    Cohen's d={d_cohen:.4f}")
    print(f"    95% CI of diff: [{ci_diff[0]:.4f}, {ci_diff[1]:.4f}]")

    # ── 3. Shapiro-Wilk ──────────────────────────────────────────────
    sw_stat, sw_p = stats.shapiro(diff)
    normal = sw_p > 0.05
    print(f"\n--- Shapiro-Wilk normality test (on differences) ---")
    print(f"    W={sw_stat:.4f}, p={sw_p:.4f} -> {'Normal' if normal else 'NON-NORMAL (use Wilcoxon)'}")

    results.append({
        'metric': metric_name, 'test': 'Paired t-test (scenario means)',
        'statistic_name': 't', 'statistic_value': round(t_stat, 4),
        'beta_or_mean_diff': round(diff.mean(), 4), 'SE': round(se_diff, 4),
        'p_value_two_tail': round(p_tt, 8), 'p_value_one_tail': round(p_tt / 2, 8),
        'CI_95_lower': round(ci_diff[0], 4), 'CI_95_upper': round(ci_diff[1], 4),
        'effect_size_name': "Cohen's d", 'effect_size_value': round(d_cohen, 4),
        'n_observations': 12, 'df': 11,
        'normality_W': round(sw_stat, 4), 'normality_p': round(sw_p, 4)
    })

    # ── 4. Wilcoxon signed-rank ──────────────────────────────────────
    w_stat, w_p = stats.wilcoxon(cap, syn, alternative='greater')
    print(f"\n--- Wilcoxon signed-rank test (one-tail: captured > synthesized) ---")
    print(f"    W={w_stat:.1f}, p={w_p:.8f}")

    results.append({
        'metric': metric_name, 'test': 'Wilcoxon signed-rank',
        'statistic_name': 'W', 'statistic_value': round(w_stat, 4),
        'beta_or_mean_diff': round(diff.mean(), 4), 'SE': np.nan,
        'p_value_two_tail': np.nan, 'p_value_one_tail': round(w_p, 8),
        'CI_95_lower': np.nan, 'CI_95_upper': np.nan,
        'effect_size_name': "Cohen's d", 'effect_size_value': round(d_cohen, 4),
        'n_observations': 12, 'df': np.nan,
        'normality_W': np.nan, 'normality_p': np.nan
    })

    return results

# ══════════════════════════════════════════════════════════════════════
# RUN FOR BOTH METRICS
# ══════════════════════════════════════════════════════════════════════

all_results = []
all_results.extend(run_all_tests(df_long, 'LI', 'NIOSH Lifting Index (LI)'))
all_results.extend(run_all_tests(df_long, 'CF', 'L4/L5 Compression Force (N)'))

# ── Save CSV ─────────────────────────────────────────────────────────

df_results = pd.DataFrame(all_results)
col_order = ['metric', 'test', 'statistic_name', 'statistic_value',
             'beta_or_mean_diff', 'SE',
             'p_value_two_tail', 'p_value_one_tail',
             'CI_95_lower', 'CI_95_upper',
             'effect_size_name', 'effect_size_value',
             'n_observations', 'df',
             'normality_W', 'normality_p']
df_results = df_results[col_order]

csv_path = 'statistical_test_results.csv'
df_results.to_csv(csv_path, index=False)
