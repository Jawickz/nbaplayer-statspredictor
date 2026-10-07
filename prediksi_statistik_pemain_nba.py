import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import difflib
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from xgboost import XGBRegressor
from sklearn.svm import LinearSVR
from sklearn.linear_model import Ridge
from math import pi

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 5)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Folder penyimpanan visualisasi
OUTPUT_DIR = 'visualisasi_output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def simpan_dan_tampilkan(nama_file):
    """Simpan figure matplotlib yang sedang aktif ke OUTPUT_DIR sebagai PNG,
    lalu tetap menampilkannya (plt.show())."""
    nama_file = re.sub(r'[^A-Za-z0-9_.-]+', '_', nama_file).strip('_').lower()
    if not nama_file.endswith('.png'):
        nama_file += '.png'
    path_lengkap = os.path.join(OUTPUT_DIR, nama_file)
    plt.savefig(path_lengkap, dpi=150, bbox_inches='tight')
    print(f'[Tersimpan] {path_lengkap}')
    plt.show()

# Statistik yang diprediksi (multi-output)
TARGET_STATS = ['pts_per_game', 'ast_per_game', 'trb_per_game', 'x3p_per_game', 'blk_per_game']
TARGET_LABELS = {'pts_per_game':'PTS', 'ast_per_game':'AST', 'trb_per_game':'TRB',
                  'x3p_per_game':'3P', 'blk_per_game':'BLK'}

# 6. Load Dataset
filename = 'player_per_game.csv'

df_raw = pd.read_csv(filename)

print('Ukuran dataset awal:', df_raw.shape)
df_raw.head()

# 7. Exploratory Data Analysis (EDA)

df_raw.info()

print('Jumlah baris duplikat:', df_raw.duplicated().shape[0] - df_raw.drop_duplicates().shape[0])
print()
print('Persentase missing value per kolom (%):')
print((df_raw.isna().mean() * 100).round(2).sort_values(ascending=False).head(15))

print('Rentang musim:', df_raw['season'].min(), '-', df_raw['season'].max())
print('Nama Liga:', df_raw['lg'].unique())
print('Posisi pemain:', df_raw['pos'].unique())
print('Jumlah pemain:', df_raw['player_id'].nunique())

df_raw.describe()[['age','g','mp_per_game','pts_per_game','trb_per_game','ast_per_game']]

fig, axes = plt.subplots(1, 5, figsize=(22,4))
for ax, col in zip(axes, TARGET_STATS):
    sns.histplot(df_raw[col].dropna(), bins=35, ax=ax, color='#1d428a')
    ax.set_title(f'Distribusi {TARGET_LABELS[col]}')
plt.tight_layout()
simpan_dan_tampilkan('eda_distribusi_target_stats')

num_cols = ['age','g','mp_per_game','fg_percent','x3p_percent','ft_percent'] + TARGET_STATS + ['stl_per_game','tov_per_game']
corr = df_raw[num_cols].corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0)
plt.title('Korelasi Antar Fitur Numerik')
simpan_dan_tampilkan('eda_korelasi_fitur_numerik')

season_trend = df_raw[df_raw['lg']=='NBA'].groupby('season')[['pts_per_game','x3p_per_game','ast_per_game','trb_per_game']].mean()

plt.figure(figsize=(12,5))
for col, label, color in [('pts_per_game','PTS','#c8102e'), ('x3p_per_game','3P Made','#1d428a'),
                            ('ast_per_game','AST','#f9a01b'), ('trb_per_game','TRB','#552583')]:
    plt.plot(season_trend.index, season_trend[col], label=label, color=color, linewidth=1.8)
plt.title('Tren Rata-Rata Statistik Liga NBA per Musim (1947-2026)')
plt.xlabel('Musim')
plt.ylabel('Rata-rata per Game')
plt.legend()
plt.tight_layout()
simpan_dan_tampilkan('eda_tren_statistik_per_musim')

plt.figure(figsize=(9,5))
order = ['PG','SG','SF','PF','C']
sns.boxplot(x='pos', y='pts_per_game', data=df_raw[df_raw['pos'].isin(order)], order=order, palette='Blues')
plt.title('Distribusi PTS per Game Berdasarkan Posisi')
plt.xlabel('Posisi')
plt.ylabel('PTS per Game')
plt.tight_layout()
simpan_dan_tampilkan('eda_boxplot_pts_per_posisi')

#8. Data Preprocessing

#Filtering Data (Relevansi Era & Kualitas Data)
df = df_raw[(df_raw['season'] >= 2000) & (df_raw['lg'] == 'NBA')].copy()
print('Ukuran setelah filtering era:', df.shape)


print('Missing value SETELAH filtering musim >= 2000:')
missing_after_filter = df.isna().sum()
missing_after_filter = missing_after_filter[missing_after_filter > 0].sort_values(ascending=False)
print(missing_after_filter if len(missing_after_filter) > 0 else 'Tidak ada missing value tersisa di luar kolom persentase.')

#Duplicate Removal
before = df.shape[0]
df = df.drop_duplicates()
print(f'Baris duplikat dihapus: {before - df.shape[0]}')

#Penanganan Pemain yang Di-Trade Tengah Musim
dup_mask = df.duplicated(subset=['player_id', 'season'], keep=False)
is_combined_team = df['team'].isin(['2TM', '3TM', '4TM'])

before_trade = df.shape[0]
df = df[~(dup_mask & ~is_combined_team)].copy()

print(f'Baris pecahan per-tim (trade) yang dibuang: {before_trade - df.shape[0]}')
print('Sisa duplikat player-season setelah pembersihan:', df.duplicated(subset=["player_id","season"]).sum())

contoh_trade = df[df['team'].isin(['2TM','3TM','4TM'])].head(1)['player_id'].values
if len(contoh_trade) > 0:
    print(df[df['player_id']==contoh_trade[0]][['season','player','team','g','pts_per_game']].to_string(index=False))

#Missing Value Handling
pct_cols = ['fg_percent','x3p_percent','x2p_percent','ft_percent','e_fg_percent']
df[pct_cols] = df[pct_cols].fillna(0)

df['pos'] = df['pos'].fillna(df['pos'].mode()[0])
df['age'] = df['age'].fillna(df['age'].median())

print('Sisa missing value:', df.isna().sum().sum())

#Feature Engineering
df = df.sort_values(['player_id', 'season'])

df['next_season'] = df.groupby('player_id')['season'].shift(-1)
df['season_gap'] = df['next_season'] - df['season']

target_cols_next = []
for col in TARGET_STATS:
    tcol = f'target_{col}_next'
    df[tcol] = df.groupby('player_id')[col].shift(-1)
    target_cols_next.append(tcol)

df_model = df[df['season_gap'] == 1].copy()
df_model = df_model[df_model['g'] >= 15].copy()

print('Ukuran dataset final untuk modeling:', df_model.shape)
df_model[['player','season','pts_per_game','next_season'] + target_cols_next].head()

#Feature Selection
feature_cols = ['age','g','gs','mp_per_game','fg_per_game','fga_per_game','fg_percent',
                'x3p_per_game','x3pa_per_game','x3p_percent','x2p_per_game','x2pa_per_game','x2p_percent',
                'e_fg_percent','ft_per_game','fta_per_game','ft_percent','orb_per_game','drb_per_game',
                'trb_per_game','ast_per_game','stl_per_game','blk_per_game','tov_per_game','pf_per_game',
                'pts_per_game','pos']

X = df_model[feature_cols].copy()
Y = df_model[target_cols_next].copy()
Y.columns = [TARGET_LABELS[c] for c in TARGET_STATS]

print('Jumlah fitur sebelum encoding:', X.shape[1])
print('Target output:', list(Y.columns))

#Encoding Variabel Kategorikal (posisi: C, PG, SG, SF, PF)
X = pd.get_dummies(X, columns=['pos'], prefix='pos')
FEATURE_COLUMNS = X.columns.tolist() 
print('Jumlah fitur setelah encoding:', X.shape[1])
X.head()

#Train-Test Split
train_mask = df_model['season'] <= 2019
X_train, X_test = X[train_mask], X[~train_mask]
Y_train, Y_test = Y[train_mask], Y[~train_mask]

print('Data latih :', X_train.shape, Y_train.shape)
print('Data uji   :', X_test.shape, Y_test.shape)

#Standardisasi
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

print('Standardisasi selesai. Rata-rata fitur latih ~0:', X_train_scaled.mean().mean().round(4))

#Data Cleaning Funnel
step1 = df_raw.shape[0]
step2 = df_raw[(df_raw['season'] >= 2000) & (df_raw['lg'] == 'NBA')].drop_duplicates().shape[0]

tmp = df_raw[(df_raw['season'] >= 2000) & (df_raw['lg'] == 'NBA')].drop_duplicates().copy()
dup_mask_tmp = tmp.duplicated(subset=['player_id', 'season'], keep=False)
is_combined_tmp = tmp['team'].isin(['2TM', '3TM', '4TM'])
tmp = tmp[~(dup_mask_tmp & ~is_combined_tmp)]
step3 = tmp.shape[0]

step4 = df_model.shape[0]

funnel_labels = ['1. Data Mentah\n(Seluruh Era)', '2. Filter Musim >=2000\n& Liga NBA',
                  '3. Dedup + Fix\nPemain Trade', '4. Filter Pasangan Musim\nValid & g>=15']
funnel_values = [step1, step2, step3, step4]

plt.figure(figsize=(9,5))
bars = plt.bar(funnel_labels, funnel_values, color=['#9e9e9e','#1d428a','#552583','#c8102e'])
for bar, val in zip(bars, funnel_values):
    plt.text(bar.get_x()+bar.get_width()/2, val + max(funnel_values)*0.01, f'{val:,}', ha='center', fontsize=10)
plt.title('Jumlah Baris Data pada Setiap Tahap Preprocessing')
plt.ylabel('Jumlah Baris')
plt.tight_layout()
simpan_dan_tampilkan('preprocessing_data_cleaning_funnel')

#9. Fungsi Bantu: Evaluasi & Diagnostik Model Multi-Output
def evaluate_multioutput(name, Y_true, Y_pred_arr, target_names=Y.columns):
    """Evaluasi tiap target output secara terpisah, lalu dirata-rata."""
    rows = []
    for i, t in enumerate(target_names):
        yt, yp = Y_true.iloc[:, i].values, Y_pred_arr[:, i]
        rmse = np.sqrt(mean_squared_error(yt, yp))
        mae = mean_absolute_error(yt, yp)
        mask = yt != 0
        mape = mean_absolute_percentage_error(yt[mask], yp[mask]) * 100
        r2 = r2_score(yt, yp)
        rows.append({'model': name, 'target': t, 'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'R2': r2})
    detail_df = pd.DataFrame(rows)
    avg = detail_df[['RMSE','MAE','MAPE','R2']].mean()
    print(f'--- {name} (rata-rata seluruh target) ---')
    print(f"RMSE : {avg['RMSE']:.3f}")
    print(f"MAE  : {avg['MAE']:.3f}")
    print(f"MAPE : {avg['MAPE']:.2f}%")
    print(f"R^2  : {avg['R2']:.4f}")
    return detail_df, {'model': name, 'RMSE': avg['RMSE'], 'MAE': avg['MAE'], 'MAPE': avg['MAPE'], 'R2': avg['R2']}

def plot_diagnostics(name, y_true, y_pred, target_label='PTS'):
    """Plot diagnostik untuk satu target utama (default PTS) agar tetap ringkas."""
    residuals = y_true - y_pred
    fig, axes = plt.subplots(1, 3, figsize=(17,4.5))

    axes[0].scatter(y_true, y_pred, alpha=0.35, color='#1d428a', s=15)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    axes[0].plot(lims, lims, 'r--', linewidth=1.5)
    axes[0].set_xlabel(f'Aktual ({target_label} musim depan)')
    axes[0].set_ylabel('Prediksi')
    axes[0].set_title(f'{name} — Prediction Plot ({target_label})')

    axes[1].scatter(y_pred, residuals, alpha=0.35, color='#c8102e', s=15)
    axes[1].axhline(0, color='black', linestyle='--', linewidth=1)
    axes[1].set_xlabel('Prediksi')
    axes[1].set_ylabel('Residual (Aktual - Prediksi)')
    axes[1].set_title(f'{name} — Residual Plot ({target_label})')

    sns.histplot(residuals, bins=40, kde=True, ax=axes[2], color='#1d428a')
    axes[2].set_title(f'{name} — Error Distribution ({target_label})')
    axes[2].set_xlabel('Residual')

    plt.tight_layout()
    simpan_dan_tampilkan(f'diagnostic_{name}_{target_label}')

results_summary = []  
results_detail = []  

#10. Algoritma 1: Linear Regressio
model_lr = MultiOutputRegressor(LinearRegression())
model_lr.fit(X_train_scaled, Y_train)
print('Parameter estimator dasar:', model_lr.estimator.get_params())

Y_pred_lr = model_lr.predict(X_test_scaled)
detail_lr, avg_lr = evaluate_multioutput('Linear Regression', Y_test, Y_pred_lr)
results_summary.append(avg_lr); results_detail.append(detail_lr)
detail_lr.round(3)

for stat in Y.columns:
    print(f"\n" + "="*60)
    print(f"📊 DIAGNOSTIC PLOT (LINEAR REGRESSION) UNTUK: {stat.upper()}")
    print("="*60)

    # Memanggil fungsi plot
    plot_diagnostics(
        'Linear Regression',
        Y_test[stat].reset_index(drop=True),
        pd.Series(Y_pred_lr[:, list(Y.columns).index(stat)]),
        target_label=stat
    )

#11. Algoritma 2: Random Forest Regressor

rf_params = dict(n_estimators=200, max_depth=7, min_samples_leaf=3, random_state=RANDOM_STATE, n_jobs=-1)
model_rf = MultiOutputRegressor(RandomForestRegressor(**rf_params))
model_rf.fit(X_train_scaled, Y_train)
print('Parameter estimator dasar:', model_rf.estimator.get_params())

Y_pred_rf = model_rf.predict(X_test_scaled)
detail_rf, avg_rf = evaluate_multioutput('Random Forest', Y_test, Y_pred_rf)
results_summary.append(avg_rf); results_detail.append(detail_rf)
detail_rf.round(3)

for stat in Y.columns:
    print(f"\n" + "="*50)
    print(f"📊 MENAMPILKAN DIAGNOSTIC PLOT UNTUK: {stat}")
    print("="*50)

    # Memanggil fungsi plot
    plot_diagnostics(
        'Random Forest',
        Y_test[stat].reset_index(drop=True),
        pd.Series(Y_pred_rf[:, list(Y.columns).index(stat)]),
        target_label=stat
    )

# Feature importance untuk target PTS (estimator ke-0)
pts_idx = list(Y.columns).index('PTS')
importances = pd.Series(model_rf.estimators_[pts_idx].feature_importances_, index=X_train.columns).sort_values(ascending=False).head(10)
plt.figure(figsize=(8,5))
sns.barplot(x=importances.values, y=importances.index, color='#1d428a')
plt.title('Top 10 Feature Importance — Random Forest (target: PTS)')
plt.xlabel('Importance')
plt.tight_layout()
simpan_dan_tampilkan('rf_feature_importance_pts')

#Visualisasi Fitur yang Paling Berpengaruh

numeric_features = [c for c in feature_cols if c != 'pos']
corr_matrix = pd.DataFrame(index=numeric_features, columns=Y.columns, dtype=float)

for feat in numeric_features:
    for target in Y.columns:
        corr_matrix.loc[feat, target] = df_model[feat].corr(df_model[f'target_{TARGET_STATS[list(Y.columns).index(target)]}_next'])

plt.figure(figsize=(7, 10))
sns.heatmap(corr_matrix.astype(float), annot=True, fmt='.2f', cmap='RdBu_r', center=0, cbar_kws={'label':'Korelasi'})
plt.title('Korelasi Setiap Fitur (Musim Berjalan) terhadap\nSetiap Target (Musim Depan)')
plt.xlabel('Target Musim Depan')
plt.ylabel('Fitur Musim Berjalan')
plt.tight_layout()
simpan_dan_tampilkan('korelasi_fitur_vs_target')

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for i, target in enumerate(Y.columns):
    imp = pd.Series(model_rf.estimators_[i].feature_importances_, index=X_train.columns).sort_values(ascending=False).head(8)
    sns.barplot(x=imp.values, y=imp.index, color='#1d428a', ax=axes[i])
    axes[i].set_title(f'Top 8 Feature Importance — Target: {target}')
    axes[i].set_xlabel('Importance')

axes[-1].axis('off')
plt.tight_layout()
simpan_dan_tampilkan('rf_feature_importance_semua_target')

#12. Algoritma 3: XGBoost Regressor

xgb_params = dict(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8,
                   colsample_bytree=0.8, min_child_weight=5, gamma=1,
                   reg_alpha=1, reg_lambda=3, random_state=RANDOM_STATE)
model_xgb = MultiOutputRegressor(XGBRegressor(**xgb_params))
model_xgb.fit(X_train_scaled, Y_train)
print('Parameter estimator dasar:', model_xgb.estimator.get_params())

Y_pred_xgb = model_xgb.predict(X_test_scaled)
detail_xgb, avg_xgb = evaluate_multioutput('XGBoost', Y_test, Y_pred_xgb)
results_summary.append(avg_xgb); results_detail.append(detail_xgb)
detail_xgb.round(3)

plot_diagnostics('XGBoost', Y_test['PTS'].reset_index(drop=True),
                  pd.Series(Y_pred_xgb[:, list(Y.columns).index('PTS')]), target_label='PTS')

#13. Algoritma 4: Lasso Regression

model_lasso = MultiOutputRegressor(Lasso(alpha=0.05, random_state=RANDOM_STATE, max_iter=5000))
model_lasso.fit(X_train_scaled, Y_train)
print('Parameter estimator dasar:', model_lasso.estimator.get_params())

Y_pred_lasso = model_lasso.predict(X_test_scaled)
detail_lasso, avg_lasso = evaluate_multioutput('Lasso Regression', Y_test, Y_pred_lasso)
results_summary.append(avg_lasso); results_detail.append(detail_lasso)
detail_lasso.round(3)

plot_diagnostics('Lasso Regression', Y_test['PTS'].reset_index(drop=True),
                  pd.Series(Y_pred_lasso[:, list(Y.columns).index('PTS')]), target_label='PTS')

lasso_train_pred = model_lasso.predict(X_train_scaled)
print('Perbandingan Train vs Test R2 (indikator overfitting):')
for i, t in enumerate(Y.columns):
    r2_train = r2_score(Y_train.iloc[:, i], lasso_train_pred[:, i])
    r2_test = r2_score(Y_test.iloc[:, i], Y_pred_lasso[:, i])
    print(f'{t:5s} | Train R2={r2_train:.3f}  Test R2={r2_test:.3f}  Gap={r2_train - r2_test:+.3f}')

#14. Algoritma 5: LinearSVR

model_svr = MultiOutputRegressor(LinearSVR(C=1.0, epsilon=0.1, max_iter=5000, random_state=RANDOM_STATE))
model_svr.fit(X_train_scaled, Y_train)

Y_pred_svr = model_svr.predict(X_test_scaled)
detail_svr, avg_svr = evaluate_multioutput('SVM (Linear SVR)', Y_test, Y_pred_svr)
results_summary.append(avg_svr); results_detail.append(detail_svr)
detail_svr.round(3)

#15. Algoritma 6: Ridge Regression

model_ridge = MultiOutputRegressor(Ridge(alpha=1.0, random_state=RANDOM_STATE))
model_ridge.fit(X_train_scaled, Y_train)

Y_pred_ridge = model_ridge.predict(X_test_scaled)
detail_ridge, avg_ridge = evaluate_multioutput('Ridge Regression', Y_test, Y_pred_ridge)
results_summary.append(avg_ridge); results_detail.append(detail_ridge)
detail_ridge.round(3) 
# Diagnostic plot untuk target PTS
plot_diagnostics('Ridge Regression', Y_test['PTS'].reset_index(drop=True),
                  pd.Series(Y_pred_ridge[:, list(Y.columns).index('PTS')]), target_label='PTS')
# Koefisien Ridge untuk target PTS
coefs_ridge = pd.Series(model_ridge.estimators_[0].coef_, index=X_train.columns).sort_values(key=abs, ascending=False).head(10)
plt.figure(figsize=(8,5))
sns.barplot(x=coefs_ridge.values, y=coefs_ridge.index, color='#1d428a')
plt.title('Top 10 Koefisien Ridge Regression (Target: PTS)')
plt.xlabel('Nilai Koefisien')
plt.tight_layout()
simpan_dan_tampilkan('ridge_koefisien_pts')

#16. Teknik Improvement Model (Minimal 5 Teknik)

#Hyperparameter Tuning (RandomizedSearchCV)
param_distributions = {
    'estimator__n_estimators': [150, 250, 350],
    'estimator__max_depth': [3, 4, 5, 6],
    'estimator__learning_rate': [0.02, 0.05, 0.08, 0.1],
    'estimator__min_child_weight': [1, 3, 5, 7],
    'estimator__gamma': [0, 0.5, 1, 2],
    'estimator__reg_alpha': [0, 0.5, 1, 2],
    'estimator__reg_lambda': [1, 2, 3, 5]
}

base_estimator = MultiOutputRegressor(XGBRegressor(random_state=RANDOM_STATE, subsample=0.8, colsample_bytree=0.8))
random_search = RandomizedSearchCV(base_estimator, param_distributions=param_distributions, n_iter=25, cv=5,
                                    scoring='neg_root_mean_squared_error', random_state=RANDOM_STATE, n_jobs=-1)
random_search.fit(X_train_scaled, Y_train)

print('Best params:', random_search.best_params_)
model_xgb_tuned = random_search.best_estimator_

Y_pred_tuned = model_xgb_tuned.predict(X_test_scaled)
detail_tuned, avg_tuned = evaluate_multioutput('XGBoost (Tuned)', Y_test, Y_pred_tuned)
results_summary.append(avg_tuned); results_detail.append(detail_tuned)

# Cek overfitting setelah tuning
tuned_train_pred = model_xgb_tuned.predict(X_train_scaled)
print('\nPerbandingan Train vs Test R2 (indikator overfitting) — XGBoost (Tuned):')
for i, t in enumerate(Y.columns):
    r2_train = r2_score(Y_train.iloc[:, i], tuned_train_pred[:, i])
    r2_test = r2_score(Y_test.iloc[:, i], Y_pred_tuned[:, i])
    print(f'{t:5s} | Train R2={r2_train:.3f}  Test R2={r2_test:.3f}  Gap={r2_train - r2_test:+.3f}')

#Feature Selection
K = 15
selected_per_target = {}
union_features = set()

for target in Y.columns:
    selector = SelectKBest(score_func=f_regression, k=K)
    selector.fit(X_train_scaled, Y_train[target])
    feats = X_train_scaled.columns[selector.get_support()].tolist()
    selected_per_target[target] = feats
    union_features.update(feats)
    print(f'{target}: {feats}')

union_features = sorted(union_features)
print(f'\nTotal fitur gabungan (union): {len(union_features)} dari {X_train_scaled.shape[1]} fitur awal')

X_train_sel = X_train_scaled[union_features]
X_test_sel = X_test_scaled[union_features]

model_fs = MultiOutputRegressor(XGBRegressor(**xgb_params))
model_fs.fit(X_train_sel, Y_train)
Y_pred_fs = model_fs.predict(X_test_sel)
detail_fs, avg_fs = evaluate_multioutput('XGBoost + Feature Selection', Y_test, Y_pred_fs)
results_summary.append(avg_fs); results_detail.append(detail_fs)

#Feature Engineering
def add_engineered_features(data):
    data = data.copy()
    data['usage_proxy'] = (data['fga_per_game'] + data['fta_per_game']*0.44 + data['tov_per_game']) / data['mp_per_game'].replace(0, np.nan)
    data['efficiency'] = data['pts_per_game'] / data['fga_per_game'].replace(0, np.nan)
    data = data.fillna(0)
    return data

X_train_eng = add_engineered_features(X_train_scaled)
X_test_eng = add_engineered_features(X_test_scaled)

model_eng = MultiOutputRegressor(XGBRegressor(**xgb_params))
model_eng.fit(X_train_eng, Y_train)
Y_pred_eng = model_eng.predict(X_test_eng)
detail_eng, avg_eng = evaluate_multioutput('XGBoost + Feature Engineering', Y_test, Y_pred_eng)
results_summary.append(avg_eng); results_detail.append(detail_eng)

#Ensemble (Voting Regressor)
def make_voting_regressor():
    return VotingRegressor(estimators=[
        ('rf', RandomForestRegressor(**rf_params)),
        ('xgb', XGBRegressor(**xgb_params))
    ])

ensemble_model = MultiOutputRegressor(make_voting_regressor())
ensemble_model.fit(X_train_scaled, Y_train)
Y_pred_ens = ensemble_model.predict(X_test_scaled)
detail_ens, avg_ens = evaluate_multioutput('Ensemble (RF + XGBoost)', Y_test, Y_pred_ens)
results_summary.append(avg_ens); results_detail.append(detail_ens)

plot_diagnostics('Ensemble (RF + XGBoost)', Y_test['PTS'].reset_index(drop=True),
                  pd.Series(Y_pred_ens[:, list(Y.columns).index('PTS')]), target_label='PTS')

#Cross Validation (10-Fold)

kf = KFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
cv_scores = cross_val_score(XGBRegressor(**{k.replace('estimator__',''):v for k,v in random_search.best_params_.items()}, random_state=RANDOM_STATE),
                             X_train_scaled, Y_train['PTS'], cv=kf, scoring='neg_root_mean_squared_error')
cv_rmse = -cv_scores

print('RMSE tiap fold (10 kali) — target PTS:')
for i, score in enumerate(cv_rmse, 1):
    print(f'  Fold {i}: {score:.3f}')
print(f'\nRata-rata RMSE  : {cv_rmse.mean():.3f}')
print(f'Standar deviasi : {cv_rmse.std():.3f}')

plt.figure(figsize=(8,4))
plt.plot(range(1,11), cv_rmse, marker='o', color='#c8102e')
plt.axhline(cv_rmse.mean(), color='gray', linestyle='--', label=f'Rata-rata = {cv_rmse.mean():.2f}')
plt.xlabel('Fold ke-')
plt.ylabel('RMSE')
plt.title('10-Fold Cross Validation — XGBoost (Tuned), target PTS')
plt.legend()
plt.tight_layout()
simpan_dan_tampilkan('cv_rmse_per_fold')

#17. Perbandingan Seluruh Model

summary_df = pd.DataFrame(results_summary).set_index('model').round(3)
summary_df = summary_df.sort_values('RMSE')
summary_df

plt.figure(figsize=(10,5))
sns.barplot(x=summary_df.index, y=summary_df['RMSE'], color='#1d428a')
plt.xticks(rotation=35, ha='right')
plt.ylabel('RMSE rata-rata seluruh target (semakin rendah semakin baik)')
plt.title('Perbandingan RMSE Seluruh Model')
plt.tight_layout()
simpan_dan_tampilkan('perbandingan_rmse_seluruh_model')

#Perbandingan R² per Target — Seluruh Model


detail_all = pd.concat(results_detail, ignore_index=True)

plt.figure(figsize=(13,6))
sns.barplot(x='target', y='R2', hue='model', data=detail_all)
plt.title('Perbandingan R² Tiap Model untuk Setiap Target')
plt.xlabel('Target')
plt.ylabel('R²')
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', title='Model')
plt.tight_layout()
simpan_dan_tampilkan('perbandingan_r2_per_target')

#Real vs Prediction (Overlay) — Model Terbaik


# Ganti BEST_MODEL_FOR_PLOT sesuai model dengan performa terbaik
BEST_MODEL_FOR_PLOT = model_svr
Y_pred_best = BEST_MODEL_FOR_PLOT.predict(X_test_scaled)

n_sample = 50
idx_sample = np.arange(min(n_sample, len(Y_test)))
pts_idx = list(Y.columns).index('PTS')

real_vals = Y_test['PTS'].values[idx_sample]
pred_vals = Y_pred_best[idx_sample, pts_idx]

plt.figure(figsize=(12,4.5))
plt.plot(idx_sample, real_vals, marker='*', color='blue', label='real', linewidth=1.2, markersize=8)
plt.plot(idx_sample, pred_vals, marker='o', color='red', label='prediction', linewidth=1.2, markersize=5)
plt.title('Real vs Prediction — PTS per Game (50 Sampel Data Uji)')
plt.xlabel('Index Sampel')
plt.ylabel('PTS per Game')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
simpan_dan_tampilkan('real_vs_prediksi_pts')


# Pilih model final
FINAL_MODEL = model_svr   # alternatif: model_xgb_tuned, model_rf, model_lr, dst.

latest_season = df['season'].max()
print('Musim terakhir pada dataset:', latest_season)

df_latest = df[(df['season'] == latest_season) & (df['g'] >= 15)].copy()
print('Jumlah pemain aktif musim terakhir (g >= 15):', df_latest.shape[0])
df_latest[['player','season','team','pos','pts_per_game']].head()

# Siapkan fitur inferensi dengan pipeline yang SAMA seperti data training
X_latest = df_latest[feature_cols].copy()
X_latest = pd.get_dummies(X_latest, columns=['pos'], prefix='pos')
X_latest = X_latest.reindex(columns=FEATURE_COLUMNS, fill_value=0)

X_latest_scaled = pd.DataFrame(scaler.transform(X_latest), columns=X_latest.columns, index=X_latest.index)

model_needs_engineered = FINAL_MODEL is model_eng
if model_needs_engineered:
    X_latest_final = add_engineered_features(X_latest_scaled)
else:
    X_latest_final = X_latest_scaled

Y_pred_latest = FINAL_MODEL.predict(X_latest_final)
pred_df = pd.DataFrame(Y_pred_latest, columns=Y.columns, index=df_latest.index)
pred_df = pred_df.clip(lower=0)

pred_df.columns = [f'Proyeksi_{c}' for c in pred_df.columns]
print('Prediksi berhasil dibuat untuk', pred_df.shape[0], 'pemain.')

top10 = pd.concat([
    df_latest[['player','team','pos','age']].reset_index(drop=True),
    pred_df.reset_index(drop=True)
], axis=1)

top10 = top10.sort_values('Proyeksi_PTS', ascending=False).head(10).reset_index(drop=True)
top10.index = top10.index + 1
top10 = top10.round({'Proyeksi_PTS':1, 'Proyeksi_AST':1, 'Proyeksi_TRB':1, 'Proyeksi_3P':1, 'Proyeksi_BLK':1})
top10.rename(columns={'player':'Pemain','team':'Tim','pos':'Posisi','age':'Usia'}, inplace=True)

print(f'TOP 10 PROYEKSI PEMAIN — MUSIM {int(latest_season)+1}')
top10

#Klasifikasi Peran Pemain: Offensive / Defensive / Seimbang


pos_to_kategori = {'PG':'Offensive', 'SG':'Offensive', 'SF':'Seimbang', 'PF':'Defensive', 'C':'Defensive'}

# Bangun pool prediksi LENGKAP (seluruh pemain musim terakhir), bukan cuma 10 teratas,
all_pred = pd.concat([
    df_latest[['player','team','pos','age']].reset_index(drop=True),
    pred_df.reset_index(drop=True)
], axis=1)

all_pred['Kategori'] = all_pred['pos'].map(pos_to_kategori)

# Skor komposit per kategori (dipakai untuk ranking masing-masing kelompok)
all_pred['Skor_Offensive'] = all_pred['Proyeksi_PTS']
all_pred['Skor_Defensive'] = all_pred['Proyeksi_TRB'] + all_pred['Proyeksi_BLK'] * 2
all_pred['Skor_Seimbang']  = all_pred['Proyeksi_PTS'] + all_pred['Proyeksi_TRB'] + all_pred['Proyeksi_AST']

all_pred.rename(columns={'player':'Pemain','team':'Tim','pos':'Posisi','age':'Usia'}, inplace=True)
print(all_pred['Kategori'].value_counts())

all_pred['Skor_3PT'] = all_pred['Proyeksi_3P']

fig, axes = plt.subplots(1, 4, figsize=(24, 7))

kategori_config = [
    ('Offensive', 'Skor_Offensive', '#1d428a', 'Proyeksi PTS per Game', axes[0]),
    ('Seimbang',  'Skor_Seimbang',  '#552583', 'Proyeksi PTS + TRB + AST', axes[1]),
    ('Defensive', 'Skor_Defensive', '#c8102e', 'Proyeksi TRB + 2xBLK', axes[2]),
]

for kategori, skor_col, warna, xlabel, ax in kategori_config:
    subset = all_pred[all_pred['Kategori'] == kategori].sort_values(skor_col, ascending=False).head(5)
    sns.barplot(x=skor_col, y='Pemain', data=subset, color=warna, ax=ax)
    for i, v in enumerate(subset[skor_col]):
        ax.text(v + 0.05, i, f'{v:.1f}', va='center', fontsize=9)
    ax.set_title(f'Top 5 {kategori}\n(Posisi: {"PG/SG" if kategori=="Offensive" else "SF" if kategori=="Seimbang" else "PF/C"})')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('')

top5_3pt = all_pred.sort_values('Skor_3PT', ascending=False).head(5)
sns.barplot(x='Skor_3PT', y='Pemain', data=top5_3pt, color='#f9a01b', ax=axes[3])
for i, v in enumerate(top5_3pt['Skor_3PT']):
    axes[3].text(v + 0.02, i, f'{v:.1f}', va='center', fontsize=9)
axes[3].set_title('Top 5 3-Point Leader\n(Lintas Posisi)')
axes[3].set_xlabel('Proyeksi 3P Made per Game')
axes[3].set_ylabel('')

fig.suptitle(f'Prediksi Stats Leader NBA Musim {int(latest_season)+1}', fontsize=13, y=1.03)
plt.tight_layout()
simpan_dan_tampilkan(f'top5_proyeksi_kategori_musim_{int(latest_season)+1}')

#Radar Chart Perbandingan Profil Statistik Kandidat Teratas

stat_cols = ['Proyeksi_PTS','Proyeksi_AST','Proyeksi_TRB','Proyeksi_3P','Proyeksi_BLK']
stat_labels = ['PTS','AST','TRB','3P','BLK']

norm_pred = all_pred.copy()
for c in stat_cols:
    norm_pred[c] = (norm_pred[c] - norm_pred[c].min()) / (norm_pred[c].max() - norm_pred[c].min())

top_offensive_player = all_pred[all_pred['Kategori']=='Offensive'].sort_values('Skor_Offensive', ascending=False).iloc[0]['Pemain']
top_defensive_player = all_pred[all_pred['Kategori']=='Defensive'].sort_values('Skor_Defensive', ascending=False).iloc[0]['Pemain']

angles = [n / len(stat_cols) * 2 * pi for n in range(len(stat_cols))]
angles += angles[:1]

fig, ax = plt.subplots(figsize=(7,7), subplot_kw=dict(polar=True))

for player, color in [(top_offensive_player, '#1d428a'), (top_defensive_player, '#c8102e')]:
    row = norm_pred[norm_pred['Pemain']==player][stat_cols].values.flatten().tolist()
    row += row[:1]
    ax.plot(angles, row, color=color, linewidth=2, label=player)
    ax.fill(angles, row, color=color, alpha=0.15)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(stat_labels)
ax.set_yticklabels([])
ax.set_title('Profil Statistik: Kandidat Offensive Teratas vs Defensive Teratas', y=1.1)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
plt.tight_layout()
simpan_dan_tampilkan('radar_chart_profil_offensive_vs_defensive')

#18. Prediksi Personal Pemain

def cari_prediksi_pemain(nama_pemain, df=all_pred, top_n_saran=3):
    """
    Cari proyeksi statistik musim depan seorang pemain berdasarkan nama.
    Mendukung pencarian sebagian (partial match) dan typo (fuzzy match).
    """
    matches = df[df['Pemain'].str.lower().str.contains(nama_pemain.lower(), na=False)]

    if len(matches) == 0:
        semua_nama = df['Pemain'].tolist()
        saran = difflib.get_close_matches(nama_pemain, semua_nama, n=top_n_saran, cutoff=0.4)
        print(f'Pemain "{nama_pemain}" tidak ditemukan di data musim terakhir.')
        if saran:
            print('Mungkin maksud kamu:', ', '.join(saran))
        else:
            print('Tidak ada nama yang mirip ditemukan. Cek ejaan nama pemain.')
        return None

    if len(matches) > 1:
        print(f'Ditemukan {len(matches)} pemain dengan nama mengandung "{nama_pemain}":')
        print(matches[['Pemain', 'Tim', 'Posisi']].to_string(index=False))
        print('Tuliskan nama lebih spesifik (misal sertakan nama depan & belakang lengkap).')
        return matches

    row = matches.iloc[0]
    print(f'=== Proyeksi Musim {int(latest_season)+1} — {row["Pemain"]} ({row["Tim"]}, {row["Posisi"]}, {row["Usia"]} th) ===')
    print(f'Kategori peran : {row["Kategori"]}')
    print(f'PTS  (Poin)      : {row["Proyeksi_PTS"]:.1f}')
    print(f'AST  (Assist)    : {row["Proyeksi_AST"]:.1f}')
    print(f'TRB  (Rebound)   : {row["Proyeksi_TRB"]:.1f}')
    print(f'3P   (Three Made): {row["Proyeksi_3P"]:.1f}')
    print(f'BLK  (Block)     : {row["Proyeksi_BLK"]:.1f}')
    return row


# CONTOH PEMAKAIAN — ganti nama sesuai pemain yang mau dicek
nama_yang_dicari = input('Masukkan nama pemain: ')
hasil = cari_prediksi_pemain(nama_yang_dicari)

# Ringkasan file visualisasi yang tersimpan
print(f'\nSemua visualisasi telah disimpan sebagai gambar di folder: {os.path.abspath(OUTPUT_DIR)}')
for f in sorted(os.listdir(OUTPUT_DIR)):
    print(' -', f)