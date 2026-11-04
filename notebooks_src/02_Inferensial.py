# %% [markdown]
# # Modul 2 — Statistik Inferensial
# Dua pertanyaan nyata:
# 1. **Apakah energi per ton katoda lebih tinggi pada hari downtime?** (uji beda dua kelompok)
# 2. **Apakah proporsi Grade A berbeda antar bulan?** (uji proporsi / independensi)
#
# Pola kerja setiap uji: hipotesis → cek asumsi → uji → ukuran efek → interval kepercayaan → kesimpulan dalam bahasa operasional.
# %%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

XLSX = next((p for p in ["Smelter_Data_Lanjutan.xlsx", "../data/Smelter_Data_Lanjutan.xlsx", "data/Smelter_Data_Lanjutan.xlsx"] if os.path.exists(p)), "Smelter_Data_Lanjutan.xlsx")
try:
    from google.colab import files
    if not os.path.exists(XLSX): files.upload()
except ImportError:
    pass
S = pd.read_excel(XLSX, sheet_name=None)
h, qc = S["Data_Harian"], S["QC_Lab"]
for df in (h, qc): df["Tanggal"] = pd.to_datetime(df["Tanggal"])
rng = np.random.default_rng(42)
# %% [markdown]
# ## A. Energi per ton: hari downtime vs hari normal
# **H0:** rata-rata energi/t sama. **H1:** berbeda. Kelompok: hari dengan `Jam_Downtime > 0` vs `= 0`.
# %%
h["Ada_Downtime"] = h.Jam_Downtime > 0
a = h.loc[h.Ada_Downtime, "Energi_per_t_Katoda_MWh"].values     # hari downtime
b = h.loc[~h.Ada_Downtime, "Energi_per_t_Katoda_MWh"].values    # hari normal
print(f"n downtime={len(a)}, n normal={len(b)}")
print(f"mean downtime={a.mean():.4f}  mean normal={b.mean():.4f}  selisih={a.mean()-b.mean():+.4f} MWh/t ({(a.mean()/b.mean()-1)*100:+.1f}%)")
# %% [markdown]
# ### Cek asumsi
# - **Normalitas** (Shapiro–Wilk): p < 0,05 berarti sebaran menyimpang dari normal.
# - **Varians sama** (Levene): menentukan apakah memakai uji t Student atau Welch. Welch lebih aman, jadi dipakai default.
# %%
print("Shapiro downtime: p=%.3f | normal: p=%.3f" % (stats.shapiro(a).pvalue, stats.shapiro(b).pvalue))
print("Levene (varians sama?): p=%.3f" % stats.levene(a, b).pvalue)
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].hist(b, bins=20, alpha=.7, label="Normal", color="#7A8CA3", density=True)
ax[0].hist(a, bins=20, alpha=.7, label="Downtime", color="#C9A227", density=True); ax[0].legend(); ax[0].set_title("Sebaran energi/t")
stats.probplot(a - a.mean(), plot=ax[1]); ax[1].set_title("Q-Q plot kelompok downtime")
plt.tight_layout(); plt.show()
# %% [markdown]
# ### Uji Welch + alternatif non-parametrik + ukuran efek + bootstrap
# %%
t = stats.ttest_ind(a, b, equal_var=False)
u = stats.mannwhitneyu(a, b, alternative="two-sided")
print(f"Welch t-test      : t={t.statistic:.3f}, p={t.pvalue:.2e}")
print(f"Mann-Whitney U    : U={u.statistic:.0f}, p={u.pvalue:.2e}")

# Cohen's d memakai simpangan baku gabungan; Hedges' g = koreksi sampel kecil
sp = np.sqrt(((len(a)-1)*a.var(ddof=1) + (len(b)-1)*b.var(ddof=1)) / (len(a)+len(b)-2))
d = (a.mean() - b.mean()) / sp
g = d * (1 - 3 / (4*(len(a)+len(b)) - 9))
print(f"Cohen's d={d:.2f}  Hedges' g={g:.2f}  (|d|: 0.2 kecil, 0.5 sedang, 0.8 besar)")

# Bootstrap: ambil ulang sampel dengan pengembalian 10.000 kali -> sebaran selisih rata-rata
boot = np.array([rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(10000)])
lo, hi = np.percentile(boot, [2.5, 97.5])
print(f"Selisih rata-rata: {a.mean()-b.mean():+.4f}, CI95% bootstrap [{lo:+.4f}, {hi:+.4f}]")
# %% [markdown]
# ### Efek hari downtime itu sendiri, atau karena durasinya?
# Regresi sederhana: energi/t terhadap jam downtime pada hari downtime saja.
# %%
dd = h[h.Ada_Downtime]
r = stats.linregress(dd.Jam_Downtime, dd.Energi_per_t_Katoda_MWh)
print(f"Hanya hari downtime (n={len(dd)}): kemiringan={r.slope:+.4f} MWh/t per jam, r={r.rvalue:.2f}, p={r.pvalue:.3g}")
rho = stats.spearmanr(dd.Jam_Downtime, dd.Energi_per_t_Katoda_MWh)
print(f"Spearman rho={rho.statistic:.2f}, p={rho.pvalue:.3g}")
# %% [markdown]
# **Cara membaca:** p kecil + efek besar = perbedaan nyata dan penting. p kecil + efek sangat kecil = nyata secara statistik tetapi mungkin tidak penting secara operasional. Selalu laporkan keduanya.
# %% [markdown]
# ## B. Proporsi Grade A antar bulan
# **H0:** peluang Grade A sama di semua bulan. Data: tabel kontingensi bulan × status.
# %%
qc["Bulan"] = qc.Tanggal.dt.month
qc["GradeA"] = (qc.Status_Mutu == "Grade A").astype(int)
tab = qc.groupby("Bulan").GradeA.agg(["sum", "count"]); tab.columns = ["GradeA", "Batch"]
tab["OffGrade"] = tab.Batch - tab.GradeA; tab["Prop_A_%"] = tab.GradeA / tab.Batch * 100
print(tab.round(1)); print("Total Grade A: %d/%d" % (tab.GradeA.sum(), tab.Batch.sum()))
# %%
ct = tab[["GradeA", "OffGrade"]].values
chi2, p, dof, exp = stats.chi2_contingency(ct)
print(f"Chi-square: chi2={chi2:.2f}, dof={dof}, p={p:.3f}")
print(f"Sel dengan frekuensi harapan < 5: {(exp < 5).sum()} dari {exp.size}  -> asumsi chi-square {'TERLANGGAR' if (exp<5).any() else 'aman'}")
v = np.sqrt(chi2 / (ct.sum() * (min(ct.shape) - 1)))
print(f"Cramer's V = {v:.3f}")

# Uji permutasi: tidak bergantung pada asumsi sampel besar
def chi2_stat(status, bulan):
    obs = pd.crosstab(bulan, status).values
    e = obs.sum(1, keepdims=True) * obs.sum(0, keepdims=True) / obs.sum()
    return ((obs - e) ** 2 / e).sum()
obs_stat = chi2_stat(qc.GradeA.values, qc.Bulan.values)
perm = np.array([chi2_stat(rng.permutation(qc.GradeA.values), qc.Bulan.values) for _ in range(5000)])
print(f"Permutasi: statistik={obs_stat:.2f}, p={(perm >= obs_stat).mean():.3f}")
# %% [markdown]
# ### Uji tren Cochran–Armitage (apakah proporsi naik/turun menurut bulan?)
# %%
x = tab.index.values.astype(float); n = tab.Batch.values; r_ = tab.GradeA.values
pbar = r_.sum() / n.sum()
T = ((x - (n*x).sum()/n.sum()) * (r_ - n*pbar)).sum()
varT = pbar*(1-pbar) * (n * (x - (n*x).sum()/n.sum())**2).sum()
z = T / np.sqrt(varT); pz = 2 * (1 - stats.norm.cdf(abs(z)))
print(f"Cochran-Armitage: z={z:.2f}, p={pz:.3f}  ({'ada' if pz<.05 else 'tidak ada'} tren linear menurut bulan)")
# %% [markdown]
# ### Pembandingan berpasangan + koreksi Holm
# 9 bulan = 36 pasangan. Menguji 36 kali menaikkan peluang "temuan palsu" — karena itu p-value dikoreksi (Holm).
# %%
from itertools import combinations
baris = []
for i, j in combinations(tab.index, 2):
    tt = [[tab.GradeA[i], tab.OffGrade[i]], [tab.GradeA[j], tab.OffGrade[j]]]
    baris.append((i, j, stats.fisher_exact(tt)[1]))
pw = pd.DataFrame(baris, columns=["bulan_i", "bulan_j", "p_mentah"])
m_ = len(pw); urut = pw.p_mentah.sort_values().index
holm = pd.Series(index=pw.index, dtype=float); prev = 0
for k, idx in enumerate(urut):
    prev = max(prev, min(1, pw.p_mentah[idx] * (m_ - k))); holm[idx] = prev
pw["p_Holm"] = holm
print(f"{(pw.p_mentah<.05).sum()} pasangan p<0.05 sebelum koreksi; {(pw.p_Holm<.05).sum()} setelah Holm")
print(pw.sort_values("p_mentah").head(5).round(3).to_string(index=False))
# %% [markdown]
# **Kesimpulan modul 2.** Tuliskan dalam bahasa operasional, misalnya: *"Energi per ton pada hari downtime X% lebih tinggi (CI95% ...), efek ... Proporsi Grade A tidak/ berbeda nyata antarbulan (p=...), sehingga fluktuasi bulanan wajar sebagai variasi acak."*
