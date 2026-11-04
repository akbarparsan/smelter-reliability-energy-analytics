# %% [markdown]
# # Modul 4 — Statistik Keandalan (Reliability)
# Pertanyaan teknik: *seberapa sering tiap unit gagal, seberapa lama diperbaiki, apakah kegagalan makin sering, dan kapan komponen sebaiknya diganti?*
#
# Isi: (A) MTBF/MTTR/availability per unit dari `Downtime_Log`, (B) uji tren Laplace + model Crow-AMSAA, (C) Weibull dengan data tersensor & left-truncated dari `Komponen_Kritis`, (D) umur B10 dan interval penggantian.
# %%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats, optimize

pd.set_option("display.width", 140, "display.max_columns", 30)
XLSX = next((p for p in ["Smelter_Data_Lanjutan.xlsx", "../data/Smelter_Data_Lanjutan.xlsx", "data/Smelter_Data_Lanjutan.xlsx"] if os.path.exists(p)), "Smelter_Data_Lanjutan.xlsx")
try:
    from google.colab import files
    if not os.path.exists(XLSX): files.upload()
except ImportError:
    pass
S = pd.read_excel(XLSX, sheet_name=None)
log, harian, komp = S["Downtime_Log"], S["Data_Harian"], S["Komponen_Kritis"]
log["Tanggal"] = pd.to_datetime(log.Tanggal); harian["Tanggal"] = pd.to_datetime(harian.Tanggal)
rng = np.random.default_rng(7)
T0 = harian.Tanggal.min(); HARI = len(harian); JAM = HARI * 24
tak = log[log.Kategori == "Tidak Terjadwal"].copy()
tak["t_hari"] = (tak.Tanggal - T0).dt.days + 0.5            # waktu kejadian (hari sejak 1 Jan)
print(f"Jendela: {HARI} hari = {JAM} jam | kejadian tidak terjadwal: {len(tak)}")
# %% [markdown]
# ## A. MTBF, MTTR, dan availability inheren per unit
# - **MTTR** = rata-rata durasi perbaikan.
# - **MTBF** = jam operasi ÷ jumlah kegagalan, dengan jam operasi unit ≈ jendela − downtime unit itu.
# - **Availability inheren** = MTBF / (MTBF + MTTR).
# %%
u = tak.groupby("Unit").agg(Kejadian=("Durasi_Jam", "size"), Jam_Down=("Durasi_Jam", "sum"), MTTR_jam=("Durasi_Jam", "mean"))
u["Jam_Operasi"] = JAM - u.Jam_Down
u["MTBF_jam"] = u.Jam_Operasi / u.Kejadian
u["Avail_inheren_%"] = u.MTBF_jam / (u.MTBF_jam + u.MTTR_jam) * 100
print(u.sort_values("Jam_Down", ascending=False).round(1))
n, dt = len(tak), tak.Durasi_Jam.sum()
print(f"\nPABRIK: MTBF={(JAM-dt)/n:.1f} jam (≈{(JAM-dt)/n/24:.1f} hari), MTTR={dt/n:.2f} jam")
# %% [markdown]
# ## B. Apakah kegagalan makin sering? — uji Laplace & Crow-AMSAA
# Uji Laplace: jika kejadian cenderung bergeser ke akhir jendela, statistik **U > 0** (memburuk); **U < 0** membaik; |U| > 1,96 berarti signifikan pada 5%.
# %%
def laplace(t, T):
    t = np.asarray(t, float); n = len(t); U = (t.mean() - T/2) / (T / np.sqrt(12*n)); return U, 2*stats.norm.sf(abs(U))
U, p = laplace(tak.t_hari, HARI)
print(f"Seluruh pabrik: U={U:.2f}, p={p:.3f} -> {'ada tren' if p<.05 else 'tidak ada tren signifikan'}")
print("Per unit (n kecil, hati-hati):")
for unit, g in tak.groupby("Unit"):
    if len(g) >= 4: Uu, pu = laplace(g.t_hari, HARI); print(f"  {unit:30s} n={len(g):2d}  U={Uu:+.2f}  p={pu:.3f}")

# Crow-AMSAA (NHPP hukum pangkat): N(t)=lambda * t^beta. beta<1 membaik, >1 memburuk. MLE untuk data dipotong waktu:
t = np.sort(tak.t_hari.values); beta = len(t) / np.log(HARI / t).sum(); lam = len(t) / HARI**beta
# CI beta: 2*n*beta_true/beta_hat ~ chi2(2n)  (kira-kira)
lo, hi = beta * stats.chi2.ppf([.025, .975], 2*len(t)) / (2*len(t))
print(f"\nCrow-AMSAA: beta={beta:.2f} (CI95% ≈ [{lo:.2f}, {hi:.2f}]), lambda={lam:.4f}")
fig, ax = plt.subplots(figsize=(8, 4))
ax.step(t, np.arange(1, len(t)+1), where="post", color="#1F3A5F", label="Kumulatif kejadian")
g_ = np.linspace(1, HARI, 200); ax.plot(g_, lam * g_**beta, "--", color="#C9A227", label=f"Crow-AMSAA β={beta:.2f}")
ax.plot(g_, len(t)*g_/HARI, ":", color="gray", label="Laju konstan"); ax.set_xlabel("hari"); ax.set_ylabel("kejadian kumulatif"); ax.legend()
plt.tight_layout(); plt.show()
# %% [markdown]
# ## C. Weibull dengan sensor & left truncation
# Data `Komponen_Kritis`: tiap komponen punya `Umur_Awal_Hari` (umur saat pengamatan dimulai), `Durasi_Pengamatan_Hari`, dan `Gagal` (1 = gagal, 0 = masih jalan/tersensor).
#
# Dua kesalahan klasik: **(1)** membuang komponen yang belum gagal (membuat umur terlihat terlalu pendek), **(2)** mengabaikan bahwa komponen sudah berumur saat diamati (left truncation). Likelihood yang benar:
# `logL = Σ[ gagal · log h(t_keluar) ] − Σ[ H(t_keluar) − H(t_masuk) ]`, dengan hazard Weibull h(t) = (β/η)(t/η)^(β−1) dan H(t) = (t/η)^β.
# %%
np.seterr(all="ignore")   # percobaan parameter ekstrem oleh optimizer boleh menghasilkan overflow sementara
def fit_weibull(masuk, keluar, gagal, truncation=True):
    masuk, keluar, gagal = map(lambda x: np.asarray(x, float), (masuk, keluar, gagal))
    if not truncation: masuk = np.zeros_like(masuk)
    def nll(par):
        b, e = np.exp(par)
        ll = (gagal * (np.log(b) - b*np.log(e) + (b-1)*np.log(keluar))).sum() - (((keluar/e)**b) - ((masuk/e)**b)).sum()
        return -ll if np.isfinite(ll) else 1e12
    r = optimize.minimize(nll, [0.0, np.log(keluar.mean())], method="Nelder-Mead", options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 4000})
    b, e = np.exp(r.x); return b, e, -r.fun

komp["t_masuk"] = komp.Umur_Awal_Hari
komp["t_keluar"] = komp.Umur_Awal_Hari + komp.Durasi_Pengamatan_Hari
hasil = []
for jenis, g in komp.groupby("Jenis"):
    b, e, ll = fit_weibull(g.t_masuk, g.t_keluar, g["Gagal_1_Sensor_0"])
    b_naif, e_naif, _ = fit_weibull(g.t_masuk, g.t_keluar, g["Gagal_1_Sensor_0"], truncation=False)
    # uji rasio likelihood: apakah shape = 1 (laju gagal konstan)? eksponensial: lambda = gagal / total paparan
    expo = g["Gagal_1_Sensor_0"].sum() / (g.t_keluar - g.t_masuk).sum()
    ll0 = (g["Gagal_1_Sensor_0"].sum() * np.log(expo)) - expo * (g.t_keluar - g.t_masuk).sum()
    lrt = 2 * (ll - ll0)
    hasil.append(dict(Jenis=jenis, n=len(g), gagal=int(g["Gagal_1_Sensor_0"].sum()), shape=b, scale_eta=e,
                      shape_tanpa_truncation=b_naif, p_shape_vs_1=stats.chi2.sf(max(lrt, 0), 1)))
W = pd.DataFrame(hasil).set_index("Jenis"); print(W.round(3))
print("\nshape>1 = aus/menua (penggantian preventif masuk akal); ≈1 = acak (preventif tidak membantu); <1 = gagal dini.")
# %% [markdown]
# ### Ketidakpastian: bootstrap parametrik atas komponen
# Dengan n kecil (6–12 komponen), titik estimasi saja menyesatkan. Kita ambil ulang komponen dengan pengembalian 1.000 kali.
# %%
ci = {}
for jenis, g in komp.groupby("Jenis"):
    g = g.reset_index(drop=True); bs = []
    for _ in range(1000):
        s = g.sample(len(g), replace=True, random_state=int(rng.integers(1e9)))
        if s["Gagal_1_Sensor_0"].sum() >= 2: bs.append(fit_weibull(s.t_masuk, s.t_keluar, s["Gagal_1_Sensor_0"])[:2])
    bs = np.array(bs); ci[jenis] = np.percentile(bs[:, 0], [2.5, 97.5]).round(2).tolist() + np.percentile(bs[:, 1], [2.5, 97.5]).round(0).tolist()
C = pd.DataFrame(ci, index=["shape_lo", "shape_hi", "eta_lo", "eta_hi"]).T; print(pd.concat([W[["shape", "scale_eta"]].round(2), C], axis=1))
# %% [markdown]
# ## D. Keputusan: umur B10 dan keandalan pada waktu tertentu
# - **R(t) = exp(−(t/η)^β)** peluang bertahan sampai umur t (tanpa syarat).
# - **B10** = umur ketika 10% komponen sudah gagal: `η·(−ln 0.9)^(1/β)`. Dipakai sebagai dasar interval penggantian preventif.
# %%
W["B10_hari"] = W.scale_eta * (-np.log(0.9)) ** (1 / W["shape"])
W["R_90hari_%"] = np.exp(-(90 / W.scale_eta) ** W["shape"]) * 100
from math import gamma
W["MTTF_hari"] = W.scale_eta * np.array([gamma(1 + 1 / sh) for sh in W["shape"]])   # rata-rata umur = eta * Gamma(1+1/beta)
print(W[["shape", "scale_eta", "B10_hari", "MTTF_hari", "R_90hari_%"]].round(1))

fig, ax = plt.subplots(1, 2, figsize=(11, 4)); tt = np.linspace(0, 450, 300)
for jenis, r in W.iterrows():
    ax[0].plot(tt, np.exp(-(tt / r.scale_eta) ** r["shape"]), label=f"{jenis} (β={r['shape']:.1f})")
    ax[1].plot(tt[1:], (r["shape"]/r.scale_eta) * (tt[1:]/r.scale_eta) ** (r["shape"]-1), label=jenis)
ax[0].set_title("Keandalan R(t)"); ax[0].set_xlabel("umur (hari)"); ax[0].legend(fontsize=7)
ax[1].set_title("Laju kegagalan h(t)"); ax[1].set_xlabel("umur (hari)"); ax[1].set_ylim(0, .03)
plt.tight_layout(); plt.show()
# %% [markdown]
# **Kesimpulan modul 4.** Bacalah tiga hal per komponen: (1) shape β dan p_shape_vs_1 → apakah ada keausan; (2) B10 → kandidat interval penggantian; (3) lebar CI bootstrap → seberapa yakin. Komponen β≈1 tidak diuntungkan penggantian terjadwal; perbaikannya lebih efektif lewat desain/kondisi. Fase 2 melanjutkan dengan Kaplan–Meier dan model Cox.
