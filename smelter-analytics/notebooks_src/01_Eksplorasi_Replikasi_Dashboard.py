# %% [markdown]
# # Modul 1 — Eksplorasi & Replikasi Dashboard
# **Tujuan:** (a) membuktikan angka Python = angka Power BI, (b) memperdalam apa yang tidak bisa dijawab dashboard: korelasi, pola hari, dan dekomposisi selisih output.
# %%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option("display.width", 140, "display.max_columns", 30)
XLSX = next((p for p in ["Smelter_Data_Lanjutan.xlsx", "../data/Smelter_Data_Lanjutan.xlsx", "data/Smelter_Data_Lanjutan.xlsx"] if os.path.exists(p)), "Smelter_Data_Lanjutan.xlsx")
try:
    from google.colab import files
    if not os.path.exists(XLSX): files.upload()
except ImportError:
    pass
S = pd.read_excel(XLSX, sheet_name=None)
h, log, qc, pr = S["Data_Harian"], S["Downtime_Log"], S["QC_Lab"], S["Proses_Harian"]
for df in (h, log, qc, pr): df["Tanggal"] = pd.to_datetime(df["Tanggal"])
NAMA_BULAN = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "Mei", 6: "Jun", 7: "Jul", 8: "Agu", 9: "Sep"}
# %% [markdown]
# ## 1. KPI utama (cocokkan dengan kartu di Power BI)
# Target total 9 bulan = 17.000 × 9 = 153.000 t. Untuk perbandingan bulanan dipakai target **prorata hari** (153.000 t ÷ 273 hari = 560,44 t/hari), sebab bulan 28–31 hari tidak adil dibandingkan dengan target flat.
# %%
TARGET_BULANAN = 17000
katoda = h.Katoda_Cu_t.sum()
target = TARGET_BULANAN * 9
recovery = h.Katoda_Cu_t.sum() / h.Cu_Masuk_t.sum() * 100          # rasio total, BUKAN rata-rata persen harian
availability = h.Jam_Operasi.sum() / (24 * len(h)) * 100
grade_a = (qc.Status_Mutu == "Grade A").sum()
print(f"Katoda total      : {katoda:,.1f} t")
print(f"Target 9 bulan    : {target:,} t   selisih {katoda-target:+,.0f} t ({(katoda/target-1)*100:+.1f}%)")
print(f"Recovery          : {recovery:.1f}%   (rata-rata sederhana harian = {h.Recovery_Persen.mean():.2f}%)")
print(f"Availability      : {availability:.1f}%")
print(f"Grade A           : {grade_a}/{len(qc)} = {grade_a/len(qc)*100:.1f}%")
# %% [markdown]
# **Catatan penting:** rata-rata dari persentase harian ≠ rasio total. Di DAX, ini sama dengan membedakan `AVERAGE(kolom)` vs `DIVIDE(SUM(a), SUM(b))`. Dashboard memakai rasio total.
# %% [markdown]
# ## 2. Ringkasan bulanan
# %%
h["Bulan_Nama"] = h.Bulan.map(NAMA_BULAN)
bln = h.groupby("Bulan").agg(Hari=("Tanggal", "size"), Katoda_t=("Katoda_Cu_t", "sum"), Downtime_jam=("Jam_Downtime", "sum"),
                             Cu_Masuk_t=("Cu_Masuk_t", "sum"), Energi_MWh=("Energi_MWh", "sum"), Jam_Operasi=("Jam_Operasi", "sum"))
TARGET_HARIAN = TARGET_BULANAN * 9 / len(h)        # 153.000 t / 273 hari = 560,44 t/hari
bln["Target_t"] = TARGET_HARIAN * bln.Hari
bln["vs_Target_t"] = bln.Katoda_t - bln.Target_t
bln["vs_Target_%"] = bln.vs_Target_t / bln.Target_t * 100
bln["Recovery_%"] = bln.Katoda_t / bln.Cu_Masuk_t * 100
bln["Availability_%"] = bln.Jam_Operasi / (24 * bln.Hari) * 100
bln["Energi_per_t"] = bln.Energi_MWh / bln.Katoda_t
print(bln.round(2).drop(columns=["Cu_Masuk_t", "Jam_Operasi"]))
# %% [markdown]
# ## 3. Pareto penyebab downtime tidak terencana
# Kumulatif harus dibagi **total seluruh penyebab** (setara `ALL(penyebab)` di DAX), bukan total dari bar yang tampil.
# %%
tak = log[log.Kategori.str.contains("Tidak", case=False)]
print("Kategori:", log.Kategori.unique(), "| kejadian tidak terencana:", len(tak))
par = tak.groupby("Penyebab").Durasi_Jam.sum().sort_values(ascending=False).to_frame("Jam")
par["Persen"] = par.Jam / par.Jam.sum() * 100
par["Kumulatif_%"] = par.Persen.cumsum()
print(f"Total tidak terencana: {par.Jam.sum():.1f} jam")
print(par.head(8).round(1))
print(f"\nTop-5 = {par['Kumulatif_%'].iloc[4]:.1f}% dari total")

top = par.head(10)
fig, ax = plt.subplots(figsize=(9, 4.5))
warna = ["#1F3A5F" if i < 5 else "#B8C0CC" for i in range(len(top))]
ax.barh(top.index[::-1], top.Jam[::-1], color=warna[::-1])
ax.set_xlabel("Jam downtime"); ax.set_title("Pareto: 10 penyebab terbesar (biru = top-5)")
plt.tight_layout(); plt.show()
# %% [markdown]
# ## 4. Dekomposisi selisih output vs baseline (volume, grade, recovery)
# Katoda bulan *m* = konsentrat × grade × recovery. Baseline = rata-rata **YTD per hari**. Tiap bulan, selisihnya diurai menjadi tiga efek yang **tepat berjumlah** selisih total:
# - efek volume = (konsentrat_m − hari × D0) · g0 · r0
# - efek grade  = konsentrat_m · (g_m − g0) · r0
# - efek recovery = Cu_masuk_m · (r_m − r0)
# %%
D0 = h.Konsentrat_Masuk_DMT.sum() / len(h)                       # konsentrat per hari
g0 = h.Cu_Masuk_t.sum() / h.Konsentrat_Masuk_DMT.sum()           # grade Cu efektif (fraksi)
r0 = h.Katoda_Cu_t.sum() / h.Cu_Masuk_t.sum()                    # recovery (fraksi)
m = h.groupby("Bulan").agg(n=("Tanggal", "size"), d=("Konsentrat_Masuk_DMT", "sum"), cu=("Cu_Masuk_t", "sum"), kat=("Katoda_Cu_t", "sum"))
m["g"] = m.cu / m.d; m["r"] = m.kat / m.cu
m["Efek_Volume"] = (m.d - m.n * D0) * g0 * r0
m["Efek_Grade"] = m.d * (m.g - g0) * r0
m["Efek_Recovery"] = m.cu * (m.r - r0)
m["Total"] = m.kat - m.n * D0 * g0 * r0
print(m[["Efek_Volume", "Efek_Grade", "Efek_Recovery", "Total"]].round(1))
print("Cek identitas (selisih maks):", (m.Efek_Volume + m.Efek_Grade + m.Efek_Recovery - m.Total).abs().max().round(10))

e = m[["Efek_Volume", "Efek_Grade", "Efek_Recovery"]]
fig, ax = plt.subplots(figsize=(9, 4))
bawah_p = np.zeros(len(e)); bawah_n = np.zeros(len(e))
for kol, wr in zip(e.columns, ["#1F3A5F", "#C9A227", "#7A8CA3"]):
    v = e[kol].values; bawah = np.where(v >= 0, bawah_p, bawah_n)
    ax.bar([NAMA_BULAN[i] for i in e.index], v, bottom=bawah, color=wr, label=kol.replace("Efek_", ""))
    bawah_p += np.where(v >= 0, v, 0); bawah_n += np.where(v < 0, v, 0)
ax.axhline(0, color="k", lw=.8); ax.set_ylabel("ton katoda"); ax.legend(ncol=3); ax.set_title("Efek terhadap baseline YTD")
plt.tight_layout(); plt.show()
# %% [markdown]
# ## 5. Memperdalam: korelasi, pola hari, hubungan downtime–output
# %%
j = h.merge(pr, on="Tanggal")
kol = ["Konsentrat_Masuk_DMT", "Cu_Grade_Persen", "Jam_Downtime", "Katoda_Cu_t", "Recovery_Persen", "Energi_per_t_Katoda_MWh",
       "Suhu_FSF_C", "O2_Enrichment_Persen", "Cu_Slag_Persen", "Kelembaban_Konsentrat_Persen", "Curah_Hujan_mm"]
kor = j[kol].corr(method="spearman")
fig, ax = plt.subplots(figsize=(8, 6.5))
im = ax.imshow(kor, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(kol))); ax.set_xticklabels([k[:14] for k in kol], rotation=60, ha="right", fontsize=8)
ax.set_yticks(range(len(kol))); ax.set_yticklabels([k[:14] for k in kol], fontsize=8)
for a in range(len(kol)):
    for b in range(len(kol)): ax.text(b, a, f"{kor.iloc[a, b]:.1f}", ha="center", va="center", fontsize=6)
plt.colorbar(im, shrink=.8); ax.set_title("Korelasi Spearman"); plt.tight_layout(); plt.show()
print("Korelasi terkuat dengan Energi per ton:\n", kor["Energi_per_t_Katoda_MWh"].drop("Energi_per_t_Katoda_MWh").sort_values(key=abs, ascending=False).round(2).head(5))
# %%
h["Hari"] = h.Tanggal.dt.day_name()
urut = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
pola = h.groupby("Hari").agg(Rata_katoda=("Katoda_Cu_t", "mean"), Rata_downtime=("Jam_Downtime", "mean"),
                              Hari_downtime=("Jam_Downtime", lambda s: (s > 0).sum())).reindex(urut)
print(pola.round(2))
h["Ada_Downtime"] = h.Jam_Downtime > 0
print("\nRata-rata katoda: hari downtime vs normal")
print(h.groupby("Ada_Downtime").agg(n=("Tanggal", "size"), katoda=("Katoda_Cu_t", "mean"), energi_per_t=("Energi_per_t_Katoda_MWh", "mean")).round(3))
# %% [markdown]
# **Kesimpulan modul 1.**
(1) Katoda 156.326 t, 2,2% di atas target 153.000 t. Recovery 98,1%, availability 95,0%, Grade A 94,5% (258 dari 273 batch).
(2) Dengan target prorata, April paling buruk (−5,9%, downtime 102,5 jam) dan September paling baik (+8,1%).
(3) Lima penyebab henti teratas menyumbang 61,1% jam henti, dengan trip kompresor udara di urutan pertama (34 jam).
(4) Hari ada downtime memproduksi 479 t/hari, sedangkan hari normal 603 t/hari. Energi per ton juga lebih tinggi (1,513 vs 1,456).
