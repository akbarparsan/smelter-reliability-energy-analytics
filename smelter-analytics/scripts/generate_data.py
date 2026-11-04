"""Membangun data/Smelter_Data_Lanjutan.xlsx dari data/raw/*.csv (3 tabel asli tidak diubah) + tabel simulasi (seed 2026).
Jalankan dari akar repositori:  python scripts/generate_data.py [berkas_keluaran.xlsx]"""
import numpy as np, pandas as pd
SRC = "data/raw/"
import sys
OUT = sys.argv[1] if len(sys.argv) > 1 else "data/Smelter_Data_Lanjutan.xlsx"
rng = np.random.default_rng(2026)
d = pd.read_csv(SRC+"Data_Harian.csv", parse_dates=["Tanggal"])
log = pd.read_csv(SRC+"Downtime_Log.csv", parse_dates=["Tanggal"])
qc = pd.read_csv(SRC+"QC_Lab.csv", parse_dates=["Tanggal"])
n = len(d)
z = lambda s: (s - s.mean()) / s.std()

# ---- 1. Proses_Harian (sensor harian, berkorelasi dengan energi/t & recovery) ----
doy = d.Tanggal.dt.dayofyear.values
suhu_amb = 27 + 2.5*np.sin((doy-60)/365*2*np.pi) + rng.normal(0, 1.0, n)
hujan = np.where(rng.random(n) < 0.25 + 0.15*np.sin((doy-340)/365*2*np.pi).clip(0), rng.gamma(2, 8, n), 0).round(1)
ept = d.Energi_per_t_Katoda_MWh
rec = d.Recovery_Persen
proses = pd.DataFrame({
  "Tanggal": d.Tanggal,
  "Suhu_FSF_C": (1255 + 9*z(ept) + rng.normal(0, 9, n)).round(1),
  "O2_Enrichment_Persen": (62 - 1.0*z(ept) + rng.normal(0, 1.1, n)).round(1),
  "Matte_Grade_Persen": (63 + 1.1*z(d.Cu_Grade_Persen) + rng.normal(0, 0.7, n)).round(2),
  "Cu_Slag_Persen": (0.78 - 0.11*z(rec) + rng.normal(0, 0.04, n)).clip(0.45).round(3),
  "Kelembaban_Konsentrat_Persen": (8.2 + 0.9*(hujan > 0) + 0.02*hujan + rng.normal(0, 0.35, n)).round(2),
  "Suhu_Ambien_C": suhu_amb.round(1),
  "Curah_Hujan_mm": hujan,
})

# ---- 2. Produksi_Shift (3 shift/hari; jumlah per hari = Data_Harian persis) ----
SH = ["Pagi", "Siang", "Malam"]
logd = log.copy()
down = np.zeros((n, 3))
idx = {t: i for i, t in enumerate(d.Tanggal)}
for _, r in logd.iterrows():
    i = idx[r.Tanggal]; rem = r.Durasi_Jam
    s = rng.integers(0, 3)
    while rem > 1e-9:
        free = 8 - down[i, s]
        take = min(free, rem); down[i, s] += take; rem -= take; s = (s+1) % 3
assert np.allclose(down.sum(1), d.Jam_Downtime)
def split(total, w, dec):
    raw = total*w/w.sum(); v = np.round(raw, dec); v[np.argmax(w)] += round(total - v.sum(), dec); return v
rows = []
crew = ["A", "B", "C", "D"]
for i, r in d.iterrows():
    w = (8 - down[i]) * rng.normal(1, 0.04, 3) + 1e-6
    kat = split(r.Katoda_Cu_t, w, 1)
    eng = split(r.Energi_MWh, (8 - down[i])*rng.normal(1, 0.05, 3) + 1e-6 + 0.3, 0)
    for s in range(3):
        rows.append([r.Tanggal, SH[s], crew[(i + s) % 4], round(8-down[i, s], 2), round(down[i, s], 2), kat[s], eng[s]])
shift = pd.DataFrame(rows, columns=["Tanggal", "Shift", "Regu", "Jam_Operasi", "Jam_Downtime", "Katoda_Cu_t", "Energi_MWh"])
shift.insert(0, "ID_Shift", [f"SH-{k+1:04d}" for k in range(len(shift))])
assert np.allclose(shift.groupby("Tanggal").Katoda_Cu_t.sum().values, d.Katoda_Cu_t, atol=1e-6)
assert np.allclose(shift.groupby("Tanggal").Energi_MWh.sum().values, d.Energi_MWh)

# ---- 3. Harga_Biaya (untuk Monte Carlo fase 2) ----
lme = 10200 + np.cumsum(rng.normal(0, 70, n)); lme = lme - lme.mean() + 10200
harga = pd.DataFrame({
  "Tanggal": d.Tanggal,
  "LME_Cu_USD_per_t": lme.round(0),
  "Tarif_Listrik_USD_per_MWh": (72 + 4*np.sin(doy/365*2*np.pi) + rng.normal(0, 1.5, n)).round(2),
  "Kurs_IDR_per_USD": (16200 + np.cumsum(rng.normal(0, 12, n))).round(0),
})
harga["Biaya_Energi_USD"] = (d.Energi_MWh*harga.Tarif_Listrik_USD_per_MWh).round(0)
harga["Nilai_Katoda_USD"] = (d.Katoda_Cu_t*harga.LME_Cu_USD_per_t).round(0)

# ---- 4. Komponen_Kritis (armada simulasi untuk survival; BUKAN 1:1 dengan Downtime_Log) ----
spec = [("Tuyere", "Flash Smelting Furnace", 12, 1.6, 150), ("Blower SO2", "Acid Plant", 6, 1.2, 260),
        ("Kompresor Udara", "Oxygen Plant", 6, 1.9, 200), ("Waste Heat Boiler Tube", "Flash Smelting Furnace", 8, 2.3, 180),
        ("Demister", "Acid Plant", 6, 1.4, 230), ("Rectifier", "Refinery (Tankhouse)", 6, 1.1, 300)]
start, end = pd.Timestamp("2026-01-01"), pd.Timestamp("2026-09-30")
rows = []; k = 1
for jenis, unit, cnt, shape, scale in spec:
    for _ in range(cnt):
        umur0 = int(rng.integers(0, 120))                # umur saat jendela pengamatan dimulai (left truncation)
        # sampel umur gagal bersyarat > umur0
        while True:
            t = scale*rng.weibull(shape)
            if t > umur0: break
        horizon = (end - start).days + 1
        gagal = int(t - umur0 <= horizon)
        dur = min(t - umur0, horizon)
        rows.append([f"KMP-{k:03d}", unit, jenis, umur0, round(dur, 1), gagal,
                     round(float(rng.normal(85, 6)), 1), rng.choice(["Rendah", "Sedang", "Tinggi"], p=[.3, .45, .25])]); k += 1
komp = pd.DataFrame(rows, columns=["ID_Komponen", "Unit", "Jenis", "Umur_Awal_Hari", "Durasi_Pengamatan_Hari",
                                   "Gagal_1_Sensor_0", "Beban_Rata2_Persen", "Kelas_Lingkungan"])

kunci = pd.DataFrame({
 "Sheet": ["Data_Harian", "Downtime_Log", "QC_Lab", "Proses_Harian", "Produksi_Shift", "Harga_Biaya", "Komponen_Kritis"],
 "Status": ["ASLI", "ASLI", "ASLI", "SIMULASI", "SIMULASI", "SIMULASI", "SIMULASI"],
 "Baris": [len(d), len(log), len(qc), len(proses), len(shift), len(harga), len(komp)],
 "Catatan": [
  "Harian 1 Jan-30 Sep 2026 (tidak diubah)", "Kejadian downtime (tidak diubah)", "Batch QC (tidak diubah)",
  "Sensor proses harian. Dibangkitkan berkorelasi dengan Energi/t dan Recovery agar regresi bermakna (seed 2026).",
  "3 shift/hari. Jumlah Katoda, Energi, Jam_Downtime per hari = Data_Harian PERSIS (dicek).",
  "LME, tarif, kurs: random walk sintetis untuk Monte Carlo (Fase 2).",
  "Armada komponen simulasi (Weibull). Tidak 1:1 dengan Downtime_Log. Durasi mulai dari awal jendela; Umur_Awal_Hari = left truncation."]})
with pd.ExcelWriter(OUT, engine="xlsxwriter", datetime_format="yyyy-mm-dd") as w:
    for name, df in [("Kunci_Simulasi", kunci), ("Data_Harian", d), ("Downtime_Log", log), ("QC_Lab", qc),
                     ("Proses_Harian", proses), ("Produksi_Shift", shift), ("Harga_Biaya", harga), ("Komponen_Kritis", komp)]:
        df.to_excel(w, sheet_name=name, index=False)
        ws = w.sheets[name]; ws.freeze_panes(1, 0)
        for j, c in enumerate(df.columns): ws.set_column(j, j, max(12, min(40, len(str(c))+2)))
print("OK", {k: len(v) for k, v in [("proses", proses), ("shift", shift), ("harga", harga), ("komp", komp)]}, "gagal", komp.Gagal_1_Sensor_0.sum())
