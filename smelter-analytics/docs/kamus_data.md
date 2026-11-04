# Kamus Data

Berkas utama: `data/Smelter_Data_Lanjutan.xlsx` (8 sheet). Tiga berkas asli juga tersedia sebagai CSV di `data/raw/`.

> **Penting:** seluruh data adalah *skenario latihan*, bukan data operasional perusahaan nyata. Sheet berstatus SIMULASI dibangkitkan oleh `scripts/generate_data.py` (seed 2026) dan dapat direproduksi persis.

## Status sheet

| Sheet | Status | Baris | Catatan |
| --- | --- | --- | --- |
| Data_Harian | ASLI | 273 | Harian 1 Jan-30 Sep 2026 (tidak diubah) |
| Downtime_Log | ASLI | 73 | Kejadian downtime (tidak diubah) |
| QC_Lab | ASLI | 273 | Batch QC (tidak diubah) |
| Proses_Harian | SIMULASI | 273 | Sensor proses harian. Dibangkitkan berkorelasi dengan Energi/t dan Recovery agar regresi bermakna (seed 2026). |
| Produksi_Shift | SIMULASI | 819 | 3 shift/hari. Jumlah Katoda, Energi, Jam_Downtime per hari = Data_Harian PERSIS (dicek). |
| Harga_Biaya | SIMULASI | 273 | LME, tarif, kurs: random walk sintetis untuk Monte Carlo (Fase 2). |
| Komponen_Kritis | SIMULASI | 44 | Armada komponen simulasi (Weibull). Tidak 1:1 dengan Downtime_Log. Durasi mulai dari awal jendela; Umur_Awal_Hari = left truncation. |

## Kolom per sheet

### Data_Harian (ASLI, 273 baris)

| Kolom | Tipe | Contoh |
| --- | --- | --- |
| `Tanggal` | datetime64[us] | 2026-01-01 |
| `Bulan` | int64 | 1 |
| `Konsentrat_Masuk_DMT` | int64 | 2471 |
| `Cu_Grade_Persen` | float64 | 25.79 |
| `Au_g_per_DMT` | float64 | 22.0 |
| `Jam_Downtime` | float64 | 0.0 |
| `Jam_Operasi` | float64 | 24.0 |
| `Katoda_Cu_t` | float64 | 629.8 |
| `Emas_kg` | float64 | 52.7 |
| `Asam_Sulfat_t` | int64 | 2176 |
| `Slag_t` | int64 | 1696 |
| `Energi_MWh` | int64 | 964 |
| `Cu_Masuk_t` | float64 | 637.2709 |
| `Recovery_Persen` | float64 | 98.8276728154385 |
| `Availability_Persen` | float64 | 100.0 |
| `Energi_per_t_Katoda_MWh` | float64 | 1.53064464909495 |
| `Rasio_Asam_per_Konsentrat` | float64 | 0.880615135572643 |

### Downtime_Log (ASLI, 73 baris)

| Kolom | Tipe | Contoh |
| --- | --- | --- |
| `ID_Kejadian` | str | DT-001 |
| `Tanggal` | datetime64[us] | 2026-01-04 |
| `Unit` | str | Acid Plant |
| `Kategori` | str | Tidak Terjadwal |
| `Penyebab` | str | Gangguan demister |
| `Durasi_Jam` | float64 | 3.0 |

### QC_Lab (ASLI, 273 baris)

| Kolom | Tipe | Contoh |
| --- | --- | --- |
| `Batch_ID` | str | KT-20260101-01 |
| `Tanggal` | datetime64[us] | 2026-01-01 |
| `Cu_Persen` | float64 | 99.9942 |
| `Pb_ppm` | float64 | 1.5 |
| `S_ppm` | float64 | 7.5 |
| `Fe_ppm` | float64 | 1.0 |
| `Status_Mutu` | str | Grade A |

### Proses_Harian (SIMULASI, 273 baris)

| Kolom | Tipe | Contoh |
| --- | --- | --- |
| `Tanggal` | datetime64[us] | 2026-01-01 |
| `Suhu_FSF_C` | float64 | 1271.7 |
| `O2_Enrichment_Persen` | float64 | 57.7 |
| `Matte_Grade_Persen` | float64 | 63.51 |
| `Cu_Slag_Persen` | float64 | 0.714 |
| `Kelembaban_Konsentrat_Persen` | float64 | 7.91 |
| `Suhu_Ambien_C` | float64 | 24.1 |
| `Curah_Hujan_mm` | float64 | 0.0 |

### Produksi_Shift (SIMULASI, 819 baris)

| Kolom | Tipe | Contoh |
| --- | --- | --- |
| `ID_Shift` | str | SH-0001 |
| `Tanggal` | datetime64[us] | 2026-01-01 |
| `Shift` | str | Pagi |
| `Regu` | str | A |
| `Jam_Operasi` | float64 | 8.0 |
| `Jam_Downtime` | float64 | 0.0 |
| `Katoda_Cu_t` | float64 | 219.1 |
| `Energi_MWh` | int64 | 314 |

### Harga_Biaya (SIMULASI, 273 baris)

| Kolom | Tipe | Contoh |
| --- | --- | --- |
| `Tanggal` | datetime64[us] | 2026-01-01 |
| `LME_Cu_USD_per_t` | int64 | 10319 |
| `Tarif_Listrik_USD_per_MWh` | float64 | 71.26 |
| `Kurs_IDR_per_USD` | int64 | 16211 |
| `Biaya_Energi_USD` | int64 | 68695 |
| `Nilai_Katoda_USD` | int64 | 6498906 |

### Komponen_Kritis (SIMULASI, 44 baris)

| Kolom | Tipe | Contoh |
| --- | --- | --- |
| `ID_Komponen` | str | KMP-001 |
| `Unit` | str | Flash Smelting Furnace |
| `Jenis` | str | Tuyere |
| `Umur_Awal_Hari` | int64 | 35 |
| `Durasi_Pengamatan_Hari` | float64 | 273.0 |
| `Gagal_1_Sensor_0` | int64 | 0 |
| `Beban_Rata2_Persen` | float64 | 88.7 |
| `Kelas_Lingkungan` | str | Sedang |

## Definisi KPI

- **Recovery (%)** = total Katoda_Cu_t ÷ total Cu_Masuk_t × 100 (rasio total, bukan rata-rata persen harian).
- **Availability (%)** = Jam_Operasi ÷ (24 × jumlah hari) × 100.
- **Energi per ton** = Energi_MWh ÷ Katoda_Cu_t (MWh/t katoda).
- **MTBF** = jam operasi ÷ jumlah kejadian tak terjadwal; **MTTR** = jam henti ÷ jumlah kejadian.
- **Target prorata** = 153.000 t ÷ 273 hari = 560,44 t/hari × jumlah hari bulan itu.
- **Angka contoh (bukan data):** batas spesifikasi SPC (USL energi/t 1,60; LSL recovery 97%), biaya program 6 juta USD, rasio durasi terencana/gagal 0,5, harga dan tarif di Harga_Biaya.
