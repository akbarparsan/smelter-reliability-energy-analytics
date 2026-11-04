# Ringkasan Eksekutif: Insight, Rekomendasi, dan Kebijakan

Periode: Januari–September 2026. Dasar: Notebook 00–12. Data proses, shift, komponen, dan biaya bersifat simulasi/contoh (lihat [kamus data](kamus_data.md)).

## Pesan utama

Kerugian produksi terbesar datang dari henti tak terjadwal yang berulang dan terpusat di beberapa unit, bukan dari proses yang menurun.

1. **Kronis, bukan memburuk.** 177,5 jam henti (2,7% dari 6.552 jam); Laplace p = 0,484; Crow-AMSAA β = 1,10 (CI 0,84–1,40).
2. **Oxygen Plant titik terlemah.** 40,5 jam (23%), MTBF 592 jam. Trip kompresor udara = 23 dari 80,5 jam henti Juni–Juli (29%).
3. **Energi per ton bisa ditekan.** Suhu FSF −10 °C ≈ 5.003 MWh/tahun (0,37 juta USD); O2 +1 poin ≈ 3.386 MWh (0,25 juta USD); downtime −1 jam/hari ≈ 1.959 MWh (0,14 juta USD). Berupa asosiasi, perlu uji terkontrol.
4. **Penggantian preventif selektif.** Layak: Blower SO2. Marginal: kompresor udara, waste heat boiler tube. Tidak layak: tuyere, demister, rectifier.
5. **Tidak ada dasar menindak shift/regu** (p Holm 0,25–0,43).

## Rekomendasi

| # | Rekomendasi | Dasar | Pemilik | Waktu |
| --- | --- | --- | --- | --- |
| 1 | Program keandalan Oxygen Plant (akar masalah trip kompresor udara, proteksi, cadangan suku cadang kritis) | 40,5 jam henti; MTBF 592 jam | Utilitas/Reliability | Mulai 30 hari |
| 2 | Uji coba penggantian terjadwal Blower SO2 (210–270 hari) | β = 4,2; T optimal 273 hari | Perencana Pemeliharaan | 60 hari |
| 3 | Pemeliharaan berbasis kondisi untuk kompresor udara dan boiler tube | Penggantian di B10 rugi pada simulasi | Reliability | 90 hari |
| 4 | Uji terkontrol suhu FSF dan O2 enrichment dengan batas keselamatan | Dua variabel terpenting model energi | Kepala Proses | 60–120 hari |
| 5 | Pemeriksaan mutu tambahan 72 jam setelah restart | Off-grade 7,9% vs 2,5% (p = 0,062) | QA/Lab | 30 hari |
| 6 | Dashboard resmi memakai target prorata (560,44 t/hari) dan kartu SPC | Target flat menyesatkan | Data Analyst | 30 hari |

## Kebijakan yang diusulkan

- **Keandalan:** setiap henti tak terjadwal > 2 jam punya kode penyebab standar dan analisis akar masalah dalam 48 jam.
- **Penggantian:** interval ditetapkan per komponen dari β dan rasio biaya, direview tiap kuartal; komponen dengan β ≈ 1 tidak masuk penggantian umur tetap.
- **Perubahan SOP proses:** setpoint diubah hanya lewat uji terkontrol dengan persetujuan Kepala Proses dan HSE.
- **Pelaporan:** satu definisi target, satu sumber data, label jelas "nyata/simulasi/asumsi".
- **Evaluasi kinerja:** perbandingan shift/regu tidak dipakai menilai individu sebelum data nyata dan koreksi uji ganda mendukung.

## Rencana 30-60-90 hari

| Periode | Kegiatan |
| --- | --- |
| 1–30 | Tim keandalan Oxygen Plant, analisis akar masalah kompresor, kode penyebab standar, dashboard prorata, verifikasi hari anomali (5 Mar, 5 Jun, 8 Jun, 7 Ags) dengan logbook, cek mutu 72 jam |
| 31–60 | Jadwal penggantian Blower SO2, pemantauan kondisi kompresor/boiler tube, protokol uji suhu/O2, biaya nyata dari Finance |
| 61–90 | Jalankan uji, hitung ulang MTBF dan model energi, review kebijakan penggantian, putuskan lanjut atau hentikan |

## Jangan disimpulkan

- Regu atau shift tertentu "lebih buruk".
- Downtime *menyebabkan* off-grade (baru indikasi).
- Penghematan energi pasti terjadi (belum kausal).
- Nilai dolar sebagai kerugian laba (harga contoh, nilai kotor).
- Semua komponen perlu diganti terjadwal.
- Kondisi pabrik memburuk atau proses bergeser pada bulan tertentu.

## Peningkatan analisis ke depan

Data nyata (historian, CMMS, laporan shift), kode penyebab dan umur komponen tersensor, biaya resmi, periode 24–36 bulan; desain uji A/B dan koreksi uji ganda; model hierarkis, risiko berulang, Cox dengan kovariat nyata, prediksi sisa umur dari sensor; pipeline terjadwal ke Power BI dengan uji kualitas data otomatis dan pemantauan drift.
