"""Mengubah notebooks_src/*.py (format '# %%') menjadi notebooks/*.ipynb DENGAN keluaran (teks + grafik) agar terbaca di GitHub.

Jalankan dari akar repositori:
    python scripts/build_notebooks.py            # bangun semua, simpan ipynb + reports/figures
    python scripts/build_notebooks.py --check    # hanya jalankan untuk uji (tidak menulis berkas)
    python scripts/build_notebooks.py notebooks_src/05_SPC.py   # satu modul saja
"""
import base64, contextlib, glob, io, json, os, sys, traceback, warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECK = "--check" in sys.argv
ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]


def parse(path):
    cells, cur, kind = [], [], None
    for line in open(path, encoding="utf-8").read().split("\n"):
        if line.startswith("# %%"):
            if kind:
                cells.append((kind, "\n".join(cur).strip("\n")))
            kind = "markdown" if "[markdown]" in line else "code"
            cur = []
        else:
            cur.append(line)
    if kind:
        cells.append((kind, "\n".join(cur).strip("\n")))
    out = []
    for k, s in cells:
        if k == "markdown":
            s = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in s.split("\n"))
        out.append((k, s))
    return out


def run_and_build(cells, name, fig_dir):
    ns, nfig, count = {"__name__": "__main__"}, 0, 0
    nb = {"nbformat": 4, "nbformat_minor": 5,
          "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                       "language_info": {"name": "python"}, "colab": {"provenance": []}},
          "cells": []}
    for i, (k, s) in enumerate(cells):
        cell = {"cell_type": k, "id": f"c{i:03d}", "metadata": {}, "source": s.splitlines(True)}
        if k == "code":
            count += 1
            buf, outs = io.StringIO(), []
            try:
                with contextlib.redirect_stdout(buf), warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    exec(compile(s, f"{name}[sel {i}]", "exec"), ns)
            except Exception:
                print(f"!! GALAT di {name}, sel {i}:\n{s[:300]}\n")
                traceback.print_exc()
                return None
            if buf.getvalue():
                outs.append({"output_type": "stream", "name": "stdout", "text": buf.getvalue().splitlines(True)})
            for n in plt.get_fignums():
                fig = plt.figure(n)
                png = io.BytesIO()
                fig.savefig(png, format="png", dpi=100, bbox_inches="tight")
                if not CHECK:
                    open(os.path.join(fig_dir, f"{name}_{nfig}.png"), "wb").write(png.getvalue())
                outs.append({"output_type": "display_data", "metadata": {},
                             "data": {"image/png": base64.b64encode(png.getvalue()).decode(), "text/plain": ["<Figure>"]}})
                nfig += 1
            plt.close("all")
            cell.update(outputs=outs, execution_count=count)
        nb["cells"].append(cell)
    return nb


def main():
    os.chdir(os.path.join(ROOT, "notebooks"))          # sama seperti saat notebook dibuka dari folder notebooks/
    srcs = [os.path.join(ROOT, a) if not os.path.isabs(a) else a for a in ARGS] or sorted(glob.glob(os.path.join(ROOT, "notebooks_src", "*.py")))
    fig_dir = os.path.join(ROOT, "reports", "figures")
    os.makedirs(fig_dir, exist_ok=True)
    gagal = 0
    for p in srcs:
        name = os.path.basename(p)[:-3]
        nb = run_and_build(parse(p), name, fig_dir)
        if nb is None:
            gagal += 1
            continue
        if not CHECK:
            json.dump(nb, open(os.path.join(ROOT, "notebooks", name + ".ipynb"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"OK  {name}: {len(nb['cells'])} sel")
    sys.exit(1 if gagal else 0)


if __name__ == "__main__":
    main()
