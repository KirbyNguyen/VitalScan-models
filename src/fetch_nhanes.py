"""
Assemble an NHANES diabetes dataset for the NN study (NN-only; not used by the served models).

Pools 3 pre-pandemic cycles (2013-14, 2015-16, 2017-18). Per cycle it downloads and merges on
the respondent id SEQN:
  DEMO  age (RIDAGEYR), sex (RIAGENDR)
  BMX   BMI (BMXBMI)
  BPX   systolic/diastolic (BPXSY1/BPXDI1)
  GHB   HbA1c (LBXGH)            -- used for the label only
  DIQ   doctor-told diabetes (DIQ010)

Features (glucose-free, vitals + demographics): age, sex, bmi, systolic, diastolic
Label: diabetes = 1 if HbA1c >= 6.5 OR DIQ010 == 1 (diagnosed or undiagnosed), adults 18+.

Run:  .venv/bin/python src/fetch_nhanes.py  ->  data/extra/nhanes_diabetes.csv
"""
from pathlib import Path
import warnings
import pandas as pd
import urllib.request

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "extra" / "nhanes_raw"
RAW.mkdir(parents=True, exist_ok=True)
CYCLES = [("2013", "H"), ("2015", "I"), ("2017", "J")]
FILES = ["DEMO", "BMX", "BPX", "GHB", "DIQ"]
BASE = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{year}/DataFiles/{name}_{suf}.xpt"

def fetch(year, suf, name):
    dest = RAW / f"{name}_{suf}.xpt"
    if not dest.exists():
        url = BASE.format(year=year, name=name, suf=suf)
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as r, open(dest, "wb") as f:
            f.write(r.read())
    return pd.read_sas(dest, format="xport")

def main():
    frames = []
    for year, suf in CYCLES:
        d = {n: fetch(year, suf, n) for n in FILES}
        m = d["DEMO"][["SEQN", "RIDAGEYR", "RIAGENDR"]]
        m = m.merge(d["BMX"][["SEQN", "BMXBMI"]], on="SEQN", how="inner")
        m = m.merge(d["BPX"][["SEQN", "BPXSY1", "BPXDI1"]], on="SEQN", how="inner")
        m = m.merge(d["GHB"][["SEQN", "LBXGH"]], on="SEQN", how="left")
        m = m.merge(d["DIQ"][["SEQN", "DIQ010"]], on="SEQN", how="left")
        m["cycle"] = f"{year}-{int(year)+1}"
        frames.append(m)
        print(f"  cycle {year}: {len(m)} merged rows")

    df = pd.concat(frames, ignore_index=True)
    out = pd.DataFrame({
        "age": df.RIDAGEYR,
        "sex": (df.RIAGENDR == 1).astype(int),          # 1 = male
        "bmi": df.BMXBMI,
        "systolic": df.BPXSY1,
        "diastolic": df.BPXDI1,
    })
    # label: HbA1c >= 6.5 (diagnostic) OR self-reported diagnosis
    out["diabetes"] = (((df.LBXGH >= 6.5) | (df.DIQ010 == 1))).astype(int)
    # adults with complete features and a determinable label
    out = out[df.RIDAGEYR >= 18]
    out = out[~(df.LBXGH.isna() & df.DIQ010.isna())]    # label must be knowable
    out = out.dropna(subset=["age", "sex", "bmi", "systolic", "diastolic"]).reset_index(drop=True)
    out.to_csv(ROOT / "data" / "extra" / "nhanes_diabetes.csv", index=False)
    print(f"\nnhanes_diabetes.csv: {len(out)} rows, {out.diabetes.mean()*100:.1f}% diabetic, "
          f"features={list(out.columns[:-1])}")

if __name__ == "__main__":
    main()
