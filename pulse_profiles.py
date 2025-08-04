import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from stingray.pulse.pulsar import fold_events, htest
from scipy.stats import norm
from astropy.io import fits
import os
import sys
import csv
import glob

# ========== Input Check ==========
if len(sys.argv) < 2:
    print("[✗] No input event file provided.")
    sys.exit(1)

event_file = sys.argv[1]
evt_filename = os.path.basename(event_file)
evt_dir = os.path.dirname(event_file)

f1 = -1e-13
nbins = 64
segment_length = 512.0

# ========== Load spin.csv ==========
try:
    spin_df = pd.read_csv("spin.csv")
except Exception as e:
    print("[✗] Could not load spin.csv:", e)
    sys.exit(1)

# ========== Load FITS File ==========
with fits.open(event_file) as hdul:
    hdr = hdul[0].header
    times = hdul[1].data["TIME"]

    if "MJDREF" in hdr:
        ref_mjd = hdr["MJDREF"]
    elif "MJDREFI" in hdr and "MJDREFF" in hdr:
        ref_mjd = hdr["MJDREFI"] + hdr["MJDREFF"]
    else:
        raise ValueError("Reference MJD not found in header")

    ref_time_sec = ref_mjd * 86400.0

    if "GTI" in [h.name for h in hdul]:
        gti_data = hdul["GTI"].data
        gti_list = np.array([[row["START"], row["STOP"]] for row in gti_data])
    else:
        gti_list = np.array([[times.min(), times.max()]])

# ========== Output Paths ==========
base_name = os.path.splitext(evt_filename)[0]
output_pdf = os.path.join(evt_dir, f"{base_name}_pulse_profiles.pdf")
segment_csv = os.path.join(evt_dir, f"{base_name}_segments_sigma.csv")
combined_csv = os.path.abspath("high_significance_all.csv")

# ========== Loop Over Segments ==========
sorted_gti = gti_list[np.argsort(gti_list[:, 0])]
segment_start = sorted_gti[0, 0]
i = 0

segment_rows = []
high_sigma_rows = []

with PdfPages(output_pdf) as pdf:
    while i < len(sorted_gti):
        current_segment_times = []
        segment_end = segment_start + segment_length

        while i < len(sorted_gti) and sorted_gti[i][0] < segment_end:
            gti_start, gti_stop = sorted_gti[i]
            overlap_start = max(segment_start, gti_start)
            overlap_end = min(segment_end, gti_stop)
            if overlap_start < overlap_end:
                current_segment_times.append([overlap_start, overlap_end])
            if gti_stop >= segment_end:
                break
            i += 1

        if not current_segment_times:
            segment_start = sorted_gti[i][0] if i < len(sorted_gti) else segment_end
            continue

        mask = np.zeros_like(times, dtype=bool)
        for t0, t1 in current_segment_times:
            mask |= (times >= t0) & (times < t1)

        seg_times = times[mask]
        if len(seg_times) < 10:
            segment_start = segment_end
            continue

        # ==== Lookup f0 from spin.csv ====
        match_row = spin_df[
            (spin_df["Event File"] == evt_filename) &
            (np.isclose(spin_df["Segment Start (s)"], segment_start, atol=0.1)) &
            (np.isclose(spin_df["Segment End (s)"], segment_end, atol=0.1))
        ]

        if match_row.empty:
            print(f"[!] No frequency found in spin.csv for segment {segment_start:.0f}-{segment_end:.0f}. Skipping.")
            segment_start = segment_end
            continue

        f0 = match_row["Best Frequency (Hz)"].values[0]

        # ==== Fold and Evaluate ====
        phase_bins, profile, profile_err = fold_events(
            seg_times, f0, f1, nbin=nbins,
            ref_time=ref_time_sec, expocorr=True, mode="ef"
        )

        x = np.concatenate([phase_bins, phase_bins + 1])
        y = np.concatenate([profile, profile])
        yerr = np.concatenate([profile_err, profile_err])

        M, H = htest(profile, datatype="binned")
        fap = np.exp(-H / 2)
        sigma = norm.isf(fap)

        # ==== MJDs ====
        seg_start_mjd = ref_mjd + segment_start / 86400.0
        seg_end_mjd = ref_mjd + segment_end / 86400.0

        # ==== Plot Segment ====
        plt.figure(figsize=(6, 4))
        plt.errorbar(x, y, yerr=yerr, fmt='o-')
        plt.title(f"{segment_start:.0f}-{segment_end:.0f}s | H = {H:.2f}, M = {M}, ~{sigma:.2f}σ")
        plt.xlabel("Pulse Phase (2 cycles)")
        plt.ylabel("Counts/bin")
        plt.grid(True)
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        # ==== Store segment info ====
        segment_rows.append([
            segment_start, segment_end,
            round(seg_start_mjd, 6), round(seg_end_mjd, 6),
            len(seg_times), round(f0, 6), round(sigma, 4)
        ])

        # ==== Add to high-sigma list ====
        if sigma >= 3.0:
            high_sigma_rows.append([
                "",  # ObsID will be updated later
                evt_filename,
                segment_start, segment_end,
                round(seg_start_mjd, 6), round(seg_end_mjd, 6),
                round(f0, 6), round(sigma, 4)
            ])

        segment_start = segment_end

# ========== Save Segment CSV ==========
with open(segment_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "Segment Start (s)", "Segment End (s)",
        "Segment Start MJD", "Segment End MJD",
        "Event Count", "Best Frequency (Hz)", "H-test Sigma"
    ])
    writer.writerows(segment_rows)

# ========== Save High-Sigma CSV ==========
if high_sigma_rows:
    write_header = not os.path.exists(combined_csv)
    with open(combined_csv, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "ObsID", "Event File",
                "Segment Start (s)", "Segment End (s)",
                "Segment Start MJD", "Segment End MJD",
                "Best Frequency (Hz)", "H-test Sigma"
            ])
        writer.writerows(high_sigma_rows)

# ========== Update ObsID in Combined CSV ==========
try:
    df = pd.read_csv(combined_csv)
except Exception as e:
    print(f"[✗] Could not read combined CSV: {e}")
    sys.exit(1)

event_to_obsid = {}
for root in glob.glob("[0-9]*-[0-9]*-[0-9]*-[0-9]*"):
    pca_dir = os.path.join(root, "pca")
    if not os.path.isdir(pca_dir):
        continue
    for fname in os.listdir(pca_dir):
        if fname.endswith(".evt"):
            event_to_obsid[fname] = root

missing_files = 0
for idx, row in df.iterrows():
    evt_file = row["Event File"]
    if evt_file in event_to_obsid:
        df.at[idx, "ObsID"] = event_to_obsid[evt_file]
    else:
        missing_files += 1

df.to_csv(combined_csv, index=False)

# ========== Final Output ==========
print(f"[✓] Pulse profiles saved to PDF: {output_pdf}")
print(f"[✓] Segment-level CSV saved: {segment_csv}")
if high_sigma_rows:
    print(f"[✓] High-significance segments appended and updated in: {combined_csv}")
    if missing_files:
        print(f"[!] {missing_files} entries had no matching ObsID")
else:
    print("[i] No segments exceeded 3σ")

