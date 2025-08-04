import pandas as pd
import numpy as np
from stingray.pulse.pulsar import fold_events, htest
from scipy.stats import norm
from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os
import sys

# === Parameters ===
if len(sys.argv) < 2:
    print("Usage: python plot_selected_segments_with_mjd.py <csv_file>")
    sys.exit(1)

csv_file = sys.argv[1]
spin_file = "spin.csv"  # Must exist in the same folder

f1 = -1e-13
nbins = 64  # 2 pulse cycles

# === Load CSVs ===
try:
    df = pd.read_csv(csv_file)
    spin_df = pd.read_csv(spin_file)
except Exception as e:
    print(f"[✗] Failed to read input file(s): {e}")
    sys.exit(1)

# Columns to store MJDs
df["Segment Start (MJD)"] = np.nan
df["Segment End (MJD)"] = np.nan

# === Prepare output PDF ===
output_pdf = "pulse_profiles_64.pdf"
pdf = PdfPages(output_pdf)

# === Process each row ===
for idx, row in df.iterrows():
    obsid = str(row["ObsID"])
    event_file = row["Event File"]
    seg_start = float(row["Segment Start (s)"])
    seg_stop = float(row["Segment End (s)"])
    
    evt_path = os.path.join(obsid, "pca", event_file)

    if not os.path.isfile(evt_path):
        print(f"[✗] File not found: {evt_path}")
        continue

    # === Match Frequency f0 from spin.csv ===
    spin_match = spin_df[
        (spin_df["Event File"] == event_file) &
        (np.isclose(spin_df["Segment Start (s)"], seg_start, atol=0.1)) &
        (np.isclose(spin_df["Segment End (s)"], seg_stop, atol=0.1))
    ]

    if spin_match.empty:
        print(f"[!] f0 not found in spin.csv for segment {seg_start}–{seg_stop} in {event_file}")
        continue

    f0 = spin_match["Best Frequency (Hz)"].values[0]

    try:
        with fits.open(evt_path) as hdul:
            hdr = hdul[0].header
            times = hdul[1].data["TIME"]

            if "MJDREF" in hdr:
                ref_mjd = hdr["MJDREF"]
            elif "MJDREFI" in hdr and "MJDREFF" in hdr:
                ref_mjd = hdr["MJDREFI"] + hdr["MJDREFF"]
            else:
                raise ValueError("Reference MJD not found in header")

            ref_time_sec = ref_mjd * 86400.0

            # Convert segment start/stop to MJD and store
            start_mjd = ref_mjd + (seg_start / 86400.0)
            stop_mjd = ref_mjd + (seg_stop / 86400.0)
            df.at[idx, "Segment Start (MJD)"] = start_mjd
            df.at[idx, "Segment End (MJD)"] = stop_mjd

            # Select segment
            seg_mask = (times >= seg_start) & (times < seg_stop)
            seg_times = times[seg_mask]

            if len(seg_times) < 10:
                print(f"[SKIP] Too few events in segment ({len(seg_times)}): {obsid}")
                continue

            phase_bins, profile, profile_err = fold_events(
                seg_times, f0, f1, nbin=nbins, ref_time=ref_time_sec,
                expocorr=True, mode="ef"
            )

            phase_bins_extended = np.concatenate([phase_bins, phase_bins + 1])
            profile_extended = np.concatenate([profile, profile])
            profile_err_extended = np.concatenate([profile_err, profile_err])

            M, H = htest(profile, datatype="binned")
            fap = np.exp(-H / 2)
            sigma = norm.isf(fap)

            # Plot
            plt.figure(figsize=(6, 4))
            plt.errorbar(phase_bins_extended, profile_extended, yerr=profile_err_extended, fmt='o-')
            plt.title(f"{obsid} | Seg: {int(seg_start)}–{int(seg_stop)}s\nH={H:.2f}, M={M}, ~{sigma:.2f}σ")
            plt.xlabel("Pulse Phase (2 cycles)")
            plt.ylabel("Counts/bin")
            plt.grid(True)
            plt.tight_layout()
            pdf.savefig()
            plt.close()
            print(f"[✓] Plotted: {obsid} segment {int(seg_start)}–{int(seg_stop)}s")

    except Exception as e:
        print(f"[✗] Error in {evt_path}: {e}")

pdf.close()
print(f"\n[✓] All selected pulse profiles saved to: {output_pdf}")

# === Save updated CSV ===
updated_csv_path = csv_file.replace(".csv", "_with_mjd.csv")
df.to_csv(updated_csv_path, index=False)
print(f"[✓] Updated CSV with MJDs saved to: {updated_csv_path}")

