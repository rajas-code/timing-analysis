import numpy as np
from astropy.io import fits
from stingray.pulse.search import z_n_search
import sys
import os
import csv

# ------------ Input Event File ------------
if len(sys.argv) < 2:
    print("Usage: python zn_spin_append.py <event_file.evt>")
    sys.exit(1)

event_file = sys.argv[1]
f0_guess = 400.975
df = 0.001
n_freqs = 201
segment_length = 512.0
nbin = 64
nharm = 2

# ------------ Load Event Data ------------
with fits.open(event_file) as hdul:
    times = hdul[1].data["TIME"]
    hdr = hdul[0].header

    # MJDREF
    if "MJDREF" in hdr:
        ref_mjd = hdr["MJDREF"]
    elif "MJDREFI" in hdr and "MJDREFF" in hdr:
        ref_mjd = hdr["MJDREFI"] + hdr["MJDREFF"]
    else:
        raise ValueError("MJDREF not found in header")

    # GTIs
    if "GTI" in [h.name for h in hdul]:
        gti_data = hdul["GTI"].data
        gti_list = np.array([[row["START"], row["STOP"]] for row in gti_data])
    else:
        gti_list = np.array([[times.min(), times.max()]])

# ------------ Frequency Grid ------------
freqs = np.linspace(f0_guess - df * (n_freqs // 2), f0_guess + df * (n_freqs // 2), n_freqs)

# ------------ Run Z^2_n Search ------------
results = []
for gti_start, gti_stop in gti_list:
    current = gti_start
    while current + segment_length <= gti_stop:
        seg_mask = (times >= current) & (times < current + segment_length)
        seg_times = times[seg_mask]

        if len(seg_times) < 20:
            current += segment_length
            continue

        fgrid, stats = z_n_search(
            seg_times,
            frequencies=freqs,
            nharm=nharm,
            nbin=nbin,
            segment_size=np.inf,
            expocorr=False,
            gti=[[current, current + segment_length]],
        )

        best_index = np.argmax(stats)
        best_freq = fgrid[best_index]
        best_stat = stats[best_index]

        # Estimate frequency error from parabola fit
        if 0 < best_index < len(stats) - 1:
            f1, f2, f3 = fgrid[best_index - 1: best_index + 2]
            z1, z2, z3 = stats[best_index - 1: best_index + 2]
            coeffs = np.polyfit([f1, f2, f3], [z1, z2, z3], 2)
            if coeffs[0] < 0:
                freq_err = np.sqrt(-1 / (2 * coeffs[0]))
            else:
                freq_err = np.nan
        else:
            freq_err = np.nan

        seg_start_mjd = ref_mjd + (current / 86400.0)
        seg_end_mjd = ref_mjd + ((current + segment_length) / 86400.0)

        print(f"[Segment {current:.0f} – {current + segment_length:.0f}s] "
              f"Best freq: {best_freq:.6f} Hz, Z^2_{nharm} = {best_stat:.2f}, "
              f"Freq error: {freq_err:.6g} Hz")

        results.append([
            os.path.basename(event_file),
            current,
            current + segment_length,
            seg_start_mjd,
            seg_end_mjd,
            best_freq,
            best_stat,
            freq_err
        ])
        current += segment_length

# ------------ Append or Create spin.csv ------------
csv_path = "spin.csv"
header = [
    "Event File",
    "Segment Start (s)",
    "Segment End (s)",
    "Segment Start MJD",
    "Segment End MJD",
    "Best Frequency (Hz)",
    f"Z^2_{nharm}",
    "Freq Error (Hz)"
]
write_header = not os.path.exists(csv_path)

with open(csv_path, "a", newline="") as f:
    writer = csv.writer(f)
    if write_header:
        writer.writerow(header)
    writer.writerows(results)

print(f"\n✅ Appended {len(results)} segment(s) with frequency errors to {csv_path}")
