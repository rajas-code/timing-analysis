import numpy as np
from astropy.io import fits
from stingray.pulse.search import z_n_search
import os
import sys
import csv
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# ---------------- Parameters ----------------
if len(sys.argv) < 2:
    print("Usage: python zn_spin_gaussian_plot.py <event_file.evt>")
    sys.exit(1)

event_file = sys.argv[1]
segment_length = 512.0
nbin = 64
nharm = 2
df = 0.001
n_freqs = 201

f0_fund = 400.975  # Fundamental frequency

fund_freqs = np.linspace(f0_fund - df * (n_freqs // 2),
                         f0_fund + df * (n_freqs // 2),
                         n_freqs)

# ---------------- Load Event Data ----------------
with fits.open(event_file) as hdul:
    times = hdul[1].data["TIME"]
    hdr = hdul[0].header

    if "MJDREF" in hdr:
        ref_mjd = hdr["MJDREF"]
    elif "MJDREFI" in hdr and "MJDREFF" in hdr:
        ref_mjd = hdr["MJDREFI"] + hdr["MJDREFF"]
    else:
        raise ValueError("MJDREF not found in header")

    if "GTI" in [h.name for h in hdul]:
        gti_data = hdul["GTI"].data
        gti_list = np.array([[row["START"], row["STOP"]] for row in gti_data])
    else:
        gti_list = np.array([[times.min(), times.max()]])

# ---------------- Gaussian Function ----------------
def gaussian(x, A, mu, sigma, C):
    return A * np.exp(-0.5 * ((x - mu) / sigma)**2) + C

# ---------------- Helper: Gaussian Fit Search ----------------
def search_and_fit_gaussian(seg_times, freqs, gti_seg, segment_start, segment_end):
    fgrid, stats = z_n_search(
        seg_times, frequencies=freqs, nharm=nharm, nbin=nbin,
        segment_size=np.inf, expocorr=False, gti=gti_seg)

    best_idx = np.argmax(stats)
    best_freq = fgrid[best_idx]
    best_stat = stats[best_idx]

    # Fit Gaussian to ±5 bins around peak
    idx_min = max(0, best_idx - 5)
    idx_max = min(len(fgrid), best_idx + 6)
    fit_freqs = fgrid[idx_min:idx_max]
    fit_stats = stats[idx_min:idx_max]

    try:
        p0 = [best_stat, best_freq, df, np.min(fit_stats)]
        popt, _ = curve_fit(gaussian, fit_freqs, fit_stats, p0=p0)
        A_fit, mu_fit, sigma_f, C_fit = popt
        freq_err = abs(sigma_f)

        # Plot and save
        plt.figure(figsize=(6, 4))
        plt.plot(fgrid, stats, "o-", label="Z² data")
        fine_freqs = np.linspace(fit_freqs.min(), fit_freqs.max(), 500)
        plt.plot(fine_freqs, gaussian(fine_freqs, *popt), "r-", lw=2, label="Gaussian fit")
        plt.axvline(mu_fit, color="g", linestyle="--", label=f"Best Fit: {mu_fit:.6f} Hz")
        plt.axvline(best_freq, color="b", linestyle=":", label=f"Max Z²: {best_freq:.6f} Hz")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel(f"Z² (n={nharm})")
        plt.title(f"Gaussian Fit for Segment {int(segment_start)}–{int(segment_end)} s")
        plt.legend()
        plt.grid(True)

        save_dir = os.path.dirname(os.path.abspath(event_file))
        save_path = os.path.join(save_dir, f"{os.path.splitext(os.path.basename(event_file))[0]}_{int(segment_start)}s_gaussian_fit.png")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"[✓] Saved Gaussian fit plot: {save_path}")

    except Exception as e:
        print(f"[!] Gaussian fit failed: {e}")
        freq_err = np.nan

    return best_freq, best_stat, freq_err

# ---------------- Process Segments ----------------
fund_results = []

for gti_start, gti_stop in gti_list:
    current = gti_start
    while current + segment_length <= gti_stop:
        seg_mask = (times >= current) & (times < current + segment_length)
        seg_times = times[seg_mask]

        if len(seg_times) < 20:
            current += segment_length
            continue

        gti_seg = [[current, current + segment_length]]
        seg_start_mjd = ref_mjd + current / 86400.0
        seg_end_mjd = ref_mjd + (current + segment_length) / 86400.0

        best_freq_f, best_stat_f, freq_err_f = search_and_fit_gaussian(seg_times, fund_freqs, gti_seg, current, current + segment_length)
        period_err_f = freq_err_f / (best_freq_f**2) if freq_err_f is not np.nan else np.nan

        fund_results.append([
            os.path.basename(event_file), current, current + segment_length,
            seg_start_mjd, seg_end_mjd, round(best_freq_f, 6),
            round(best_stat_f, 2), round(freq_err_f, 9), round(period_err_f, 9)
        ])

        current += segment_length

# ---------------- Save CSV ----------------
header = [
    "Event File", "Segment Start (s)", "Segment End (s)",
    "Segment Start MJD", "Segment End MJD",
    "Best Frequency (Hz)", f"Z^2_{nharm}",
    "Freq Error (Hz)", "Period Error (s)"
]

csv_fund = "spin_fundamental_gaussian.csv"

write_header = not os.path.exists(csv_fund)
with open(csv_fund, "a", newline="") as f:
    writer = csv.writer(f)
    if write_header:
        writer.writerow(header)
    writer.writerows(fund_results)

print(f"\n✅ Appended {len(fund_results)} segments to {csv_fund}")

