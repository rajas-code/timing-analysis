#!/usr/bin/env python3
import os
import sys
import numpy as np
from astropy.io import fits
from astropy.time import Time

# -----------------------------
# Parameters
SEGMENT_LENGTH = 512.0  # segment length in seconds
# -----------------------------

if len(sys.argv) < 2:
    print("Usage: python3 segment_events.py <event_file.fits>")
    sys.exit(1)

event_file = sys.argv[1]
evt_dir = os.path.dirname(event_file)
evt_filename = os.path.basename(event_file)

# Create output directory
output_dir = os.path.join(evt_dir, "events")
os.makedirs(output_dir, exist_ok=True)

# Load event file
with fits.open(event_file) as hdul:
    hdr = hdul[0].header
    events = hdul[1].data
    times = events["TIME"]

    # Determine reference MJD
    if "MJDREF" in hdr:
        ref_mjd = hdr["MJDREF"]
    elif "MJDREFI" in hdr and "MJDREFF" in hdr:
        ref_mjd = hdr["MJDREFI"] + hdr["MJDREFF"]
    else:
        raise ValueError("MJD reference not found in header")

    # Load GTI if available
    if "GTI" in [h.name for h in hdul]:
        gti_data = hdul["GTI"].data
        gti_list = np.array([[row["START"], row["STOP"]] for row in gti_data])
    else:
        gti_list = np.array([[times.min(), times.max()]])

# Sort GTIs
gti_list = gti_list[np.argsort(gti_list[:, 0])]

# Segment loop
for gti_start, gti_stop in gti_list:
    segment_start = gti_start
    while segment_start + SEGMENT_LENGTH <= gti_stop:
        segment_end = segment_start + SEGMENT_LENGTH

        # Mask events in this segment
        mask = (times >= segment_start) & (times < segment_end)
        seg_times = times[mask]

        if len(seg_times) == 0:
            segment_start = segment_end
            continue

        # Create new FITS HDU
        new_events = events[mask]
        new_hdu = fits.BinTableHDU(new_events, header=hdul[1].header)

        # Update primary header
        new_hdr = hdul[0].header.copy()
        new_hdr["DATE-OBS"] = Time(ref_mjd + segment_start / 86400.0, format="mjd").isot
        new_hdr["DATE-END"] = Time(ref_mjd + segment_end / 86400.0, format="mjd").isot
        new_hdr["TSTART"] = segment_start
        new_hdr["TSTOP"] = segment_end
        new_hdu.header.update(hdul[1].header)

        new_hdul = fits.HDUList([fits.PrimaryHDU(header=new_hdr), new_hdu])

        # Output filename
        out_file = os.path.join(
            output_dir,
            f"{os.path.splitext(evt_filename)[0]}_seg_{int(segment_start)}_{int(segment_end)}.fits"
        )

        new_hdul.writeto(out_file, overwrite=True)
        print(f"[+] Saved segment: {out_file}")

        # Move to next segment
        segment_start = segment_end

print("[✓] All segments created successfully!")
