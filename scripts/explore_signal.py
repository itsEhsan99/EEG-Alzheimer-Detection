"""Quick script to load one recording and plot its raw signal."""

import sys
from pathlib import Path

# Make the src/ package importable when running this script directly
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data_loader import load_recording


def main():
    # Adjust this path to point at one of your .set files
    recording_path = Path("DATA") / "AD" / "sub-01_task-eyesclosed_eeg.set"

    raw = load_recording(recording_path)

    # Print a quick summary so we know what we loaded
    print(raw.info)
    print(f"\nSampling rate: {raw.info['sfreq']} Hz")
    print(f"Number of channels: {len(raw.ch_names)}")
    print(f"Channel names: {raw.ch_names}")
    print(f"Duration: {raw.times[-1]:.1f} seconds")

    # Plot the raw signal (opens an interactive MNE window)
    raw.plot(block=True)


if __name__ == "__main__":
    main()