import numpy as np
import audioflux as af
import matplotlib.pyplot as plt
from audioflux.display import fill_wave

# Read audio data
audio_path = 'raga2.wav'  #
audio_arr, sr = af.read(audio_path)

pitch_obj = af.PitchCEP(samplate=sr,
                        low_fre=20,
                        high_fre=4000,
                        radix2_exp=12,
                        slide_length=256)

fre_arr = pitch_obj.pitch(audio_arr)

times = np.arange(fre_arr.shape[-1]) * (pitch_obj.slide_length / sr)

# Plot results
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Plot waveform
fill_wave(audio_arr, samplate=sr, axes=ax1)
ax1.set_title('Waveform')

# Plot pitch
scatter = ax2.scatter(times, fre_arr, c=fre_arr, cmap='viridis', s=2)
ax2.set_ylabel('Frequency (Hz)')
ax2.set_xlabel('Time (s)')
ax2.set_title('Pitch Estimation (CEP)')

# Add colorbar
plt.colorbar(scatter, ax=ax2, label='Frequency (Hz)')

# Adjust layout and display
plt.tight_layout()
plt.show()

midi_numbers = np.arange(0, 128)  # all notes on the keyboard

equal_temperament_scale = 440.8 * 2 ** ((midi_numbers - 69) / 12)


def freq_to_closest_cents(freq, ref_freq_array):
    closest_ref_freq = ref_freq_array[np.argmin(np.abs(ref_freq_array - freq))]
    cents_diff = 1200 * np.log2(freq / closest_ref_freq)

    if cents_diff > 600:
        cents_diff -= 1200
    elif cents_diff < -600:
        cents_diff += 1200

    return closest_ref_freq, cents_diff


def analyze_microtonal_deviations(fre_arr, scale_frequencies):
    # Remove zeros and very low frequencies
    fre_arr = fre_arr[fre_arr > 20]

    deviations = []
    closest_freqs = []

    for freq in fre_arr:
        closest_freq, deviation = freq_to_closest_cents(freq, scale_frequencies)
        closest_freqs.append(closest_freq)
        deviations.append(deviation)

    return np.array(deviations)


# Analyze deviations
deviations = analyze_microtonal_deviations(fre_arr, equal_temperament_scale)

# Print statistics
print(f"Average deviation: {np.mean(np.abs(deviations)):.2f} cents")
print(f"Maximum deviation: {np.max(np.abs(deviations)):.2f} cents")
print(f"Minimum deviation: {np.min(np.abs(deviations)):.2f} cents")

# Plot histogram of deviations
plt.figure(figsize=(12, 6))
plt.hist(deviations, bins=100, edgecolor='black')
plt.title('Distribution of Deviations from 12 toneScale')
plt.xlabel('Deviation (cents)')
plt.ylabel('Frequency')
plt.axvline(x=0, color='r', linestyle='--')  # Add a line at 0 cents
plt.show()

# Plot deviations over time
plt.figure(figsize=(12, 6))
plt.scatter(times, deviations, c=deviations, cmap='coolwarm', s=2)
plt.colorbar(label='Deviation (cents)')
plt.title('Deviations from 12 tone Scale Over Time')
plt.xlabel('Time (s)')
plt.ylabel('Deviation (cents)')
plt.ylim(-100, 100)  # Adjust this range as needed
plt.show()
