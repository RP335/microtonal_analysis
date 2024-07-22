import numpy as np
import audioflux as af
import matplotlib.pyplot as plt
from audioflux.display import fill_wave

# Read audio data
audio_path = 'raga2.wav'  # Replace with your audio file path
audio_arr, sr = af.read(audio_path)

# Initialize PitchCEP object
pitch_obj_cep = af.PitchCEP(samplate=sr, low_fre=20, high_fre=4000, radix2_exp=12, slide_length=256)

# Initialize PitchYIN object
pitch_obj_yin = af.PitchYIN(samplate=sr, low_fre=50, high_fre=1000, slide_length=4096)

# Compute pitch using CEP
fre_arr_cep = pitch_obj_cep.pitch(audio_arr)

# Compute pitch using YIN
fre_arr_yin, _, _ = pitch_obj_yin.pitch(audio_arr)

# Calculate time arrays
times_cep = np.arange(fre_arr_cep.shape[-1]) * (pitch_obj_cep.slide_length / sr)
times_yin = np.arange(fre_arr_yin.shape[-1]) * (pitch_obj_yin.slide_length / sr)

# Plot results
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

# Plot waveform
fill_wave(audio_arr, samplate=sr, axes=ax1)
ax1.set_title('Waveform')

# Plot pitch (CEP)
scatter_cep = ax2.scatter(times_cep, fre_arr_cep, c=fre_arr_cep, cmap='viridis', s=2)
ax2.set_ylabel('Frequency (Hz)')
ax2.set_title('Pitch Estimation (CEP)')
plt.colorbar(scatter_cep, ax=ax2, label='Frequency (Hz)')

# Plot pitch (YIN)
scatter_yin = ax3.scatter(times_yin, fre_arr_yin, c=fre_arr_yin, cmap='viridis', s=2)
ax3.set_ylabel('Frequency (Hz)')
ax3.set_xlabel('Time (s)')
ax3.set_title('Pitch Estimation (YIN)')
plt.colorbar(scatter_yin, ax=ax3, label='Frequency (Hz)')

plt.tight_layout()
plt.show()

# Generate equal temperament scale
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
    valid_freq_mask = fre_arr > 20
    fre_arr = fre_arr[valid_freq_mask]

    deviations = []
    closest_freqs = []

    for freq in fre_arr:
        closest_freq, deviation = freq_to_closest_cents(freq, scale_frequencies)
        closest_freqs.append(closest_freq)
        deviations.append(deviation)

    return np.array(deviations), valid_freq_mask


# Analyze deviations using YIN results
deviations_yin, valid_freq_mask = analyze_microtonal_deviations(fre_arr_yin, equal_temperament_scale)

# Adjust times_yin to match the valid frequencies
times_yin_filtered = times_yin[valid_freq_mask]

# Print statistics
print("YIN Analysis Results:")
print(f"Average deviation: {np.mean(np.abs(deviations_yin)):.2f} cents")
print(f"Maximum deviation: {np.max(np.abs(deviations_yin)):.2f} cents")
print(f"Minimum deviation: {np.min(np.abs(deviations_yin)):.2f} cents")

# Plot histogram of deviations (YIN)
plt.figure(figsize=(12, 6))
plt.hist(deviations_yin, bins=100, edgecolor='black')
plt.title('Distribution of Deviations from 12 tone Scale (YIN)')
plt.xlabel('Deviation (cents)')
plt.ylabel('Frequency')
plt.axvline(x=0, color='r', linestyle='--')  # Add a line at 0 cents
plt.show()

# Plot deviations over time (YIN)
plt.figure(figsize=(12, 6))
plt.scatter(times_yin_filtered, deviations_yin, c=deviations_yin, cmap='coolwarm', s=2)
plt.colorbar(label='Deviation (cents)')
plt.title('Deviations from 12 tone Scale Over Time (YIN)')
plt.xlabel('Time (s)')
plt.ylabel('Deviation (cents)')
plt.ylim(-100, 100)  # Adjust this range as needed
plt.show()