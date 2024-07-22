import numpy as np
import audioflux as af
import matplotlib.pyplot as plt

audio_path = 'raga2.wav'
audio_arr, sr = af.read(audio_path)

pitch_obj = af.PitchYIN(samplate=sr, low_fre=20, high_fre=2000)

fre_arr, _, _ = pitch_obj.pitch(audio_arr)

fre_arr = fre_arr[fre_arr > 20]

# Generate standard 12 tone equal temperament frequencies (A-440 Hz scale)
base_c_sharp = 440 * 2**(4/12)
c_sharp_freqs_standard = base_c_sharp * np.power(2.0, np.arange(-5, 6))  # Covers 11 octaves centered around the base C#

def find_closest_c_sharp(freq):
    index = np.argmin(np.abs(c_sharp_freqs_standard - freq))
    return c_sharp_freqs_standard[index]

deviations = []
for freq in fre_arr:
    closest_c_sharp = find_closest_c_sharp(freq)
    deviation = freq - closest_c_sharp
    deviations.append((freq, closest_c_sharp, deviation))

deviations.sort(key=lambda x: x[0])


abs_deviations = [(d[2]) for d in deviations]
avg_deviation = np.mean(abs_deviations)
print(f"\nAverage absolute deviation for all frequencies: {avg_deviation:.2f} Hz")

close_deviations = [(d[0], d[1], d[2]) for d in deviations if abs(d[2]) < 2]

if close_deviations:
    close_freqs = [d[0] for d in close_deviations]
    close_deviations_values = [abs(d[2]) for d in close_deviations]
    avg_close_deviation = np.mean(close_deviations_values)

    print(f"\nNumber of frequencies with <2 Hz deviation: {len(close_deviations)}")
    print(f"Average deviation for frequencies <2 Hz from C#: {avg_close_deviation:.2f} Hz")

    # Print these close frequencies
    print("\nFrequencies with <2 Hz deviation from nearest C#:")
    print("Detected (Hz)\tClosest C# (Hz)\tDeviation (Hz)")
    for detected, closest, deviation in close_deviations:
        print(f"{detected:.2f}\t\t{closest:.2f}\t\t{deviation:.2f}")

    # Plot these close frequencies
    plt.figure(figsize=(12, 6))
    plt.scatter(close_freqs, close_deviations_values, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('Frequencies with <2 Hz Deviation from Closest C#')
    plt.xlabel('Detected Frequency (Hz)')
    plt.ylabel('Absolute Deviation (Hz)')
    plt.xscale('log')  # Use log scale for frequency axis
    plt.show()
else:
    print("\nNo frequencies found with less than 2 Hz deviation from nearest C#.")
