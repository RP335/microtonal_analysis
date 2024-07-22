import numpy as np
import audioflux as af

import matplotlib.pyplot as plt
from audioflux.display import fill_spec, fill_wave
from audioflux.type import PitchType, SpectralFilterBankNormalType
from audioflux.utils import note_to_hz

audio_arr, sr = af.read('raga2.wav')

# Create CQT object
cqt_obj = af.CQT(num=168, samplate=sr, low_fre=note_to_hz('C1'), bin_per_octave=24,
             slide_length=256, normal_type=SpectralFilterBankNormalType.AREA)


cqt_arr = cqt_obj.cqt(audio_arr)
chroma_cqt_arr = cqt_obj.chroma(cqt_arr, chroma_num=24)

audio_len = audio_arr.shape[-1]

fig, ax = plt.subplots()
img = fill_spec(np.abs(cqt_arr), axes=ax,
                x_coords=cqt_obj.x_coords(audio_len), x_axis='time',
                y_coords=cqt_obj.y_coords(), y_axis='log',
                title='CQT')
fig.colorbar(img, ax=ax)

fig, ax = plt.subplots()
img = fill_spec(chroma_cqt_arr, axes=ax,
                x_coords=cqt_obj.x_coords(audio_len),
                x_axis='time', y_axis='chroma',
                title='Chroma-CQT')
fig.colorbar(img, ax=ax)

plt.show()

# Fundamental Frequency estimation

audio_arr, sr = af.read('raga2.wav')

obj = af.PitchYIN()

fre_arr, value_arr1, value_arr2 = obj.pitch(audio_arr)
fre_arr[fre_arr < 1] = np.nan

fig, ax = plt.subplots(nrows=2, figsize=(8, 6), sharex=True)
times = np.arange(0, fre_arr.shape[-1]) * (obj.slide_length / obj.samplate)

fill_wave(audio_arr, samplate=sr, axes=ax[0])

ax[1].xaxis.set_label_text("Time(s)")
ax[1].yaxis.set_label_text("Frequency(Hz)")
ax[1].plot(times, fre_arr, label='fre', linewidth=3)

real_fre_arr = np.zeros_like(fre_arr)
real_fre_arr[25:48] = 261.6
real_fre_arr[56:78] = 293.7
real_fre_arr[87:107] = 329.6
real_fre_arr[118:135] = 349.2
real_fre_arr[150:169] = 392.0
real_fre_arr[179:200] = 440.0
real_fre_arr[212:243] = 493.9
real_fre_arr[real_fre_arr == 0] = np.nan
ax[1].plot(times, real_fre_arr, color='red', label='fre', linewidth=2)

plt.show()
