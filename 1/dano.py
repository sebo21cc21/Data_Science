import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.signal import welch, periodogram

# Dane
t = np.linspace(0, 4 * math.pi, 100)
y = np.sin(t)
y2 = np.sin(2 * t) + 2 * np.sin(3 * t) + np.sin(5 * t)
y3 = y + np.random.randn(100)
y4 = y2 + np.random.randn(100)

# Parametry do analizy widma mocy
fs = 1 / (t[1] - t[0])  # Częstotliwość próbkowania (Hz)

# Tworzenie subplotów
plt.figure(figsize=(12, 8))

# Podwykres 1: Welch dla y
plt.subplot(221)
frequencies, Pxx_y = welch(y, fs=fs)
plt.semilogy(frequencies, Pxx_y)
plt.title("Welch dla y")
plt.xlabel("Częstotliwość (Hz)")
plt.ylabel("Gęstość mocy")

# Podwykres 2: Periodogram dla y
plt.subplot(222)
frequencies, Pxx_y = periodogram(y, fs=fs)
plt.semilogy(frequencies, Pxx_y)
plt.title("Periodogram dla y")
plt.xlabel("Częstotliwość (Hz)")
plt.ylabel("Gęstość mocy")

# Podwykres 3: Welch dla y2
plt.subplot(223)
frequencies, Pxx_y2 = welch(y2, fs=fs)
plt.semilogy(frequencies, Pxx_y2)
plt.title("Welch dla y2")
plt.xlabel("Częstotliwość (Hz)")
plt.ylabel("Gęstość mocy")

# Podwykres 4: Periodogram dla y2
plt.subplot(224)
frequencies, Pxx_y2 = periodogram(y2, fs=fs)
plt.semilogy(frequencies, Pxx_y2)
plt.title("Periodogram dla y2")
plt.xlabel("Częstotliwość (Hz)")
plt.ylabel("Gęstość mocy")

plt.tight_layout()
plt.show()
