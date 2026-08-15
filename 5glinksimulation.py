

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc

# ---------------------- 1. System Parameters ----------------------
N_sc = 64                      # Number of OFDM subcarriers (FFT size)
N_cp = 16                      # Cyclic prefix length (25% of N_sc)
N_symbols = 3000               # OFDM symbols simulated per SNR point
                                # (raised from 500: at high SNR, few hundred
                                # symbols aren't enough to catch a single bit
                                # error, which produces BER=0 and breaks the
                                # log-scale plot - always check your lowest
                                # BER point has caught enough error events)
MOD_ORDER = 4                  # 4 = QPSK, 16 = 16-QAM
bits_per_sym = int(np.log2(MOD_ORDER))

SNR_dB_range = np.arange(0, 22, 2)
EbN0_dB_range = SNR_dB_range - 10 * np.log10(bits_per_sym)

# Multipath channel: simple 3-tap power-delay profile, normalized to unit power
channel_taps = np.array([0.8, 0.5, 0.3])
channel_taps = channel_taps / np.linalg.norm(channel_taps)

rng = np.random.default_rng(seed=42)  # reproducible results


# ---------------------- 2. QPSK mapping / demapping ----------------------
def qpsk_mod(bits):
    """Gray-coded QPSK: bit pairs -> unit-average-power complex symbols."""
    bits = bits.reshape(-1, 2)
    i = 1 - 2 * bits[:, 0]
    q = 1 - 2 * bits[:, 1]
    return (i + 1j * q) / np.sqrt(2)


def qpsk_demod(symbols):
    """Hard-decision QPSK demapping back to bits."""
    bits = np.zeros((len(symbols), 2), dtype=int)
    bits[:, 0] = (symbols.real < 0).astype(int)
    bits[:, 1] = (symbols.imag < 0).astype(int)
    return bits.reshape(-1)


def awgn(signal, snr_db):
    """Add complex AWGN at the specified SNR (dB), power measured from signal."""
    sig_power = np.mean(np.abs(signal) ** 2)
    noise_power = sig_power / (10 ** (snr_db / 10))
    noise = np.sqrt(noise_power / 2) * (
        rng.standard_normal(signal.shape) + 1j * rng.standard_normal(signal.shape)
    )
    return signal + noise


# ---------------------- 3. Main simulation loop ----------------------
BER_sim = np.zeros(len(SNR_dB_range))

for idx, snr_db in enumerate(SNR_dB_range):
    total_bit_errors = 0
    total_bits = 0

    for _ in range(N_symbols):
        # 3a. Bit generation
        tx_bits = rng.integers(0, 2, N_sc * bits_per_sym)

        # 3b. QPSK mapping
        tx_symbols = qpsk_mod(tx_bits)

        # 3c. OFDM modulation (IFFT)
        tx_time = np.fft.ifft(tx_symbols, N_sc) * np.sqrt(N_sc)

        # 3d. Cyclic prefix insertion
        tx_cp = np.concatenate([tx_time[-N_cp:], tx_time])

        # 3e. Multipath channel (linear convolution, truncated to symbol length)
        rx_faded = np.convolve(tx_cp, channel_taps, mode="full")[: len(tx_cp)]

        # 3f. AWGN
        rx_noisy = awgn(rx_faded, snr_db)

        # 3g. Cyclic prefix removal
        rx_time = rx_noisy[N_cp:]

        # 3h. OFDM demodulation (FFT)
        rx_freq = np.fft.fft(rx_time, N_sc) / np.sqrt(N_sc)

        # 3i. Zero-Forcing equalization (perfect CSI assumed)
        H = np.fft.fft(channel_taps, N_sc)
        rx_eq = rx_freq / H

        # 3j. QPSK demapping
        rx_bits = qpsk_demod(rx_eq)

        # 3k. BER accumulation
        total_bit_errors += np.sum(rx_bits != tx_bits)
        total_bits += len(tx_bits)

    BER_sim[idx] = total_bit_errors / total_bits
    print(f"SNR = {snr_db:2d} dB | BER = {BER_sim[idx]:.6f}")


# ---------------------- 4. Theoretical BER (QPSK over AWGN, no fading) ----------------------
def qpsk_theoretical_ber(ebn0_db):
    ebn0_lin = 10 ** (ebn0_db / 10)
    return 0.5 * erfc(np.sqrt(ebn0_lin))


BER_theory = qpsk_theoretical_ber(EbN0_dB_range)

# ---------------------- 5. Plot results ----------------------
plt.figure(figsize=(8, 6))
plt.semilogy(SNR_dB_range, BER_sim, "b-o", linewidth=1.5, label="Simulated (multipath + ZF eq.)")
plt.semilogy(SNR_dB_range, BER_theory, "r--", linewidth=1.5, label="Theoretical (AWGN only, QPSK)")
plt.grid(True, which="both")
plt.xlabel("SNR (dB)")
plt.ylabel("Bit Error Rate (BER)")
plt.title(f"OFDM Link BER vs SNR (QPSK, {N_sc} subcarriers, {len(channel_taps)}-tap channel)")
plt.legend(loc="lower left")
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/ofdm_ber_vs_snr.png", dpi=150)
print("\nPlot saved to ofdm_ber_vs_snr.png")

print("\nSimulation complete. The simulated curve sits above the AWGN")
print("theoretical curve because of residual multipath distortion and")
print("noise enhancement from Zero-Forcing equalization at deep fades -")
print("this gap is expected and worth explaining in an interview.")

