import numpy as np

K = 2048  # subcarriers = K
CP = K // 4
P = 64  # number of pilot carriers per OFDM block
#pilotValue = 1+1j
# allCarriers = np.arange(K)  # indices of all subcarriers ([0, 1, ... K-1])
# pilotCarriers = allCarriers[::K // P]  # Pilots is every (K/P)th carrier.
#pilotCarriers = np.hstack([pilotCarriers, np.array([allCarriers[-1]])])
#P = P+1
# dataCarriers = np.delete(allCarriers, pilotCarriers)
mu = 2    # one symbol combined with two bits for QAM or QPSK (LJS)
# number of payload bits per OFDM symbol
# payloadBits_per_OFDM = len(dataCarriers) * mu

# payloadbits per OFDM version 2 (decided by how many data carriers per OFDM , LJS)
# payloadBits_per_OFDM = K * mu

SNRdb = 25  # signal to noise-ratio in dB at the receiver

mapping_table = {
    (0, 0): -1 - 1j,
    (0, 1): -1 + 1j,
    (1, 0): 1 - 1j,
    (1, 1): 1 + 1j,
}

demapping_table = {v: k for k, v in mapping_table.items()}


def Modulation(bits):
    # for i in range(*bits.shape):
    # This is just for QAM modulation
    bit_r = bits.reshape((int(len(bits) / mu), mu))
    return (2 * bit_r[:, 0] - 1) + 1j * (2 * bit_r[:, 1] - 1)


def IDFT(OFDM_data):
    return np.fft.ifft(OFDM_data)   # np.fft.ifft(OFDM_data)*np.sqrt(K)  (lJS)


def addCP(OFDM_time):
    cp = OFDM_time[-CP:]  # take the last CP samples ...
    return np.hstack([cp, OFDM_time])  # ... and add them to the beginning


# construct the another version is including impulse noise(LJS)
def channel_BG(signal, SNRdb):
    # Bernoulli-Gaussian channel          # lJS
    prob = 0.002  # prob
    # convolved = np.convolve(signal, channelResponse)
    signal_power = np.mean(abs(signal**2))
    print(signal_power)
    # sigma2 = signal_power * 10**(-SNRdb / 10)      # (signal_power/2)  (LJS)
    sigma3 = 50
    # Gaussian = np.random.randn(*signal.shape) + 1j * \
    #     np.random.randn(*signal.shape)
    power1 = np.zeros([*signal.shape])
    # for i in range(*signal.shape):
    #     k = np.random.rand()
    #     # if k > prob:
    #     #     power1[i] = np.sqrt(sigma2 / 2)
    #     if k <= prob:
    #         power1[i] = np.sqrt(sigma3 / 2)
    #         print('p1=', i + 1)
    power2 = np.zeros([*signal.shape])
    for i in range(*signal.shape):
        k = np.random.rand()
        n = np.random.binomial(n=1, p=0.5)
        # if k > prob:
        # power2[i] = np.sqrt(sigma2 / 2)
        if k <= prob:
            if i <= 1000:
                if n == 1:
                    power1[i] = np.sqrt(sigma3 / 2)
                    power2[i] = np.sqrt(sigma3 / 2)
                    print('ps=', i + 1)
                else:
                    power1[i] = np.sqrt(sigma3 / 2)
                    power2[i] = np.sqrt(sigma3 / 2)
                    power1[i+1] = np.sqrt(sigma3 / 2)
                    power2[i+1] = np.sqrt(sigma3 / 2)
                    power1[i+2] = np.sqrt(sigma3 / 2)
                    power2[i+2] = np.sqrt(sigma3 / 2)
                    power1[i+3] = np.sqrt(sigma3 / 2)
                    power2[i+3] = np.sqrt(sigma3 / 2)
                    power1[i+4] = np.sqrt(sigma3 / 2)
                    power2[i+4] = np.sqrt(sigma3 / 2)
                    print('pc=', i + 1)
    # noise1 = np.multiply(power1, Gaussian.real)
    # noise2 = np.multiply(power2, Gaussian.imag)
    noise1 = power1
    noise2 = power2
    noise_BG = np.zeros([*signal.shape]).astype(complex)
    noise_BG.real = noise1
    noise_BG.imag = noise2
    nsymbol = noise_BG + signal     # NoiseSymbol
    nsymbolr = nsymbol.real
    nsymboli = nsymbol.imag
    nsymbol = []
    for i in range(0, K+CP):
        nsymbol.append([nsymbolr[i], nsymboli[i]])
    nsymbol = np.array(nsymbol)
    np.savetxt('Noise.txt', noise_BG)
    np.savetxt('NoiseReal.txt', noise_BG.real)
    np.savetxt('NoiseImag.txt', noise_BG.imag)
    np.savetxt('NoiseSymbol.txt', nsymbol)
    np.savetxt('NoiseSymbolReal.txt', noise_BG.real + signal.real)
    np.savetxt('NoiseSymbolImag.txt', noise_BG.imag + signal.imag)
    np.hstack


def ofdm_simulate_BG(codeword, SNRdb):       # LJS
    # OFDM_data = np.zeros(K, dtype=complex)
    # OFDM_data[allCarriers] = pilotValue
    # OFDM_time = IDFT(OFDM_data)
    # OFDM_withCP = addCP(OFDM_time)
    # OFDM_TX = OFDM_withCP
    # OFDM_RX = channel_BG(OFDM_TX, channelResponse, SNRdb)
    # OFDM_RX_noCP = removeCP(OFDM_RX)
    # ----- target inputs ---
    symbol = np.zeros(K, dtype=complex)
    codeword_qam = Modulation(codeword)
    symbol[np.arange(K)] = codeword_qam
    OFDM_data_codeword = symbol
    OFDM_time_codeword = np.fft.ifft(OFDM_data_codeword) * np.sqrt(K)
    OFDM_withCP_cordword = addCP(OFDM_time_codeword)
    np.savetxt('PreSymbol.txt', OFDM_data_codeword)
    np.savetxt('Symbol.txt', OFDM_withCP_cordword)
    np.savetxt('SymbolReal.txt', OFDM_withCP_cordword.real)
    np.savetxt('SymbolImag.txt', OFDM_withCP_cordword.imag)
    OFDM_RX_codeword = channel_BG(OFDM_withCP_cordword, SNRdb)


for index_k in range(0, 1):
    bits = np.random.binomial(n=1, p=0.5, size=(4096, ))
    print(bits)
    ofdm_simulate_BG(bits, SNRdb)
    # channel_response = channel_response_set_train[np.random.randint(0,len(128))]
    # signal_output, para = ofdm_simulate_AWGN(bits,channel_response,SNRdb)
    # np.savetxt('Symbolori.txt', bits)
