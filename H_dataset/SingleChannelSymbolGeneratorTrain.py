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

SNRdb = 20  # signal to noise-ratio in dB at the receiver

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
def channel_BG(signal, channelResponse, SNRdb):
    # Bernoulli-Gaussian channel          # lJS
    # IGR = 50  # impulse gaussian ratio
    np.random.seed()
    prob = 0.005  # prob
    convolved = np.convolve(signal, channelResponse)
    signal_power = np.mean(abs(convolved**2))
    sigma2 = signal_power * 10**(-SNRdb / 10)      # (signal_power/2)  (LJS)
    # sigma3 = sigma2 * IGR
    sigma3 = 15
    Gaussian = np.random.randn(*convolved.shape) + 1j * \
        np.random.randn(*convolved.shape)
    power1 = np.zeros([*convolved.shape])
    power2 = np.zeros([*convolved.shape])
    noise_position = []
    print('Channel Length :', len(channelResponse))
    print('Bits Length :', len(convolved))
    # print('IGR :', IGR)
    print('Signal Power :', signal_power)
    print('AWGN Power :', sigma2)
    print('Impulse Power :', sigma3)
    print('SNR:', SNRdb)
    print('Impulse Probability :', prob*100, '%')
    for i in range(*convolved.shape):
        k = np.random.rand()
        if k > prob:
            power1[i] = np.sqrt(sigma2 / 2)
            power2[i] = np.sqrt(sigma2 / 2)
    for i in range(*convolved.shape):
        k = np.random.rand()
        n = np.random.binomial(n=1, p=0.5)
        if k <= prob:
            if i <= 1000:
                if n == 1:
                    power1[i] = np.sqrt(sigma3 / 2)
                    power2[i] = np.sqrt(sigma3 / 2)
                    # print('impulse_position_single =', i + 1)
                    j = i + 1
                    # position = 'single ' + str(j)
                    position = str(j)
                    noise_position.append(position)
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
                    # print('impulse_position_mutiple =', i + 1)
                    j = i + 1
                    # position = 'mutiple ' + str(j)
                    for m in range(5):
                        position = str(j)
                        j += 1
                        noise_position.append(position)
    print('Real anomaly is on the', noise_position)
    noise1 = np.multiply(power1, Gaussian.real)
    noise2 = np.multiply(power2, Gaussian.imag)
    noise_BG = np.zeros([*convolved.shape]).astype(complex)
    noise_BG.real = noise1
    noise_BG.imag = noise2
    noise_symbol = noise_BG + convolved     # NoiseSymbol
    noise_symbol_real = noise_symbol.real
    noise_symbol_image = noise_symbol.imag
    noise_symbol = []
    for i in range(0, len(convolved)):
        noise_symbol.append(
            np.sqrt((noise_symbol_image[i]**2)+(noise_symbol_real[i]**2)))
    noise_symbol = np.array(noise_symbol)
    np.savetxt('Noise.txt', noise_BG)
    np.savetxt('NoiseReal.txt', noise_BG.real)
    np.savetxt('NoiseImag.txt', noise_BG.imag)
    np.savetxt('NoiseSymbol.txt', noise_symbol)
    np.savetxt('NoiseSymbolReal.txt', noise_BG.real + convolved.real)
    np.savetxt('NoiseSymbolImag.txt', noise_BG.imag + convolved.imag)
    np.savetxt('NoisePosition.npy', noise_position, fmt="%s")


def ofdm_simulate_BG(codeword, channelResponse, SNRdb):       # LJS
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
    channel_BG(OFDM_withCP_cordword, channelResponse, SNRdb)


H_folder_test = '../H_dataset/test/'
test_idx_low = 301
test_idx_high = 401
channel_response_set_test = []
for test_idx in range(test_idx_low, test_idx_high):
    # print("Processing the train", train_idx, "th document")
    H_file = H_folder_test + str(test_idx) + '.txt'
    with open(H_file) as f:
        for line in f:
            numbers_str = line.split()
            # np.shape(numbers_str)=32 x 1
            numbers_float = [float(x) for x in numbers_str]
            # np.shape(numbers_float)=32 x 1
            h_response = np.asarray(numbers_float[0:int(len(numbers_float) / 2)]) +\
                1j * \
                np.asarray(
                    numbers_float[int(len(numbers_float) / 2):len(numbers_float)])
            channel_response_set_test.append(h_response)


for index_k in range(0, 1):
    bits = np.random.binomial(n=1, p=0.5, size=(4096, ))
    np.random.seed(6)
    # print(bits)
    # ofdm_simulate_BG(bits, SNRdb)
    channel_response = channel_response_set_test[np.random.randint(
        0, len(channel_response_set_test))]
    ofdm_simulate_BG(bits, channel_response, SNRdb)
