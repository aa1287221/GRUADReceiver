import numpy as np

Noise_Position_filepath = '/home/wky/GRUADReceiver/RNN-Time-series-Anomaly-Detection-master/dataset/ofdm/raw/NoisePosition.npy'
Noise_Symbol_filepath = '/home/wky/GRUADReceiver/RNN-Time-series-Anomaly-Detection-master/dataset/ofdm/raw/NoiseSymbol.txt'

K = 1024  # subcarriers = K
CP = K // 64
# P = 64  # number of pilot carriers per OFDM block
mu = 2    # one symbol combined with two bits for QAM or QPSK (LJS)
# payloadbits per OFDM version 2 (decided by how many data carriers per OFDM , LJS)
payloadBits_per_OFDM = K * mu


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
def channel_BG(signal, channelResponse, SNRdb, y):
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
    noise_position_res = []
    # print('Channel Length :', len(channelResponse))
    # print('Bits Length :', len(convolved))
    # print('Cyclic Prefix Length :', CP)
    # print('Signal Power :', signal_power)
    # print('AWGN Power :', sigma2)
    # print('Impulse Power :', sigma3)
    # print('SNR:', SNRdb)
    # print('Impulse Probability :', prob*100, '%')
    z = y*len(convolved)
    for i in range(*convolved.shape):
        power1[i] = np.sqrt(sigma2 / 2)
        power2[i] = np.sqrt(sigma2 / 2)
    for i in range(*convolved.shape):
        k = np.random.rand()
        n = np.random.binomial(n=1, p=0.7)
        c = np.random.randint(low=2, high=6)
        if k <= prob:
            if y < 50:
                if n == 1:
                    power1[i] = np.sqrt(sigma3 / 2)
                    power2[i] = np.sqrt(sigma3 / 2)
                    # print('impulse_position_single =', i + 1)
                    j = i + 1
                    # position = 'single ' + str(j)
                    position = str(j+z)
                    noise_position.append(position)
                else:
                    if i > 0:
                        if (i+c) < len(convolved):
                            for ii in range(c):
                                power1[i] = np.sqrt(sigma3 / 2)
                                power2[i] = np.sqrt(sigma3 / 2)
                                # power1[i+1] = np.sqrt(sigma3 / 2)
                                # power2[i+1] = np.sqrt(sigma3 / 2)
                                # power1[i+2] = np.sqrt(sigma3 / 2)
                                # power2[i+2] = np.sqrt(sigma3 / 2)
                                # power1[i+3] = np.sqrt(sigma3 / 2)
                                # power2[i+3] = np.sqrt(sigma3 / 2)
                                # power1[i+4] = np.sqrt(sigma3 / 2)
                                # power2[i+4] = np.sqrt(sigma3 / 2)
                                # print('impulse_position_mutiple =', i + 1)
                                j = i + 1
                                # position = 'mutiple ' + str(j)
                                position = str(j+z)
                                noise_position.append(position)
    [noise_position_res.append(x)
     for x in noise_position if x not in noise_position_res]
    # print('Real anomaly is on the', noise_position)
    noise1 = np.multiply(power1, Gaussian.real)
    noise2 = np.multiply(power2, Gaussian.imag)
    # noise1 = power1
    # noise2 = power2
    noise_BG = np.zeros([*convolved.shape]).astype(complex)
    noise_BG.real = noise1
    noise_BG.imag = noise2
    noise_symbol = noise_BG + convolved     # NoiseSymbol
    noise_symbol_real = noise_symbol.real
    noise_symbol_image = noise_symbol.imag
    noise_symbol = []
    for i in range(0, len(convolved)):
        noise_symbol.append(
            (noise_symbol_image[i]**2) + (noise_symbol_real[i]**2))
    noise_symbol = np.array(noise_symbol)
    # np.savetxt('Noise.txt', noise_BG)
    # np.savetxt('NoiseReal.txt', noise_BG.real)
    # np.savetxt('NoiseImag.txt', noise_BG.imag)
    # np.savetxt('NoiseSymbolReal.txt', noise_BG.real + convolved.real)
    # np.savetxt('NoiseSymbolImag.txt', noise_BG.imag + convolved.imag)
    np.savetxt(Noise_Symbol_filepath, noise_symbol)
    np.savetxt(Noise_Position_filepath, noise_position_res, fmt="%s")
    return noise_symbol, noise_position_res


def ofdm_simulate_BG(codeword, channelResponse, SNRdb, y):       # LJS
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
    # np.savetxt('PreSymbol.txt', OFDM_data_codeword)
    # np.savetxt('Symbol.txt', OFDM_withCP_cordword)
    # np.savetxt('SymbolReal.txt', OFDM_withCP_cordword.real)
    # np.savetxt('SymbolImag.txt', OFDM_withCP_cordword.imag)
    noise_symbol, noise_position_res = channel_BG(
        OFDM_withCP_cordword, channelResponse, SNRdb, y)
    return noise_symbol, noise_position_res


H_folder_test = '../RNN-Time-series-Anomaly-Detection-master/test/'
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

total_noise_symbol = []
total_noise_position_res = []
for y in range(100):
    for index_k in range(0, 1):
        np.random.seed(0)
        bits = np.random.binomial(n=1, p=0.5, size=(payloadBits_per_OFDM, ))
        np.random.seed()
        SNRdb = np.random.uniform(0, 100)
        channel_response = channel_response_set_test[np.random.randint(
            0, len(channel_response_set_test))]
        noise_symbol, noise_position_res = ofdm_simulate_BG(
            bits, channel_response, SNRdb, y)
        total_noise_symbol.extend(noise_symbol)
        total_noise_position_res.extend(noise_position_res)
np.savetxt(Noise_Symbol_filepath, total_noise_symbol)
np.savetxt(Noise_Position_filepath, total_noise_position_res, fmt="%s")
