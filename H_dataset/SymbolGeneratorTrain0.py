import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
import scipy.signal

K = 1024  # subcarriers = K
CP = K // 64
P = 128  # number of pilot carriers per OFDM block
pilotValue = 1+1j
allCarriers = np.arange(K)  # indices of all subcarriers ([0, 1, ... K-1])
pilotCarriers = allCarriers[::K // P]  # Pilots is every (K/P)th carrier.
pilotCarriers = np.hstack([pilotCarriers, np.array([allCarriers[-1]])])
P = P+1
dataCarriers = np.delete(allCarriers, pilotCarriers)

mu = 2    # one symbol combined with two bits for QAM or QPSK (LJS)
# number of payload bits per OFDM symbol
payloadBits_per_OFDM = len(dataCarriers) * mu

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


def SP(bits):
    return bits.reshape((len(dataCarriers), mu))
    # return bits.reshape(K, mu)


def Modulation(bits):
    return np.array([mapping_table[tuple(b)] for b in bits])


def OFDM_symbol(QAM_payload):
    symbol = np.zeros(K, dtype=complex)  # the overall K subcarriers
    # symbol = QAM_payload
    symbol[pilotCarriers] = pilotValue  # allocate the pilot subcarriers
    symbol[dataCarriers] = QAM_payload  # allocate the pilot subcarriers
    return symbol


def IDFT(OFDM_data):
    # np.fft.ifft(OFDM_data)*np.sqrt(K)  (lJS)
    return np.fft.ifft(OFDM_data)*np.sqrt(K)


def addCP(OFDM_time):
    cp = OFDM_time[-CP:]  # take the last CP samples ...
    return np.hstack([cp, OFDM_time])  # ... and add them to the beginning


# construct the another version is including impulse noise(LJS)
def channel_BG(signal, channelResponse, SNRdb):
    # Bernoulli-Gaussian channel          # lJS
    # IGR = 50  # impulse gaussian ratio
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
            if i <= 500:
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
    # np.savetxt('Noise.txt', noise_BG)
    # np.savetxt('NoiseReal.txt', noise_BG.real)
    # np.savetxt('NoiseImag.txt', noise_BG.imag)
    # np.savetxt('NoiseSymbolReal.txt', noise_BG.real + convolved.real)
    # np.savetxt('NoiseSymbolImag.txt', noise_BG.imag + convolved.imag)
    np.savetxt('NoiseSymbol.txt', noise_symbol)
    np.savetxt('NoisePosition.npy', noise_position, fmt="%s")
    return noise_BG + convolved


def removeCP(signal):
    return signal[CP:(CP+K)]


def DFT(OFDM_RX):
    return np.fft.fft(OFDM_RX/np.sqrt(K))


def channelEstimate(OFDM_demod):
    # extract the pilot values from the RX signal
    pilots = OFDM_demod[pilotCarriers]
    Hest_at_pilots = pilots / pilotValue  # divide by the transmitted pilot values

    # Perform interpolation between the pilot carriers to get an estimate
    # of the channel in the data carriers. Here, we interpolate absolute value and phase
    # separately
    Hest_abs = scipy.interpolate.interp1d(
        pilotCarriers, abs(Hest_at_pilots), kind='linear')(allCarriers)
    Hest_phase = scipy.interpolate.interp1d(
        pilotCarriers, np.angle(Hest_at_pilots), kind='linear')(allCarriers)
    Hest = Hest_abs * np.exp(1j*Hest_phase)

    plt.plot(allCarriers, abs(H_exact), label='Correct Channel')
    plt.stem(pilotCarriers, abs(Hest_at_pilots), label='Pilot estimates')
    plt.plot(allCarriers, abs(Hest),
             label='Estimated channel via interpolation')
    plt.grid(True)
    plt.xlabel('Carrier index')
    plt.ylabel('$|H(f)|$')
    plt.legend(fontsize=10)
    plt.ylim(0, 2)

    return Hest


def equalize(OFDM_demod, Hest):
    return OFDM_demod / Hest
    # return OFDM_demod


def get_payload(equalized):
    return equalized[dataCarriers]
    # return np.array(equalized)


def Demapping(QAM):
    # array of possible constellation points
    constellation = np.array([x for x in demapping_table.keys()])

    # calculate distance of each RX point to each possible point
    dists = abs(QAM.reshape((-1, 1)) - constellation.reshape((1, -1)))

    # for each element in QAM, choose the index in constellation
    # that belongs to the nearest constellation point
    const_index = dists.argmin(axis=1)

    # get back the real constellation point
    hardDecision = constellation[const_index]

    # transform the constellation point into the bit groups
    return np.vstack([demapping_table[C] for C in hardDecision]), hardDecision


def PS(bits):
    return bits.reshape((-1,))


def ofdm_simulate_BG(codeword, channelResponse, SNRdb):       # LJS
    bits_SP = SP(codeword)
    QAM = Modulation(bits_SP)
    OFDM_data = OFDM_symbol(QAM)
    OFDM_time = IDFT(OFDM_data)
    OFDM_withCP = addCP(OFDM_time)
    OFDM_TX = OFDM_withCP
    OFDM_RX = channel_BG(OFDM_TX, channel_response, SNRdb)
    # OFDM_RX, remainder = scipy.signal.deconvolve(OFDM_RX, channelResponse)
    OFDM_RX_noCP = removeCP(OFDM_RX)
    OFDM_demod = DFT(OFDM_RX_noCP)
    Hest = channelEstimate(OFDM_demod)
    equalized_Hest = equalize(OFDM_demod, Hest)
    # equalized_Hest = equalize(OFDM_demod, channelResponse)
    QAM_est = get_payload(equalized_Hest)
    PS_est, hardDecision = Demapping(QAM_est)
    bits_est = PS(PS_est)
    BER = np.sum(abs(bits-bits_est))/len(bits)
    return BER


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
    channel_response = channel_response_set_test[np.random.randint(
        0, len(channel_response_set_test))]
    H_exact = np.fft.fft(channel_response, K)
    bits = np.random.binomial(n=1, p=0.5, size=(payloadBits_per_OFDM, ))
    # print(bits)
    # ofdm_simulate_BG(bits, SNRdb)
    BER = ofdm_simulate_BG(bits, channel_response, SNRdb)
    print('BER =', BER)
