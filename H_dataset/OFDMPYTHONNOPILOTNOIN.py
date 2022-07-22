import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
import os

K = 1024  # number of OFDM subcarriers
CP = K // 64  # length of the cyclic prefix: 25% of the block
# P = 128  # number of pilot carriers per OFDM block
# pilotValue = 1+1j  # The known value each pilot transmits
allCarriers = np.arange(K)  # indices of all subcarriers ([0, 1, ... K-1])
# pilotCarriers = allCarriers[::K//P]  # Pilots is every (K/P)th carrier.
# For convenience of channel estimation, let's make the last carrier also be a pilot
# pilotCarriers = np.hstack([pilotCarriers, np.array([allCarriers[-1]])])
# P = P+1
# data carriers are all remaining carriers
dataCarriers = allCarriers

mu = 2  # bits per symbol (i.e. 16QAM)
# number of payload bits per OFDM symbol
payloadBits_per_OFDM = len(dataCarriers)*mu
mapping_table = {
    (0, 0): -1 - 1j,
    (0, 1): -1 + 1j,
    (1, 0): 1 - 1j,
    (1, 1): 1 + 1j,
}
demapping_table = {v: k for k, v in mapping_table.items()}

# the impulse response of the wireless channel
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


def SP(bits):
    return bits.reshape((len(dataCarriers), mu))


def Modulation(bits):
    return np.array([mapping_table[tuple(b)] for b in bits])


def OFDM_symbol(QAM_payload):
    symbol = np.zeros(K, dtype=complex)  # the overall K subcarriers
    # symbol[pilotCarriers] = pilotValue  # allocate the pilot subcarriers
    symbol = QAM_payload  # allocate the pilot subcarriers
    return symbol


def IDFT(OFDM_data):
    return np.fft.ifft(OFDM_data)*np.sqrt(K)


def addCP(OFDM_time):
    cp = OFDM_time[-CP:]               # take the last CP samples ...
    return np.hstack([cp, OFDM_time])  # ... and add them to the beginning


def channel(signal, channelResponse, SNRdb):
    # Bernoulli-Gaussian channel          # lJS
    prob = 0.005
    SINRdb = -15
    convolved = np.convolve(signal, channelResponse)
    # convolved = signal
    signal_power = np.mean(abs(convolved**2))
    sigma2 = signal_power * 10**(-SNRdb / 10)      # (signal_power/2)  (LJS)
    sigma3 = signal_power * 10**(-SINRdb / 10)
    Gaussian = np.random.randn(*convolved.shape) + 1j * \
        np.random.randn(*convolved.shape)
    power1 = np.zeros([*convolved.shape])
    power2 = np.zeros([*convolved.shape])
    noise_position = []
    for i in range(*convolved.shape):
        power1[i] = np.sqrt(sigma2 / 2)
        power2[i] = np.sqrt(sigma2 / 2)
    for i in range(*convolved.shape):
        k = np.random.rand()
        if k <= prob:
            power1[i] = np.sqrt(sigma3 / 2)
            power2[i] = np.sqrt(sigma3 / 2)
            # print('impulse_position_single =', i + 1)
            j = i + 1
            # position = 'single ' + str(j)
            position = str(j)
            noise_position.append(position)
    # [noise_position_res.append(x)
    #  for x in noise_position if x not in noise_position_res]
    # print('Real anomaly is on the', noise_position_res)
    noise1 = np.multiply(power1, Gaussian.real)
    noise2 = np.multiply(power2, Gaussian.imag)
    noise_BG = np.zeros([*convolved.shape]).astype(complex)
    noise_BG.real = noise1
    noise_BG.imag = noise2
    noise_symbol = noise_BG + convolved     # NoiseSymbol
    return noise_symbol, convolved, noise_position, sigma2, sigma3


def removeCP(signal):
    return signal[CP:(CP+K)]


def DFT(OFDM_RX):
    return np.fft.fft(OFDM_RX/np.sqrt(K))


# def channelEstimate(OFDM_demod):
#     # extract the pilot values from the RX signal
#     pilots = OFDM_demod[pilotCarriers]
#     Hest_at_pilots = pilots / pilotValue  # divide by the transmitted pilot values
#
#     # Perform interpolation between the pilot carriers to get an estimate
#     # of the channel in the data carriers. Here, we interpolate absolute value and phase
#     # separately
#     Hest_abs = scipy.interpolate.interp1d(
#         pilotCarriers, abs(Hest_at_pilots), kind='linear')(allCarriers)
#     Hest_phase = scipy.interpolate.interp1d(
#         pilotCarriers, np.angle(Hest_at_pilots), kind='linear')(allCarriers)
#     Hest = Hest_abs * np.exp(1j*Hest_phase)
#
#     plt.plot(allCarriers, abs(H_exact), label='Correct Channel')
#     plt.stem(pilotCarriers, abs(Hest_at_pilots), label='Pilot estimates')
#     plt.plot(allCarriers, abs(Hest),
#              label='Estimated channel via interpolation')
#     plt.grid(True)
#     plt.xlabel('Carrier index')
#     plt.ylabel('$|H(f)|$')
#     plt.legend(fontsize=10)
#     plt.ylim(0, 2)
#
#     return Hest


def equalize(OFDM_demod, Hest):
    # Hest = np.linalg.inv(Hest)
    # OFDM_demod = np.expand_dims(OFDM_demod, axis=1)
    return OFDM_demod / Hest


def get_payload(equalized):
    return equalized[dataCarriers]


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


for SNRdb in range(5, 55, 5):
    total_epochs = 10000
    if os.path.isfile('checkpoint.txt'):
        checkpoint = np.loadtxt('checkpoint.txt')
    else:
        np.savetxt('checkpoint.txt', [0, 0, 0, 0])
        checkpoint = np.loadtxt('checkpoint.txt')
    valid_epochs = int(checkpoint[0])
    total_BER = float(checkpoint[1])
    total_impulse = int(checkpoint[2])
    total_best_threshold = int(checkpoint[3])
    BestTname = 'BestT'+str(SNRdb)+'.txt'
    BestBERname = 'BestBER'+str(SNRdb)+'.txt'
    # BT = [4.5, 4.5, 4, 4, 4, 4, 4, 4, 4, 4]
    # BT = BT[int((SNRdb/5)-1)]
    temp_th = 0
    for x in range(total_epochs - valid_epochs):
        output_SNR = []
        output_BER = []
        threshold = []
        checkpoint = []
        result = []
        Eyksk = 0
        Eykyk = 0
        np.random.seed(0)
        channelResponse = channel_response_set_test[np.random.randint(
            0, len(channel_response_set_test))]
        np.random.seed()
        bits = np.random.binomial(n=1, p=0.5, size=(payloadBits_per_OFDM, ))
        OFDM_RX_CHANNEL = 0
        OFDM_RX_CHANNEL_INFORMATION = []
        bits_SP = SP(bits)
        QAM = Modulation(bits_SP)
        OFDM_data = OFDM_symbol(QAM)
        OFDM_time = IDFT(OFDM_data)
        OFDM_withCP = addCP(OFDM_time)
        OFDM_TX = OFDM_withCP
        OFDM_RX, OFDM_convolved, datacheck, w, g = channel(
            OFDM_TX, channelResponse, SNRdb)
        p = [0.999, 0.001]
        dk = [w, w+g]
        OFDM_RX_noCP = removeCP(OFDM_RX)
        for th in np.arange(0.5, 15.25, 0.25):
            OFDM_RX_ORIGIN = np.copy(OFDM_RX_noCP)
            Eyksk = ((1+((th**2)/(2*(1+dk[0]))))*p[0]*(np.exp(-((th**2)/(2*(1+dk[0]))))))+(
                (1+((th**2)/(2*(1+dk[1]))))*p[1]*(np.exp(-((th**2)/(2*(1+dk[1]))))))
            Eykyk = (p[0]*(dk[0]-(((th**2)/2)+(1+dk[0]))*(np.exp(-((th**2)/(2*(1+dk[0]))))))) + \
                (p[1]*(dk[1]-(((th**2)/2)+(1+dk[1]))
                 * (np.exp(-((th**2)/(2*(1+dk[1])))))))
            alpha = 1-Eyksk
            eout = 2+(2*Eykyk)
            gamma = ((eout/(2*(alpha**2)))-1)**(-1)
            output_SNR.append(10*np.log(np.abs(gamma))/2)
            threshold.append(th)
        best_threshold = 0
        OFDM_RX_copy = np.copy(OFDM_RX_noCP)
        norm = np.abs(OFDM_RX_copy)
        th_index = output_SNR.index(max(output_SNR))
        best_th = [threshold[th_index]]
        print(best_th)
        temp_th += best_th[0]
        print(temp_th/valid_epochs)
        for Known_impulse in range(*norm.shape):
            if (norm[int(Known_impulse)]) > best_th:
                OFDM_RX_copy[int(Known_impulse)] = 0
        OFDM_demod = DFT(OFDM_RX_copy)
        Hest = np.fft.fft(channelResponse, K)
        equalized_Hest = equalize(OFDM_demod, Hest)
        QAM_est = get_payload(equalized_Hest)
        PS_est, hardDecision = Demapping(QAM_est)
        bits_est = PS(PS_est)
        BER = np.mean(abs(bits - bits_est))
        total_best_threshold += best_threshold
        total_BER += BER
        total_impulse += len(datacheck)
        valid_epochs += 1
        avg_BER = (total_BER / valid_epochs)
        avg_impulse_prob = (total_impulse/(1055*valid_epochs)) * 100
        avg_best_threshold = (total_best_threshold/valid_epochs)
        checkpoint.extend([valid_epochs,
                           total_BER,
                           total_impulse,
                           total_best_threshold])
        np.savetxt(BestTname, output_SNR)
        # np.savetxt(BestBERname, output_BER)
        np.savetxt('threshold.txt', threshold)
        np.savetxt('checkpoint.txt', checkpoint)
        if valid_epochs % 1 == 0:
            print('INs :', len(datacheck))
            print('Epochs :', valid_epochs, '\navg.BER =',
                  avg_BER, '\navg.IN_prob =', avg_impulse_prob, '%', '\navg.best_threshold =', avg_best_threshold)
        result.extend(['BER', 'IN_prob', 'best_threshold', avg_BER,
                      avg_impulse_prob, avg_best_threshold])
        if valid_epochs == total_epochs:
            np.savetxt(BestBERname, np.reshape(
                result, (3, 2), order='F'), fmt="%s")
            os.remove('checkpoint.txt')
