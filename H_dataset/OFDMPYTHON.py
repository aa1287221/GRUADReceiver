import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
import os

K = 1024  # number of OFDM subcarriers
CP = K // 64  # length of the cyclic prefix: 25% of the block
P = 128  # number of pilot carriers per OFDM block
pilotValue = -1-1j  # The known value each pilot transmits
allCarriers = np.arange(K)  # indices of all subcarriers ([0, 1, ... K-1])
pilotCarriers = allCarriers[::K//P]  # Pilots is every (K/P)th carrier.
# For convenience of channel estimation, let's make the last carrier also be a pilot
pilotCarriers = np.hstack([pilotCarriers, np.array([allCarriers[-1]])])
P = P+1
INCounter = 0

# data carriers are all remaining carriers
dataCarriers = np.delete(allCarriers, pilotCarriers)
print(len(dataCarriers))
# plt.figure(figsize=(8, 0.8))
# plt.plot(pilotCarriers, np.zeros_like(pilotCarriers), 'bo', label='pilot')
# plt.plot(dataCarriers, np.zeros_like(dataCarriers), 'ro', label='data')
# plt.legend(fontsize=10, ncol=2)
# plt.xlim((-1, K))
# plt.ylim((-0.1, 0.3))
# plt.xlabel('Carrier index')
# plt.yticks([])
# plt.grid(True)
# plt.savefig("pilots.png")

mu = 2  # bits per symbol (i.e. 16QAM)
# number of payload bits per OFDM symbol
payloadBits_per_OFDM = len(dataCarriers)*mu
print(payloadBits_per_OFDM)
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


SNRdb = 45  # signal to noise-ratio in dB at the receiver


def SP(bits):
    return bits.reshape((len(dataCarriers), mu))


def Mapping(bits):
    return np.array([mapping_table[tuple(b)] for b in bits])


def OFDM_symbol(QAM_payload):
    symbol = np.zeros(K, dtype=complex)  # the overall K subcarriers
    symbol[pilotCarriers] = pilotValue  # allocate the pilot subcarriers
    symbol[dataCarriers] = QAM_payload  # allocate the pilot subcarriers
    return symbol


def IDFT(OFDM_data):
    return np.fft.ifft(OFDM_data)*np.sqrt(K)


def addCP(OFDM_time):
    cp = OFDM_time[-CP:]               # take the last CP samples ...
    return np.hstack([cp, OFDM_time])  # ... and add them to the beginning


def channel(signal, channelResponse):
    prob = 0.005  # prob
    convolved = np.convolve(signal, channelResponse)
    signal_power = np.mean(abs(convolved**2))
    # calculate noise power based on signal power and SNR
    sigma2 = signal_power * 10**(-SNRdb/10)
    sigma3 = 15
    Gaussian = np.random.randn(*convolved.shape) + 1j * \
        np.random.randn(*convolved.shape)
    power1 = np.zeros([*convolved.shape])
    power2 = np.zeros([*convolved.shape])
    for i in range(*convolved.shape):
        power1[i] = np.sqrt(sigma2 / 2)
        power2[i] = np.sqrt(sigma2 / 2)
    for i in range(*convolved.shape):
        k = np.random.rand()
        n = np.random.binomial(n=1, p=0.5)
        if k <= prob:
            if n == 1:
                power1[i] = np.sqrt(sigma3 / 2)
                power2[i] = np.sqrt(sigma3 / 2)
                INCounter += 1
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
                INCounter += 5

    noise1 = np.multiply(power1, Gaussian.real)
    noise2 = np.multiply(power2, Gaussian.imag)
    noise_BG = np.zeros([*convolved.shape]).astype(complex)
    noise_BG.real = noise1
    noise_BG.imag = noise2
    noise_symbol = noise_BG + convolved
    # print("RX Signal power: %.4f. Noise power: %.4f" % (signal_power, sigma2))
    return noise_symbol


def removeCP(signal):
    return signal[CP:(CP+K)]


def DFT(OFDM_RX):
    return np.fft.fft(OFDM_RX/np.sqrt(K))


def channelEstimate(OFDM_demod, H_exact):
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
    plt.savefig("channelEstimate.png")
    return Hest


def equalize(OFDM_demod, Hest):
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


total_epochs = 10000
if os.path.isfile('checkpoint.txt'):
    checkpoint = np.loadtxt('checkpoint.txt')
else:
    np.savetxt('checkpoint.txt', [0, 0])
    checkpoint = np.loadtxt('checkpoint.txt')
valid_epochs = int(checkpoint[0])
total_BER = float(checkpoint[1])
for x in range(total_epochs - valid_epochs):
    checkpoint = []
    result = []
    channelResponse = channel_response_set_test[np.random.randint(
        0, len(channel_response_set_test))]
    bits = np.random.binomial(n=1, p=0.5, size=(payloadBits_per_OFDM, ))
    bits_SP = SP(bits)
    QAM = Mapping(bits_SP)
    OFDM_data = OFDM_symbol(QAM)
    OFDM_time = IDFT(OFDM_data)
    OFDM_withCP = addCP(OFDM_time)
    OFDM_TX = OFDM_withCP
    OFDM_RX = channel(OFDM_TX, channelResponse)
    OFDM_RX_noCP = removeCP(OFDM_RX)
    OFDM_demod = DFT(OFDM_RX_noCP)
    # plt.figure()
    H_exact = np.fft.fft(channelResponse, K)
    Hest = channelEstimate(OFDM_demod, H_exact)
    # plt.savefig("channelEstimate.png")
    equalized_Hest = equalize(OFDM_demod, Hest)
    QAM_est = get_payload(equalized_Hest)
    PS_est, hardDecision = Demapping(QAM_est)
    bits_est = PS(PS_est)
    BER = np.sum(abs(bits - bits_est)) / len(bits)
    total_BER += BER
    valid_epochs += 1
    avg_BER = (total_BER / valid_epochs)
    checkpoint.extend([valid_epochs,
                       total_BER])
    np.savetxt('checkpoint.txt', checkpoint)
    if valid_epochs % 1000 == 0:
        print('avg.BER =', avg_BER)
    result.extend(['BER',
                   avg_BER])
    if valid_epochs == total_epochs:
        np.savetxt('resultSNR40.txt', np.reshape(
            result, (1, 2), order='F'), fmt="%s")
        os.remove('checkpoint.txt')


# plt.figure(figsize=(8, 2))
# plt.plot(abs(OFDM_TX), label='TX signal')
# plt.plot(abs(OFDM_RX), label='RX signal')
# plt.legend(fontsize=10)
# plt.xlabel('Time')
# plt.ylabel('$|x(t)|$')
# plt.grid(True)
# plt.savefig("Time-domainSignals.png")
#
# plt.figure()
# plt.plot(QAM_est.real, QAM_est.imag, 'bo')
# plt.grid(True)
# plt.xlabel('Real part')
# plt.ylabel('Imaginary Part')
# plt.title("Received constellation")  # il
# plt.savefig("Constellation.png")
#
# plt.figure()
# for qam, hard in zip(QAM_est, hardDecision):
#     plt.plot([qam.real, hard.real], [qam.imag, hard.imag], 'b-o')
#     plt.plot(hardDecision.real, hardDecision.imag, 'ro')
# plt.grid(True)
# plt.xlabel('Real part')
# plt.ylabel('Imaginary part')
# plt.title('Hard Decision demapping')  # il
# plt.savefig("HardDecision.png")
#
# # print("Obtained Bit error rate: ", np.sum(abs(bits-bits_est))/len(bits))
#
# plt.show()
