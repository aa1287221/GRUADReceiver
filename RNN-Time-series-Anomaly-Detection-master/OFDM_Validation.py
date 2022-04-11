import os
import numpy as np
import scipy.interpolate
import scipy.signal
from numpy import linalg as LA

Noise_Position_filepath = '/home/wky/GRUADReceiver/RNN-Time-series-Anomaly-Detection-master/dataset/ofdm/raw/NoisePosition.npy'
Noise_Symbol_filepath = '/home/wky/GRUADReceiver/RNN-Time-series-Anomaly-Detection-master/dataset/ofdm/raw/NoiseSymbol.txt'

K = 1024  # subcarriers = K
CP = K // 64
# P = 128  # number of pilot carriers per OFDM block
# pilotValue = 1 + 1j
allCarriers = np.arange(K)  # indices of all subcarriers ([0, 1, ... K-1])
# pilotCarriers = allCarriers[::K // P]  # Pilots is every (K/P)th carrier.
# pilotCarriers = np.hstack([pilotCarriers, np.array([allCarriers[-1]])])
# P = P + 1
# dataCarriers = np.delete(allCarriers, pilotCarriers)
dataCarriers = allCarriers
mu = 2    # one symbol combined with two bits for QAM or QPSK (LJS)
payloadBits_per_OFDM = len(dataCarriers) * mu
# payloadBits_per_OFDM = K * mu

mapping_table = {
    (0, 0): -1 - 1j,
    (0, 1): -1 + 1j,
    (1, 0): 1 - 1j,
    (1, 1): 1 + 1j,
}

demapping_table = {v: k for k, v in mapping_table.items()}

total_epochs = 1000

if os.path.isfile('checkpoint.txt'):
    checkpoint = np.loadtxt('checkpoint.txt')
else:
    np.savetxt('checkpoint.txt', [0, 0, 0, 0, 0, 0, 0, 0, 0])
    checkpoint = np.loadtxt('checkpoint.txt')
valid_epochs = int(checkpoint[0])
total_accuracy = float(checkpoint[1])
total_fbeta = float(checkpoint[2])
total_precision = float(checkpoint[3])
total_recall = float(checkpoint[4])
total_τ = float(checkpoint[5])
total_signal_power = float(checkpoint[6])
total_BER = float(checkpoint[7])
total_impulse = int(checkpoint[8])

if os.path.isfile('checkpoint_SNR.txt'):
    checkpoint_SNR = np.loadtxt('checkpoint_SNR.txt')
else:
    np.savetxt('checkpoint_SNR.txt',
               np.reshape([5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], (26, 4), order='F'))
    checkpoint_SNR = np.loadtxt('checkpoint_SNR.txt')


def anomaly_detection(besttau):
    import argparse
    import torch
    import pickle
    import preprocess_data
    from model import model
    from pathlib import Path
    from anomalyDetector import fit_norm_distribution_param
    from anomalyDetector import anomalyScore
    from anomalyDetector import get_precision_recall
    parser = argparse.ArgumentParser(
        description='PyTorch RNN Anomaly Detection Model')
    parser.add_argument('--prediction_window_size', type=int, default=1,
                        help='prediction_window_size')
    parser.add_argument('--data', type=str, default='ofdm',
                        help='type of the dataset (ecg, gesture, power_demand, space_shuttle, respiration, nyc_taxi, ofdm')
    parser.add_argument('--filename', type=str, default='NoiseSymbol.pkl',
                        help='filename of the dataset')
    parser.add_argument('--beta', type=float, default=0.1,
                        help='beta value for f-beta score')

    args_ = parser.parse_args()
    # print('-' * 120)
    # print("=> loading checkpoint ")
    checkpoint = torch.load(
        str(Path('save', args_.data, 'checkpoint', args_.filename).with_suffix('.pth')))
    args = checkpoint['args']
    args.prediction_window_size = args_.prediction_window_size
    args.beta = args_.beta
    # print("=> loaded checkpoint")

    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    ##########################################################################
    # Load data
    ##########################################################################
    TimeseriesData = preprocess_data.PickleDataLoad(
        data_type=args.data, filename=args.filename, augment_test_data=False)
    train_dataset = TimeseriesData.batchify(
        args, TimeseriesData.trainData[:TimeseriesData.length], bsz=1)
    test_dataset = TimeseriesData.batchify(
        args, TimeseriesData.testData, bsz=1)

    ##########################################################################
    # Build the model
    ##########################################################################
    nfeatures = TimeseriesData.trainData.size(-1)
    model = model.RNNPredictor(
        rnn_type=args.model,
        enc_inp_size=nfeatures,
        rnn_inp_size=args.emsize,
        rnn_hid_size=args.nhid,
        dec_out_size=nfeatures,
        nlayers=args.nlayers,
        res_connection=args.res_connection).to(
        args.device)
    model.load_state_dict(checkpoint['state_dict'])
    # del checkpoint

    scores, predicted_scores, precisions, recalls, f_betas = list(
    ), list(), list(), list(), list()
    targets, mean_predictions, oneStep_predictions, Nstep_predictions = list(
    ), list(), list(), list()
    # For each channel in the dataset
    for channel_idx in range(nfeatures):
        ''' 1. Load mean and covariance if they are pre-calculated, if not calculate them. '''
        # Mean and covariance are calculated on train dataset.
        if 'means' in checkpoint.keys() and 'covs' in checkpoint.keys():
            # print('=> loading pre-calculated mean and covariance')
            mean, cov = checkpoint['means'][channel_idx], checkpoint['covs'][channel_idx]
        else:
            # print('=> calculating mean and covariance')
            mean, cov = fit_norm_distribution_param(
                args, model, train_dataset, channel_idx=channel_idx)

        score_predictor = None

        ''' 3. Calculate anomaly scores'''
        # Anomaly scores are calculated on the test dataset
        # given the mean and the covariance calculated on the train dataset
        # print('=> calculating anomaly scores')
        score, sorted_prediction, sorted_error, _, predicted_score = anomalyScore(
            args, model, test_dataset, mean, cov, score_predictor=score_predictor, channel_idx=channel_idx)

        ''' 4. Evaluate the result '''
        # The obtained anomaly scores are evaluated by measuring precision, recall, and f_beta scores
        # The precision, recall, f_beta scores are are calculated repeatedly,
        # sampling the threshold from 1 to the maximum anomaly score value, either equidistantly or logarithmically.
        # print('=> calculating precision, recall, and f_beta')
        precision, recall, f_beta, error_point, accuracy, threshold, τ = get_precision_recall(
             tau, mean, cov, sorted_error, args, score, num_samples=1000, beta=args.beta, label=TimeseriesData.testLabel.to(
                args.device))

        target = preprocess_data.reconstruct(
            test_dataset.cpu()[:, 0, channel_idx],
            TimeseriesData.mean[channel_idx],
            TimeseriesData.std[channel_idx]).numpy()
        mean_prediction = preprocess_data.reconstruct(
            sorted_prediction.mean(dim=1).cpu(),
            TimeseriesData.mean[channel_idx],
            TimeseriesData.std[channel_idx]).numpy()
        oneStep_prediction = preprocess_data.reconstruct(sorted_prediction[:, -1].cpu(
        ), TimeseriesData.mean[channel_idx], TimeseriesData.std[channel_idx]).numpy()
        Nstep_prediction = preprocess_data.reconstruct(sorted_prediction[:, 0].cpu(
        ), TimeseriesData.mean[channel_idx], TimeseriesData.std[channel_idx]).numpy()
        sorted_errors_mean = sorted_error.abs().mean(dim=1).cpu()
        sorted_errors_mean *= TimeseriesData.std[channel_idx]
        sorted_errors_mean = sorted_errors_mean.numpy()
        score = score.cpu()
        scores.append(score), targets.append(
            target), predicted_scores.append(predicted_score)
        mean_predictions.append(
            mean_prediction), oneStep_predictions.append(oneStep_prediction)
        Nstep_predictions.append(Nstep_prediction)
        precisions.append(precision), recalls.append(
            recall), f_betas.append(f_beta)

    save_dir = Path('result', args.data, args.filename).with_suffix('')
    save_dir.mkdir(parents=True, exist_ok=True)
    pickle.dump(targets, open(str(save_dir.joinpath('target.pkl')), 'wb'))
    pickle.dump(mean_predictions, open(
        str(save_dir.joinpath('mean_predictions.pkl')), 'wb'))
    pickle.dump(oneStep_predictions, open(
        str(save_dir.joinpath('oneStep_predictions.pkl')), 'wb'))
    pickle.dump(Nstep_predictions, open(
        str(save_dir.joinpath('Nstep_predictions.pkl')), 'wb'))
    pickle.dump(scores, open(str(save_dir.joinpath('score.pkl')), 'wb'))
    pickle.dump(predicted_scores, open(
        str(save_dir.joinpath('predicted_scores.pkl')), 'wb'))
    pickle.dump(precisions, open(
        str(save_dir.joinpath('precision.pkl')), 'wb'))
    pickle.dump(recalls, open(str(save_dir.joinpath('recall.pkl')), 'wb'))
    pickle.dump(f_betas, open(str(save_dir.joinpath('f_beta.pkl')), 'wb'))
    precision = precision.cpu().data.numpy()
    recall = recall.cpu().data.numpy()
    f_beta = f_beta.cpu().data.numpy()
    accuracy = accuracy[threshold >= τ]
    precision = precision[threshold >= τ]
    recall = recall[threshold >= τ]
    f_beta = f_beta[threshold >= τ]
    return accuracy[0], f_beta[0], precision[0], recall[0], τ, error_point


def generate_dataset():
    import requests
    from pathlib import Path
    import pickle
    from shutil import unpack_archive
    import numpy as np
    urls = dict()
    urls['ofdm'] = []
    # NoisePosition = []
    NoisePosition = np.loadtxt(Noise_Position_filepath, dtype=str)
    k = 0
    for dataname in urls:
        raw_dir = Path('dataset', dataname, 'raw')
        raw_dir.mkdir(parents=True, exist_ok=True)
        for url in urls[dataname]:
            filename = raw_dir.joinpath(Path(url).name)
            # print('Downloading', url)
            resp = requests.get(url)
            filename.write_bytes(resp.content)
            if filename.suffix == '':
                filename.rename(filename.with_suffix('.txt'))
            # print('Saving to', filename.with_suffix('.txt'))
            if filename.suffix == '.zip':
                # print('Extracting to', filename)
                unpack_archive(str(filename), extract_dir=str(raw_dir))

        for filepath in raw_dir.glob('*.txt'):
            with open(str(filepath)) as f:
                # Label anomaly points as 1 in the dataset
                labeled_data = []
                for i, line in enumerate(f):
                    tokens = [float(token) for token in line.split()]
                    if filepath.name == 'NoiseSymbol.txt':
                        if i == int(NoisePosition[k])-1:
                            tokens.append(1.0)
                            if k < len(NoisePosition)-1:
                                k += 1
                        else:
                            tokens.append(0.0)
                    labeled_data.append(tokens)
                # np.savetxt('label_data.txt', labeled_data)

                # Save the labeled dataset as .pkl extension
                labeled_whole_dir = raw_dir.parent.joinpath('labeled', 'whole')
                labeled_whole_dir.mkdir(parents=True, exist_ok=True)
                with open(str(labeled_whole_dir.joinpath(filepath.name).with_suffix('.pkl')), 'wb') as pkl:
                    pickle.dump(labeled_data, pkl)

                # Divide the labeled dataset into trainset and testset, then
                # save them
                labeled_train_dir = raw_dir.parent.joinpath('labeled', 'train')
                labeled_train_dir.mkdir(parents=True, exist_ok=True)
                labeled_test_dir = raw_dir.parent.joinpath('labeled', 'test')
                labeled_test_dir.mkdir(parents=True, exist_ok=True)
                if filepath.name == 'NoiseSymbol.txt':
                    # with open(str(labeled_train_dir.joinpath(filepath.name).with_suffix('.pkl')), 'wb') as pkl:
                    #     pickle.dump(labeled_data[1000:], pkl)
                    with open(str(labeled_test_dir.joinpath(filepath.name).with_suffix('.pkl')), 'wb') as pkl:
                        pickle.dump(labeled_data, pkl)


def SP(bits):
    return bits.reshape((len(dataCarriers), mu))
    # return bits.reshape(K, mu)


def Modulation(bits):
    return np.array([mapping_table[tuple(b)] for b in bits])


def OFDM_symbol(QAM_payload):
    symbol = np.zeros(K, dtype=complex)  # the overall K subcarriers
    # symbol[pilotCarriers] = pilotValue  # allocate the pilot subcarriers
    symbol = QAM_payload  # allocate the pilot subcarriers
    return symbol


def IDFT(OFDM_data):
    # np.fft.ifft(OFDM_data)*np.sqrt(K)  (lJS)
    return np.fft.ifft(OFDM_data) * np.sqrt(K)


def addCP(OFDM_time):
    cp = OFDM_time[-CP:]  # take the last CP samples ...
    return np.hstack([cp, OFDM_time])  # ... and add them to the beginning


# construct the another version is including impulse noise(LJS)
def channel_BG(signal, channelResponse, SNRdb):
    # Bernoulli-Gaussian channel          # lJS
    # IGR = 50  # impulse gaussian ratio
    prob = 0.001  # prob
    # prob = np.random.uniform(0.001, 0.01)
    convolved = np.convolve(signal, channelResponse)
    signal_power = np.mean(abs(convolved**2))
    sigma2 = signal_power * 10**(-SNRdb / 10)      # (signal_power/2)  (LJS)
    # sigma3 = sigma2 * IGR
    # sigma3 = 15
    sigma3 = signal_power * \
        np.random.uniform(10*np.log10(3.1622776602), 10*np.log10(10))
    Gaussian = np.random.randn(*convolved.shape) + 1j * \
        np.random.randn(*convolved.shape)
    power1 = np.zeros([*convolved.shape])
    power2 = np.zeros([*convolved.shape])
    noise_position = []
    noise_position_res = []
    # print('Channel Length :', len(channelResponse))
    # print('Bits Length :', len(convolved))
    # print('IGR :', IGR)
    # print('Signal Power :', signal_power)
    # print('AWGN Power :', sigma2)
    # print('Impulse Power :', sigma3)
    # print('SNR:', SNRdb)
    # print('Impulse Probability :', prob*100, '%')
    for i in range(*convolved.shape):
        power1[i] = np.sqrt(sigma2 / 2)
        power2[i] = np.sqrt(sigma2 / 2)
    for i in range(*convolved.shape):
        k = np.random.rand()
        n = np.random.binomial(n=1, p=0.5)
        c = np.random.randint(low=2, high=4)
        if k <= prob:
            if n == 1:
                power1[i] = np.sqrt(sigma3 / 2)
                power2[i] = np.sqrt(sigma3 / 2)
                # print('impulse_position_single =', i + 1)
                j = i + 1
                # position = 'single ' + str(j)
                position = str(j)
                noise_position.append(position)
            else:
                if i > 0:
                    if (i+c) < len(convolved):
                        for ii in range(c):
                            power1[ii+i] = np.sqrt(sigma3 / 2)
                            power2[ii+i] = np.sqrt(sigma3 / 2)
                            # power1[i+1] = np.sqrt(sigma3 / 2)
                            # power2[i+1] = np.sqrt(sigma3 / 2)
                            # power1[i+2] = np.sqrt(sigma3 / 2)
                            # power2[i+2] = np.sqrt(sigma3 / 2)
                            # power1[i+3] = np.sqrt(sigma3 / 2)
                            # power2[i+3] = np.sqrt(sigma3 / 2)
                            # power1[i+4] = np.sqrt(sigma3 / 2)
                            # power2[i+4] = np.sqrt(sigma3 / 2)
                            # print('impulse_position_mutiple =', i + 1)
                            j = i + ii + 1
                            # position = 'mutiple ' + str(j)
                            position = str(j)
                            noise_position.append(position)
    [noise_position_res.append(x)
     for x in noise_position if x not in noise_position_res]
    # print('Real anomaly is on the', noise_position_res)
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
            (noise_symbol_image[i]**2) + (noise_symbol_real[i]**2))
    noise_symbol = np.array(noise_symbol)
    # np.savetxt('Noise.txt', noise_BG)
    # np.savetxt('NoiseReal.txt', noise_BG.real)
    # np.savetxt('NoiseImag.txt', noise_BG.imag)
    # np.savetxt('NoiseSymbolReal.txt', noise_BG.real + convolved.real)
    # np.savetxt('NoiseSymbolImag.txt', noise_BG.imag + convolved.imag)
    np.savetxt(Noise_Symbol_filepath, noise_symbol)
    np.savetxt(Noise_Position_filepath, noise_position_res, fmt="%s")
    return noise_BG + convolved, noise_position_res, signal_power


def removeCP(signal):
    return signal[CP:(CP + K)]


def DFT(OFDM_RX):
    return np.fft.fft(OFDM_RX / np.sqrt(K))


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
#     Hest = Hest_abs * np.exp(1j * Hest_phase)
#     return Hest


def equalize(OFDM_demod, Hest):
    Hest = np.linalg.inv(Hest)
    # OFDM_demod = np.expand_dims(OFDM_demod, axis=1)
    return np.matmul(Hest, OFDM_demod)


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
            h_response = np.asarray(numbers_float[0:int(len(numbers_float) / 2)]) + \
                1j * \
                np.asarray(
                             numbers_float[int(len(numbers_float) / 2):len(numbers_float)])
            channel_response_set_test.append(h_response)

for x in range(total_epochs - valid_epochs):
    checkpoint = []
    result = []
    OFDM_RX_CHANNEL = 0
    OFDM_RX_CHANNEL_INFORMATION = []
    bestBER = 0
    besttau = 0
    channel_response = channel_response_set_test[np.random.randint(
        0, len(channel_response_set_test))]
    # H_exact = np.fft.fft(channel_response, K)
    for i in range(1024):
        OFDM_RX_CHANNEL = 0
        for ithChannel in range(16):
            OFDM_RX_CHANNEL += channel_response[ithChannel] * \
                np.exp((-1j)*(2*np.pi*ithChannel*i)/1024)
        OFDM_RX_CHANNEL_INFORMATION.append(OFDM_RX_CHANNEL)
    Hest = np.diag(OFDM_RX_CHANNEL_INFORMATION)
    bits = np.random.binomial(n=1, p=0.5, size=(payloadBits_per_OFDM, ))
    # ofdm_simulate_BG(bits, SNRdb)
    bits_SP = SP(bits)
    QAM = Modulation(bits_SP)
    OFDM_data = OFDM_symbol(QAM)
    OFDM_time = IDFT(OFDM_data)
    OFDM_withCP = addCP(OFDM_time)
    OFDM_TX = OFDM_withCP
    for correct_data_check in range(1000):
        # signal to noise-ratio in dB at the receiver
        # SNRdb = np.random.randint(5, 26)
        SNRdb = 30
        OFDM_RX, datacheck, signal_power = channel_BG(
            OFDM_TX, channel_response, SNRdb)
        if len(datacheck) > 1:
            break
    generate_dataset()
    for tau in range(10, 100, 10):
        accuracy, fbeta, precision, recall, τ, error_point = anomaly_detection(
            tau)
        Unknown_OFDM_RX = np.copy(OFDM_RX)
        # for impulse in error_point:
        #     # Unknown_OFDM_RX[int(impulse)-1] = 0
        #     clipper_threshold = 0.5
        #     norm = np.abs(Unknown_OFDM_RX)
        #     angle = np.angle(Unknown_OFDM_RX)
        #     if norm[int(impulse)-1] >= clipper_threshold:
        #         Unknown_OFDM_RX.real[int(
        #             impulse)-1] = clipper_threshold * np.cos(angle[int(impulse)-1])
        #         Unknown_OFDM_RX.imag[int(
        #             impulse)-1] = clipper_threshold * np.sin(angle[int(impulse)-1])
        OFDM_RX_noCP = removeCP(Unknown_OFDM_RX)
        OFDM_demod = DFT(OFDM_RX_noCP)
        equalized_Hest = equalize(OFDM_demod, Hest)
        QAM_est = get_payload(equalized_Hest)
        PS_est, hardDecision = Demapping(QAM_est)
        bits_est = PS(PS_est)
        BER = np.sum(abs(bits - bits_est)) / len(bits)
        if bestBER == 0:
            bestBER = BER
            besttau = tau
        if bestBER > BER:
            bestBER = BER
            besttau = tau
        if BER == 0:
            break
        # print(error_point)
        # print(datacheck)
        # print(tau)
    #     print(BER)
    # print(bestBER)
    # print(besttau)
    total_accuracy += accuracy
    total_fbeta += fbeta
    total_precision += precision
    total_recall += recall
    total_τ += besttau
    total_signal_power += signal_power
    total_BER += bestBER
    total_impulse += len(datacheck)
    valid_epochs += 1
    avg_accuracy = (total_accuracy / valid_epochs) * 100
    avg_fbeta = (total_fbeta / valid_epochs) * 100
    avg_precision = (total_precision / valid_epochs) * 100
    avg_recall = (total_recall / valid_epochs) * 100
    avg_τ = (total_τ / valid_epochs)
    avg_signal_power = (total_signal_power / valid_epochs)
    avg_BER = (total_BER / valid_epochs)
    avg_impulse_prob = (total_impulse/(1055*valid_epochs)) * 100
    # index_SNR = SNRdb - 5
    # checkpoint_SNR[index_SNR, 1] += accuracy  # total_accuracy_index_SNR
    # checkpoint_SNR[index_SNR, 2] += fbeta  # total_fbeta_index_SNR
    # checkpoint_SNR[index_SNR, 3] += 1  # total_avg_index_SNR
    checkpoint.extend([valid_epochs,
                       total_accuracy,
                       total_fbeta,
                       total_precision,
                       total_recall,
                       total_τ,
                       total_signal_power,
                       total_BER,
                       total_impulse])
    np.savetxt('checkpoint.txt', checkpoint)
    # np.savetxt('checkpoint_SNR.txt', checkpoint_SNR)
    if valid_epochs % 1 == 0:
        print('-' * 120)
        print('Ac:', accuracy, ' F-beta:',
              fbeta, ' Pr:', precision, ' Rc:', recall, ' BER:', bestBER, ' IN_prob:', (len(datacheck)/1055))
        print('-' * 120)
        print(
            'epoch '
            + str(valid_epochs)
            + '\navg.accuracy = '
            + str(avg_accuracy)
            + ' %\navg.f-beta = '
            + str(avg_fbeta)
            + ' %\navg.precision = '
            + str(avg_precision)
            + ' %\navg.recall = '
            + str(avg_recall)
            + ' %\navg.τ = '
            + str(avg_τ)
            + ' \navg.signal power = '
            + str(avg_signal_power)
            + ' \navg.BER = '
            + str(avg_BER)
            + ' \navg.impulse_prob = '
            + str(avg_impulse_prob)
            + ' %')

result.extend(['accuracy',
               'fbeta',
               'precision',
               'recall',
               'τ',
               'signal_power',
               'BER',
               'impulse_prob',
               avg_accuracy,
               avg_fbeta,
               avg_precision,
               avg_recall,
               avg_τ,
               avg_signal_power,
               avg_BER,
               avg_impulse_prob])

# for SNR in range(21):
#     checkpoint_SNR[SNR, 1] = checkpoint_SNR[SNR, 1] / checkpoint_SNR[SNR, 3]
#     checkpoint_SNR[SNR, 2] = checkpoint_SNR[SNR, 2] / checkpoint_SNR[SNR, 3]

if valid_epochs == total_epochs:
    np.savetxt('resultSNR30.txt', np.reshape(
        result, (8, 2), order='F'), fmt="%s")
    # np.savetxt('SNR_resultSNR35.txt', checkpoint_SNR, fmt="%s")
    os.remove('checkpoint.txt')
    # os.remove('checkpoint_SNR.txt')
