from __future__ import division
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()
print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # BE QUIET!!!!

# Part I : Relevant information

K = 64  # subcarriers = K
CP = K // 4
P = 64  # number of pilot carriers per OFDM block
# pilotValue = 1+1j
allCarriers = np.arange(K)  # indices of all subcarriers ([0, 1, ... K-1])
pilotCarriers = allCarriers[::K // P]  # Pilots is every (K/P)th carrier.
# pilotCarriers = np.hstack([pilotCarriers, np.array([allCarriers[-1]])])
# P = P+1
dataCarriers = np.delete(allCarriers, pilotCarriers)
mu = 2    # one symbol combined with two bits for QAM or QPSK (LJS)
# number of payload bits per OFDM symbol
payloadBits_per_OFDM = len(dataCarriers) * mu

# payloadbits per OFDM version 2 (decided by how many data carriers per OFDM , LJS)
payloadBits_per_OFDM = K * mu

SNRdb = 25  # signal to noise-ratio in dB at the receiver

mapping_table = {
    (0, 0): -1 - 1j,
    (0, 1): -1 + 1j,
    (1, 0): 1 - 1j,
    (1, 1): 1 + 1j,
}

demapping_table = {v: k for k, v in mapping_table.items()}

# Part II : Define functions


def Modulation(bits):
    bit_r = bits.reshape((int(len(bits) / mu), mu))
    # This is just for QAM modulation
    return (2 * bit_r[:, 0] - 1) + 1j * (2 * bit_r[:, 1] - 1)


def OFDM_symbol(Data, pilot_flag):
    symbol = np.zeros(K, dtype=complex)  # the overall K subcarriers
    # symbol = np.zeros(K)
    symbol[pilotCarriers] = pilotValue  # allocate the pilot subcarriers
    symbol[dataCarriers] = Data  # allocate the pilot subcarriers
    return symbol


def IDFT(OFDM_data):
    return np.fft.ifft(OFDM_data)   # np.fft.ifft(OFDM_data)*np.sqrt(K)  (lJS)


def addCP(OFDM_time):
    cp = OFDM_time[-CP:]  # take the last CP samples ...
    return np.hstack([cp, OFDM_time])  # ... and add them to the beginning


# construct the another version is including impulse noise(LJS)
def channel_AWGN(signal, channelResponse, SNRdb):
    # AWGN channel
    convolved = np.convolve(signal, channelResponse)
    signal_power = np.mean(abs(convolved**2))
    sigma2 = signal_power * 10**(-SNRdb / 10)
    noise = np.sqrt(sigma2 / 2) * (np.random.randn(
                                                   * convolved.shape) + 1j * np.random.randn(*convolved.shape))
    return convolved + noise


# construct the another version is including impulse noise(LJS)
def channel_BG(signal, channelResponse, SNRdb):
    # Bernoulli-Gaussian channel          # lJS
    prob = 0.001  # prob
    convolved = np.convolve(signal, channelResponse)
    signal_power = np.mean(abs(convolved**2))
    sigma2 = signal_power * 10**(-SNRdb / 10)      # (signal_power/2)  (LJS)
    sigma3 = 2
    Gaussian = np.random.randn(*convolved.shape) + \
        1j * np.random.randn(*convolved.shape)
    power1 = np.zeros([*convolved.shape])
    for i in range(*convolved.shape):
        k = np.random.rand()
        if k > prob:
            power1[i] = np.sqrt(sigma2 / 2)
        if k <= prob:
            power1[i] = np.sqrt(sigma3 / 2)
    power2 = np.zeros([*convolved.shape])
    for i in range(*convolved.shape):
        k = np.random.rand()
        if k > prob:
            power2[i] = np.sqrt(sigma2 / 2)
        if k <= prob:
            power2[i] = np.sqrt(sigma3 / 2)
    noise1 = np.multiply(power1, Gaussian.real)
    noise2 = np.multiply(power2, Gaussian.imag)
    noise_BG = np.zeros([*convolved.shape]).astype(complex)
    noise_BG.real = noise1
    noise_BG.imag = noise2
    np.savetxt('Noise.txt', noise_BG)
    return convolved + noise_BG


def removeCP(signal):
    return signal[CP:(CP + K)]


def equalize(OFDM_demod, Hest):   # *(LJS)
    return OFDM_demod / Hest


def get_payload(equalized):       # **(LJS)
    return equalized[dataCarriers]


def PS(bits):                     # *(LJS)
    return bits.reshape((-1,))


def clipper(OFDM_RX_noCP):         # LJS
    clipper_threshold = 0.5         # initial value = 0.1 : 0.05 : 0.5
    norm = np.abs(OFDM_RX_noCP)
    angle = np.angle(OFDM_RX_noCP)
    for i in range(*OFDM_RX_noCP.shape):
        if norm[i] >= clipper_threshold:
            OFDM_RX_noCP.real[i] = clipper_threshold * np.cos(angle[i])
            OFDM_RX_noCP.imag[i] = clipper_threshold * np.sin(angle[i])
    return OFDM_RX_noCP


def ofdm_simulate_AWGN(codeword, channelResponse, SNRdb):       # LJS
    OFDM_data = np.zeros(K, dtype=complex)
    OFDM_data[allCarriers] = pilotValue
    OFDM_time = IDFT(OFDM_data)
    OFDM_withCP = addCP(OFDM_time)
    OFDM_TX = OFDM_withCP
    OFDM_RX = channel_AWGN(OFDM_TX, channelResponse, SNRdb)
    OFDM_RX_noCP = removeCP(OFDM_RX)
    # ----- target inputs ---
    symbol = np.zeros(K, dtype=complex)
    codeword_qam = Modulation(codeword)
    symbol[np.arange(K)] = codeword_qam
    OFDM_data_codeword = symbol
    OFDM_time_codeword = np.fft.ifft(OFDM_data_codeword)
    OFDM_withCP_cordword = addCP(OFDM_time_codeword)
    OFDM_RX_codeword = channel_AWGN(
        OFDM_withCP_cordword, channelResponse, SNRdb)
    OFDM_RX_noCP_codeword = removeCP(OFDM_RX_codeword)
    return np.concatenate((np.concatenate((np.real(OFDM_RX_noCP), np.imag(OFDM_RX_noCP))), np.concatenate((np.real(OFDM_RX_noCP_codeword), np.imag(OFDM_RX_noCP_codeword))))), abs(channelResponse)


def ofdm_simulate_BG(codeword, channelResponse, SNRdb):       # LJS
    OFDM_data = np.zeros(K, dtype=complex)
    OFDM_data[allCarriers] = pilotValue
    OFDM_time = IDFT(OFDM_data)
    OFDM_withCP = addCP(OFDM_time)
    OFDM_TX = OFDM_withCP
    OFDM_RX = channel_BG(OFDM_TX, channelResponse, SNRdb)
    OFDM_RX_noCP = removeCP(OFDM_RX)
    # ----- target inputs ---
    symbol = np.zeros(K, dtype=complex)
    codeword_qam = Modulation(codeword)
    symbol[np.arange(K)] = codeword_qam
    OFDM_data_codeword = symbol
    OFDM_time_codeword = np.fft.ifft(OFDM_data_codeword)
    OFDM_withCP_cordword = addCP(OFDM_time_codeword)
    OFDM_RX_codeword = channel_BG(OFDM_withCP_cordword, channelResponse, SNRdb)
    OFDM_RX_noCP_codeword = removeCP(OFDM_RX_codeword)
    return np.concatenate((np.concatenate((np.real(OFDM_RX_noCP), np.imag(OFDM_RX_noCP))), np.concatenate((np.real(OFDM_RX_noCP_codeword), np.imag(OFDM_RX_noCP_codeword))))), abs(channelResponse)


def ofdm_simulate_clip(codeword, channelResponse, SNRdb):       # LJS
    OFDM_data = np.zeros(K, dtype=complex)
    OFDM_data[allCarriers] = pilotValue
    OFDM_time = IDFT(OFDM_data)
    OFDM_withCP = addCP(OFDM_time)
    OFDM_TX = OFDM_withCP
    OFDM_RX = channel(OFDM_TX, channelResponse, SNRdb)
    OFDM_RX_noCP = removeCP(OFDM_RX)
    OFDM_RX_noCP_clip = clipper(OFDM_RX_noCP)  # LJS
    # ----- target inputs ---
    symbol = np.zeros(K, dtype=complex)
    codeword_qam = Modulation(codeword)
    symbol[np.arange(K)] = codeword_qam
    OFDM_data_codeword = symbol
    OFDM_time_codeword = np.fft.ifft(OFDM_data_codeword)
    OFDM_withCP_cordword = addCP(OFDM_time_codeword)
    OFDM_RX_codeword = channel(OFDM_withCP_cordword, channelResponse, SNRdb)
    OFDM_RX_noCP_codeword = removeCP(OFDM_RX_codeword)
    OFDM_RX_noCP_codeword_clip = clipper(OFDM_RX_noCP_codeword)  # LJS
    return np.concatenate((np.concatenate((np.real(OFDM_RX_noCP_clip), np.imag(OFDM_RX_noCP_clip))), np.concatenate((np.real(OFDM_RX_noCP_codeword_clip), np.imag(OFDM_RX_noCP_codeword_clip))))), abs(channelResponse)


def decision(y_pred_np, batch_y):       # LJS
    size = np.shape(batch_y)
    y_pred_trans = np.zeros([size[0], size[1]])
    for k in range(size[0]):
        for p in range(size[1]):
            if y_pred_np[k, p] > 0.5:
                y_pred_trans[k, p] = 1
            if y_pred_np[k, p] < 0.5:
                y_pred_trans[k, p] = 0
            if y_pred_np[k, p] == 0.5:
                y_pred_trans[k, p] = np.random.randint(0, 2)
    count = 0
    for i in range(size[0]):
        for j in range(size[1]):
            if y_pred_trans[i, j] != batch_y[i, j]:
                count = count + 1
    BER = count / (size[0] * size[1])
    return BER


Pilot_file_name = 'Pilot_' + str(P)   # Here file name is "Pilot_64" (LJS)
if os.path.isfile(Pilot_file_name):
    print('Load Training Pilots txt')
    # load file
    bits = np.loadtxt(Pilot_file_name, delimiter=',')
else:
    # write file
    bits = np.random.binomial(n=1, p=0.5, size=(K * mu, ))
    np.savetxt(Pilot_file_name, bits, delimiter=',')

pilotValue = Modulation(bits)

# Part III : Deep learning training

# %%time
# Training parameters
# training_epochs = 20
# batch_size = 256
display_step = 100  # 5
test_step = 200
cost_step = 25  # LJS
# examples_to_show = 10
# Network Parameters
n_hidden_1 = 500
n_hidden_2 = 250  # 1st layer num features
n_hidden_3 = 120  # 2nd layer num features
n_input = 256
n_output = 16  # every 16 bit are predicted by a model
# tf Graph input (only pictures)
X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_output])

weights = {
    'encoder_h1': tf.Variable(tf.truncated_normal([n_input, n_hidden_1], stddev=0.1)),
    'encoder_h2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2], stddev=0.1)),
    'encoder_h3': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_3], stddev=0.1)),
    'encoder_h4': tf.Variable(tf.truncated_normal([n_hidden_3, n_output], stddev=0.1)),
}
biases = {
    'encoder_b1': tf.Variable(tf.truncated_normal([n_hidden_1], stddev=0.1)),
    'encoder_b2': tf.Variable(tf.truncated_normal([n_hidden_2], stddev=0.1)),
    'encoder_b3': tf.Variable(tf.truncated_normal([n_hidden_3], stddev=0.1)),
    'encoder_b4': tf.Variable(tf.truncated_normal([n_output], stddev=0.1)),

}

# Encoder Hidden layer with sigmoid activation # 1
# layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']), biases['encoder_b1']))
layer_1 = tf.nn.relu(
    tf.add(tf.matmul(X, weights['encoder_h1']), biases['encoder_b1']))
layer_2 = tf.nn.relu(
    tf.add(tf.matmul(layer_1, weights['encoder_h2']), biases['encoder_b2']))
layer_3 = tf.nn.relu(
    tf.add(tf.matmul(layer_2, weights['encoder_h3']), biases['encoder_b3']))
layer_4 = tf.nn.sigmoid(
    tf.add(tf.matmul(layer_3, weights['encoder_h4']), biases['encoder_b4']))

y_pred = layer_4
# Targets (Labels) are the input data.
y_true = Y

# Define loss and optimizer, minimize the squared error
cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
learning_rate = tf.placeholder(tf.float32, shape=[])
optimizer = tf.train.RMSPropOptimizer(
    learning_rate=learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

# Start Training
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
# The H information set
H_folder_train = '../H_dataset/train/'
H_folder_test = '../H_dataset/test/'
train_idx_low = 1
train_idx_high = 301
test_idx_low = 301
test_idx_high = 401

# =================== Saving Channel conditions to a large matrix ================ #
channel_response_set_train = []
for train_idx in range(train_idx_low, train_idx_high):
    print("Processing the train", train_idx, "th document")
    H_file = H_folder_train + str(train_idx) + '.txt'
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
            channel_response_set_train.append(h_response)

channel_response_set_test = []
for test_idx in range(test_idx_low, test_idx_high):
    print("Processing the test", test_idx, "th document")
    H_file = H_folder_test + str(test_idx) + '.txt'
    with open(H_file) as f:
        for line in f:
            numbers_str = line.split()
            numbers_float = [float(x) for x in numbers_str]
            h_response = np.asarray(numbers_float[0:int(len(numbers_float) / 2)]) +\
                1j * \
                np.asarray(
                    numbers_float[int(len(numbers_float) / 2):len(numbers_float)])
            channel_response_set_test.append(h_response)
            # print(np.shape(channel_response_set_test))

print('length of training channel response is', len(channel_response_set_train))
print('length of testing channel response is', len(channel_response_set_test))
print('shape of training channel response is',
      np.shape(channel_response_set_train))
print('shape of testing channel response is',
      np.shape(channel_response_set_test))
#=================== end ================ #

with tf.Session(config=config) as sess:
    sess.run(init)
    training_epochs = 5000  # 20000
    learning_rate_current = 0.001  # 0.01
    training_cost = np.zeros([1, 200])  # LJS
    for epoch in range(training_epochs):
        print("========================================")
        print('Processing the', epoch + 1, 'th epoch')
#         if epoch > 0 and epoch % 500 == 0: # 2000
#             learning_rate_current = learning_rate_current/5
        avg_cost = 0.0
        total_batch = 50  # 50

        if epoch <= 2000:
            print('learning rate', learning_rate_current)
            for index_m in range(total_batch):
                input_samples = []
                input_labels = []
                for index_k in range(0, 500):  # 1000
                    bits = np.random.binomial(
                        n=1, p=0.5, size=(payloadBits_per_OFDM, ))
                    channel_response = channel_response_set_train[np.random.randint(
                        0, len(channel_response_set_train))]
                    signal_output, para = ofdm_simulate_AWGN(
                        bits, channel_response, SNRdb)
                    input_labels.append(bits[16:32])
                    input_samples.append(signal_output)

                batch_x = np.asarray(input_samples)
                batch_y = np.asarray(input_labels)
                _, c = sess.run([optimizer, cost], feed_dict={X: batch_x,  # input
                                                              Y: batch_y,  # labels
                                                              learning_rate: learning_rate_current})
                avg_cost += c / total_batch

        if epoch > 2000:
            if epoch % cost_step == 0:
                learning_rate_current = learning_rate_current - 0.00001
                if learning_rate_current <= 0.00001:
                    learning_rate_current = 0.00001
                print('learning rate', learning_rate_current)
            for index_m in range(total_batch):
                input_samples = []
                input_labels = []
                for index_k in range(0, 500):  # 1000
                    bits = np.random.binomial(
                        n=1, p=0.5, size=(payloadBits_per_OFDM, ))
                    channel_response = channel_response_set_train[np.random.randint(
                        0, len(channel_response_set_train))]
                    signal_output, para = ofdm_simulate_BG(
                        bits, channel_response, SNRdb)
                    input_labels.append(bits[16:32])
                    input_samples.append(signal_output)
                batch_x = np.asarray(input_samples)
                batch_y = np.asarray(input_labels)
                _, c = sess.run([optimizer, cost], feed_dict={X: batch_x,  # input
                                                              Y: batch_y,  # labels
                                                              learning_rate: learning_rate_current})
                avg_cost += c / total_batch

        if epoch % cost_step == 0:
            training_cost[0, epoch // cost_step] = avg_cost

#         if learning_rate_current <= 0.00001 :
#             break

        if epoch % display_step == 0:  # == 0
            print("epoch:", '%04d' % (epoch + 1))
            print("cost=", "{:.9f}".format(avg_cost))
            # print ("========================================")
            input_samples_test = []
            input_labels_test = []
            test_number = 1  # 1000
            # set test channel response for this epoch
            if epoch % test_step == 0:
                # print ("========================================")
                print("This is a Big Test Set ")
                test_number = 1  # 10000
            for i in range(0, test_number):
                bits = np.random.binomial(
                    n=1, p=0.5, size=(payloadBits_per_OFDM, ))
                channel_response = channel_response_set_test[np.random.randint(
                    0, len(channel_response_set_test))]
                signal_output, para = ofdm_simulate_BG(
                    bits, channel_response, SNRdb)
                input_labels_test.append(bits[16:32])
                input_samples_test.append(signal_output)
                print('b')

            batch_x = np.asarray(input_samples_test)
            batch_y = np.asarray(input_labels_test)
            # encode_decode = sess.run(y_pred, feed_dict = {X:batch_x})
            mean_error = tf.reduce_mean(abs(y_pred - batch_y))
            mean_error_rate = 1 - tf.reduce_mean(tf.reduce_mean(tf.to_float(tf.equal(
                tf.sign(y_pred - 0.5), tf.cast(tf.sign(batch_y - 0.5), tf.float32))), 1))
            # print("OFDM Detection QAM output number is", n_output, ",SNR = ", SNRdb, ",Num Pilot = ", P,", prediction and the mean error on test set are:", mean_error.eval({X:batch_x}), mean_error_rate.eval({X:batch_x}))
            print("OFDM Detection QAM output number is", n_output)
            print("SNR = ", SNRdb)
            print("Num Pilot", P)
            print("prediction and the mean error on test set are:",
                  mean_error.eval({X: batch_x}), mean_error_rate.eval({X: batch_x}))
            y_pred_np = y_pred.eval({X: batch_x})  # LJS
            BER = decision(y_pred_np, batch_y)     # LJS
            print('BER :', BER)                  # LJS

            batch_x = np.asarray(input_samples)
            batch_y = np.asarray(input_labels)
            # encode_decode = sess.run(y_pred, feed_dict = {X:batch_x})
            mean_error = tf.reduce_mean(abs(y_pred - batch_y))
            mean_error_rate = 1 - tf.reduce_mean(tf.reduce_mean(tf.to_float(tf.equal(
                tf.sign(y_pred - 0.5), tf.cast(tf.sign(batch_y - 0.5), tf.float32))), 1))
            print("Prediction and the mean error on train set are:",
                  mean_error.eval({X: batch_x}), mean_error_rate.eval({X: batch_x}))
            # print ("========================================")
    print("Total epochs cost : \n", training_cost)  # LJS
    print("optimization finished")
    # save the model
    saver = tf.train.Saver()
    save_path = saver.save(sess, "net_Impulse_thesis/epochs_5000.ckpt")
    print("Save to path: ", save_path)

# Part IV : Testing the model

# %%time
with tf.Session() as sess:
    saver.restore(sess, "net_Impulse_thesis/epochs_5000.ckpt")
    for k in range(5, 31):  # (5,26)
        SNRdB = k   # SNRdB=k
        input_samples_test = []
        input_labels_test = []
        test_number = 10000         # 1000
        for i in range(0, test_number):
            bits = np.random.binomial(
                n=1, p=0.5, size=(payloadBits_per_OFDM, ))
            channel_response = channel_response_set_test[np.random.randint(
                0, len(channel_response_set_test))]
            signal_output, para = ofdm_simulate_BG(
                bits, channel_response, SNRdB)
            input_labels_test.append(bits[16:32])
            input_samples_test.append(signal_output)

        batch_x = np.asarray(input_samples_test)
        batch_y = np.asarray(input_labels_test)
        y_pred_np = y_pred.eval({X: batch_x})            # LJS
        BER = decision(y_pred_np, batch_y)               # LJS
        print('SNR = ', SNRdB, 'BER :', BER)             # LJS
