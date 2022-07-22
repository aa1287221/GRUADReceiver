import numpy as np
from matplotlib import pyplot as plt
from math import log
BER = np.abs(np.loadtxt('error.txt'))
MEAN = np.loadtxt('mean.txt')
# BER = [
#     0.22344474860335103,
#     0.1551048044692742,
#     0.12530491620111742,
#     0.11669916201117268,
#     0.11278150837988819,
#     0.11130162011173199]
# SNR = [5, 10, 15, 20, 25, 30]
plt.figure(figsize=(7, 4))
OFDM = plt.hist(BER)
# MEAN = plt.plot(MEAN)
# plt.xlim(0, 35)
# plt.xticks([0, 5, 10, 15, 20, 25, 30, 35])
# plt.yscale('log')
# plt.yticks([10**(-2), 10**(-1), 10**0])
plt.grid(True)
plt.yscale('log')
plt.title('Variance',
          fontsize=14, fontweight='bold')
plt.xlabel('Error', fontsize=10, fontweight='bold')
plt.ylabel('Frequency', fontsize=10, fontweight='bold')
# plt.setp(OFDM, marker='o',
#          linestyle='-', markersize=5, linewidth=1)
# plt.setp(MEAN, color='red', marker='o',
#          linestyle='-', markersize=5, linewidth=1)
plt.savefig('OFDM Simulate on 10000 Blocks.jpg')
plt.cla()
# Fb38 = plt.plot(SNR38, FBETA38, label='Threshold 38%')
# Fb39 = plt.plot(SNR39, FBETA39, label='Threshold 39%')
# Fb40 = plt.plot(SNR40, FBETA40, label='Threshold 40%')
# Fb41 = plt.plot(SNR41, FBETA41, label='Threshold 41%')
# plt.ylim(0.89, 0.95)
# plt.xlim(0, 30)
# plt.xticks([0, 5, 10, 15, 20, 25, 30])
# plt.grid(True)
# plt.title('OFDM Simulate on 10000 Blocks Validation',
#           fontsize=18, fontweight='bold')
# plt.ylabel('F-0.1 Score', fontsize=10, fontweight='bold')
# plt.xlabel('SNR(Eb/N0)', fontsize=10, fontweight='bold')
# plt.setp(Fb38, marker='o',
#          linestyle='-', markersize=5, linewidth=1)
# plt.setp(Fb39, color='red', marker='*',
#          linestyle='-', markersize=5, linewidth=1)
# plt.setp(Fb40, color='purple', marker='P',
#          linestyle='-', markersize=5, linewidth=1)
# plt.setp(Fb41, color='brown', marker='p',
#          linestyle='-', markersize=5, linewidth=1)
# plt.legend()
# plt.savefig('OFDM Simulate on 10000 Blocks_F-0.1.jpg')
# plt.cla()
# Pr = plt.plot([5,
#                10,
#                15,
#                20,
#                25],
#               [0.9884194885311648,
#                0.9913594731761958,
#               0.9925750170802581,
#               0.9925373739657341,
#               0.9946216550577547])
# plt.ylim(0.987, 0.995)
# plt.xlim(0, 30)
# plt.xticks([0, 5, 10, 15, 20, 25, 30])
# plt.grid(True)
# plt.title('OFDM Simulate on 10000 Blocks Test',
#           fontsize=18, fontweight='bold')
# plt.ylabel('Precision', fontsize=10, fontweight='bold')
# plt.xlabel('SNR(Eb/N0)', fontsize=10, fontweight='bold')
# plt.setp(Pr, color='purple', marker='o',
#          linestyle='--', markersize=5, linewidth=1)
# plt.savefig('OFDM Simulate on 10000 Blocks_Precision.jpg')
# plt.cla()
# Rc = plt.plot([5,
#                10,
#                15,
#                20,
#                25],
#               [0.4065132861509919,
#                0.4297448675628752,
#               0.44822052444405855,
#               0.4415464423671365,
#               0.45631645740456875])
# plt.ylim(0.40, 0.46)
# plt.xlim(0, 30)
# plt.xticks([0, 5, 10, 15, 20, 25, 30])
# plt.grid(True)
# plt.title('OFDM Simulate on 10000 Blocks Test',
#           fontsize=18, fontweight='bold')
# plt.ylabel('Recall', fontsize=10, fontweight='bold')
# plt.xlabel('SNR(Eb/N0)', fontsize=10, fontweight='bold')
# plt.setp(Rc, color='brown', marker='o',
#          linestyle='--', markersize=5, linewidth=1)
# plt.savefig('OFDM Simulate on 10000 Blocks_Recall.jpg')
