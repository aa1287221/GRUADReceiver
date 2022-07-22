import numpy as np
from matplotlib import pyplot as plt
from math import log
threshold = np.loadtxt('threshold.txt')
output_SNR = np.loadtxt('output_SNR.txt')
output_BER = np.loadtxt('output_BER.txt')
fig, plt1 = plt.subplots(figsize=(7, 4))
# MEAN = plt.plot(MEAN)
# plt.xlim(0, 35)
# plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# plt.yscale('log')
# plt.yticks([10**(-2), 10**(-1), 10**0])
# plt.grid(True)
# plt.yscale('log')
# plt.title('Variance',
#           fontsize=14, fontweight='bold')
plt1.plot(threshold, output_SNR, label='Output_SNR',
          marker='o', linestyle='-', markersize=5, linewidth=1)
plt1.set_xlabel('Threshold', fontsize=10, fontweight='bold')
plt1.set_ylabel('Optimum Output SNR', fontsize=10, fontweight='bold')
plt1.grid(True)
plt1.legend(loc='upper right')
plt2 = plt.twinx()
plt2.plot(threshold, output_BER, label='BER', color='red',
          marker='o', linestyle='-', markersize=5, linewidth=1)
plt2.set_yscale('log')
plt2.set_ylabel('BER', fontsize=10, fontweight='bold')
plt2.legend(loc='lower right')
plt2.grid(True)
plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
plt.savefig('Output_SNRBER.jpg')
plt.cla()
