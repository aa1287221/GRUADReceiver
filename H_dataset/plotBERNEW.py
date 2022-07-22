import numpy as np
from matplotlib import pyplot as plt
from math import log

RANDOMCHANNEL = [0.06628978515625, 0.026172890625, 0.010550087890625, 0.005328232421875,
                 0.00367310546875, 0.00305724609375, 0.002828955078125, 0.00281162109375, 0.00276083984375, 0.00276748046875]
# FIXEDCHANNEL = [0.06732900390625, 0.01886181640625, 0.00316611328125, 0.000373828125,
# 0.000121728515625, 8.0908203125e-05, 7.138671875e-05, 5.95703125e-05, 5.6591796875e-05, 5.0439453125e-05]
FIXEDCHANNEL = [0.076725048828125, 0.022400048828125, 0.004063671875, 0.000448388671875,
                0.000110009765625, 6.6259765625e-05, 6.142578125e-05, 5.1708984375e-05, 5.13671875e-05, 5.3759765625e-05]
FIXEDCHANNELby10 = [0.086455712890625, 0.030893212890625, 0.012438232421875, 0.00669638671875,
                    0.00509111328125, 0.00462119140625, 0.004422119140625, 0.0044009765625, 0.00440244140625, 0.0044513671875]
FIXEDCHANNELby5 = [0.08150029296875, 0.026598046875, 0.0074943359375, 0.00242939453125,
                   0.001402734375, 0.001127001953125, 0.001048779296875, 0.000977587890625, 0.001025927734375, 0.001039794921875]
SNR = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

plt.figure(figsize=(6, 6))
# BERGRU10db = plt.plot(SNR, BERGRU10db, label='GRU SINR = -10 db')
RANDOMCHANNEL = plt.plot(
    SNR, RANDOMCHANNEL, label='Random Channel SINR = -15 db p = 0.001')
FIXEDCHANNEL = plt.plot(
    SNR, FIXEDCHANNEL, label='Fixed Channel SINR = -15 db p = 0.001')
FIXEDCHANNELby5 = plt.plot(
    SNR, FIXEDCHANNELby5, label='Fixed Channel SINR = -15 db p = 0.005')
FIXEDCHANNELby10 = plt.plot(
    SNR, FIXEDCHANNELby10, label='Fixed Channel SINR = -15 db p = 0.01')
# plt.xlim(25, 75)
plt.xticks(SNR)
plt.yscale('log')
# plt.yticks([10**(-4), 10**(-3), 10**(-2), 10**(-1)])
plt.grid(True)
plt.title('OFDM Simulate on 50000 Blocks Validation',
          fontsize=14, fontweight='bold')
plt.ylabel('BER', fontsize=10, fontweight='bold')
plt.xlabel('SNR(Eb/N0)', fontsize=10, fontweight='bold')
plt.setp(RANDOMCHANNEL, marker='o',
         linestyle='-', markersize=5, linewidth=1)
plt.setp(FIXEDCHANNEL, color='red', marker='*',
         linestyle='-', markersize=5, linewidth=1)
plt.setp(FIXEDCHANNELby5, color='gray', marker='s',
         linestyle='-', markersize=5, linewidth=1)
plt.setp(FIXEDCHANNELby10, color='orange', marker='p',
         linestyle='-', markersize=5, linewidth=1)
plt.legend()
plt.savefig('OFDM Simulate on 50000 Blocks.jpg')
plt.cla()

############################################################

OUTPUTSNR = np.loadtxt('BestT25.txt')
TH = np.loadtxt('threshold.txt')

plt.figure(figsize=(6, 6))
# BERGRU10db = plt.plot(SNR, BERGRU10db, label='GRU SINR = -10 db')
BERCONVENTIONAL10db = plt.plot(
    TH, OUTPUTSNR, label='SNR = 25 db SINR = -15 db p = 0.001')
# plt.xlim(25, 75)
plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
plt.yticks([-25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25])
plt.grid(True)
plt.ylabel('Output SNR(db)', fontsize=10, fontweight='bold')
plt.xlabel('Blanking Threshold(T)', fontsize=10, fontweight='bold')
plt.setp(BERCONVENTIONAL10db, color='red', marker='*',
         linestyle='-', markersize=5, linewidth=1)
plt.legend()
plt.savefig('OutputSNR.jpg')
plt.cla()
