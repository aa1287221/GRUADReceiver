import numpy as np
from matplotlib import pyplot as plt
from math import log

BER = [0.0907373046875, 0.031068359375, 0.0102265625, 0.0038603515625, 0.00121875, 0.00056787109375,
       0.000306640625, 0.000234130859375, 0.000194580078125, 0.00019189453125]
BERdiv10 = [0.0886826171875, 0.03134619140625, 0.01058056640625, 0.0035478515625, 0.0010517578125,
            0.00040087890625, 0.00013037109375, 6.005859375e-05, 3.3447265625e-05, 2.2216796875e-05]
BERnoIN = [0.0894609375, 0.031893115234375, 0.01041748046875, 0.0032697265625, 0.0010162109375, 0.000332373046875,
           9.716796875e-05, 2.890625e-05, 1.103515625e-05, 3.22265625e-06]
BERbyGRU = []

SNR = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

plt.figure(figsize=(6, 6))
OFDM = plt.plot(SNR, BER, label='Impulse prob.0.1%')
OFDMdiv10 = plt.plot(SNR, BERdiv10, label='Impulse prob.0.01%')
OFDMnoIN = plt.plot(SNR, BERnoIN, label='No Impulse')
# plt.xlim(25, 75)
plt.xticks(SNR)
plt.yscale('log')
plt.yticks([10**(-4), 10**(-3), 10**(-2), 10**(-1), 10**(-0)])
plt.grid(True)
plt.title('OFDM Simulate on 2000 Blocks Validation',
          fontsize=14, fontweight='bold')
plt.ylabel('BER', fontsize=10, fontweight='bold')
plt.xlabel('SNR(Eb/N0)', fontsize=10, fontweight='bold')
plt.setp(OFDM, marker='o',
         linestyle='-', markersize=5, linewidth=1)
plt.setp(OFDMdiv10, color='red', marker='*',
         linestyle='--', markersize=5, linewidth=1)
plt.setp(OFDMnoIN, color='purple', marker='p',
         linestyle='-.', markersize=5, linewidth=1)
plt.legend()
plt.savefig('OFDM Simulate on 10000 Blocks0.jpg')
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
