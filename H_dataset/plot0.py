import numpy as np
from matplotlib import pyplot as plt
from math import log

BERold = [0.0907373046875, 0.031068359375, 0.0102265625, 0.0038603515625, 0.00121875, 0.00056787109375,
          0.000306640625, 0.000234130859375, 0.000194580078125, 0.00019189453125]
# BERdiv10 = [0.0886826171875, 0.03134619140625, 0.01058056640625, 0.0035478515625, 0.0010517578125,
#             0.00040087890625, 0.00013037109375, 6.005859375e-05, 3.3447265625e-05, 2.2216796875e-05]
# BERnoIN = [0.0894609375, 0.031893115234375, 0.01041748046875, 0.0032697265625, 0.0010162109375, 0.000332373046875,
#            9.716796875e-05, 2.890625e-05, 1.103515625e-05, 3.22265625e-06]
BER = [0.093197265625, 0.03501806640625, 0.013025390625, 0.0066552734375, 0.004193359375,
       0.003349609375, 0.0030646484375, 0.002769921875, 0.0026448015603799187, 0.00258349609375]
BERknown = [0.09061962890625, 0.032229248046875, 0.01091943359375, 0.004106201171875, 0.001825927734375,
            0.0012122844827586207, 0.00107080078125, 0.00087783203125, 0.00086640625, 0.00081181640625]

SNR = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

plt.figure(figsize=(6, 6))
OFDM = plt.plot(SNR, BER, label='Unknown Impulse Blank prob.0.33%')
OFDMknown = plt.plot(SNR, BERknown, label='Known Impulse Blank prob.0.33%')
OFDMnoIN = plt.plot(SNR, BERold, label='known Impulse Blank prob.0.1%')
# plt.xlim(25, 75)
plt.xticks(SNR)
plt.yscale('log')
plt.yticks([10**(-4), 10**(-3), 10**(-2), 10**(-1)])
plt.grid(True)
plt.title('OFDM Simulate on 2000 Blocks Validation',
          fontsize=14, fontweight='bold')
plt.ylabel('BER', fontsize=10, fontweight='bold')
plt.xlabel('SNR(Eb/N0)', fontsize=10, fontweight='bold')
plt.setp(OFDM, marker='o',
         linestyle='-', markersize=5, linewidth=1)
plt.setp(OFDMknown, color='red', marker='*',
         linestyle='-', markersize=5, linewidth=1)
plt.setp(OFDMnoIN, color='purple', marker='p',
         linestyle='-', markersize=5, linewidth=1)
plt.legend()
plt.savefig('OFDM Simulate on 10000 Blocks0.jpg')
plt.cla()
