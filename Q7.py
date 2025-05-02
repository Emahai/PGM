import numpy as np
import matplotlib.pyplot as plt

# Load tiger image
img = np.loadtxt('tiger.txt')

# (a) Display & save as grayscale
plt.figure()
plt.imshow(img, cmap='gray', vmin=0, vmax=1)
plt.axis('off')
plt.title('Figure 1: Grayscale Tiger Image')
plt.savefig('figure1_tiger.png', dpi=300, bbox_inches='tight')
plt.close()

# (b) Histogram counts for tiger
counts_t, bins_t = np.histogram(img.flatten(), bins=256, range=(0,1))
plt.figure()
plt.bar(bins_t[:-1], counts_t, width=1/256)
plt.xlabel('Gray Level')
plt.ylabel('Count')
plt.title('Figure 2: Histogram of Tiger Image (Counts)')
plt.savefig('figure2_tiger_counts.png', dpi=300, bbox_inches='tight')
plt.close()

# (b) Histogram probabilities for tiger
probs_t = counts_t / counts_t.sum()
plt.figure()
plt.bar(bins_t[:-1], probs_t, width=1/256)
plt.xlabel('Gray Level')
plt.ylabel('Probability')
plt.title('Figure 3: Histogram of Tiger Image (Probabilities)')
plt.savefig('figure3_tiger_probs.png', dpi=300, bbox_inches='tight')
plt.close()

# (c) Generate two random-noise images of same shape
rand1 = np.random.randint(0, 256, img.shape, dtype=np.uint8)
rand2 = np.random.randint(0, 256, img.shape, dtype=np.uint8)

# Figure 4: Random Image 1
plt.figure()
plt.imshow(rand1, cmap='gray', vmin=0, vmax=255)
plt.axis('off')
plt.title('Figure 4: Random Image 1')
plt.savefig('figure4_random1.png', dpi=300, bbox_inches='tight')
plt.close()

# Figure 5: Random Image 2
plt.figure()
plt.imshow(rand2, cmap='gray', vmin=0, vmax=255)
plt.axis('off')
plt.title('Figure 5: Random Image 2')
plt.savefig('figure5_random2.png', dpi=300, bbox_inches='tight')
plt.close()

# (b/c) Histogram counts for Random Image 1
counts_r, bins_r = np.histogram(rand1.flatten(), bins=256, range=(0,255))
plt.figure()
plt.bar(bins_r[:-1], counts_r, width=1)
plt.xlabel('Gray Level')
plt.ylabel('Count')
plt.title('Figure 6: Histogram of Random Image 1 (Counts)')
plt.savefig('figure6_random1_counts.png', dpi=300, bbox_inches='tight')
plt.close()

# (b/c) Histogram probabilities for Random Image 1
probs_r = counts_r / counts_r.sum()
plt.figure()
plt.bar(bins_r[:-1], probs_r, width=1)
plt.xlabel('Gray Level')
plt.ylabel('Probability')
plt.title('Figure 7: Histogram of Random Image 1 (Probabilities)')
plt.savefig('figure7_random1_probs.png', dpi=300, bbox_inches='tight')
plt.close()

# (d) Estimate average draws to see the exact tiger image
pixels = img.shape[0] * img.shape[1]
log10N = pixels * np.log10(256)
log10_expected_draw = log10N - np.log10(2)
print(f'log10(N) ≈ {log10N:.1f}')
print(f'Expected draw position log10 ≈ {log10_expected_draw:.1f}')
