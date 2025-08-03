import matplotlib.pyplot as plt
import numpy as np

veh_ids = np.arange(15)

# --- Table V (clustering) ---
acc_v = np.array([np.nan, 0.9201, 0.8960, 0.9272, 0.9613, 0.9513, 0.8887,
                  0.9107, 0.8943, 0.9222, 0.9477, 0.8494, 0.9460,
                  0.9057, 0.7952])
low_v = np.array([np.nan, 0.6161, 0.5700, 0.5238, 0.6598, 0.8345, 0.7918,
                  0.5645, 0.7670, 0.5585, 0.8437, 0.7157, 0.8359,
                  0.8829, 0.5822])
up_v  = np.array([np.nan, 0.9720, 0.9510, 0.9515, 0.9536, 0.9969, 0.8652,
                  0.9479, 0.9873, 0.9568, 0.9951, 0.9840, 0.9710,
                  0.9720, 0.8989])

# --- Table VI (driver-ID) ---
acc_vi = np.array([0.802, 0.809, 0.894, 0.835, 0.717, 0.797, 0.984,
                   0.820, 0.781, 0.782, 0.818, 0.915, 0.801, 0.949,
                   0.800])
low_vi = np.array([0.787, 0.786, 0.883, 0.816, 0.701, 0.783, 0.981,
                   0.804, 0.755, 0.768, 0.806, 0.903, 0.780, 0.941,
                   0.786])
up_vi  = np.array([0.817, 0.832, 0.905, 0.852, 0.732, 0.812, 0.987,
                   0.835, 0.804, 0.795, 0.831, 0.926, 0.822, 0.955,
                   0.811])

# error bars (non-negative)
err_low_v  = np.maximum(0, acc_v - low_v)
err_high_v = np.maximum(0, up_v  - acc_v)
err_low_vi = acc_vi - low_vi
err_high_vi= up_vi  - acc_vi

fig, ax = plt.subplots(figsize=(12, 6))

valid_v = ~np.isnan(acc_v)          # Table V has no vehicle 0

# 1) faint stems as guides
ax.stem(veh_ids[valid_v], acc_v[valid_v], basefmt=" ", linefmt="C1:", markerfmt=" ")
ax.stem(veh_ids + 0.2,  acc_vi,        basefmt=" ", linefmt="C0:", markerfmt=" ")

# 2) mean markers
ax.scatter(veh_ids[valid_v], acc_v[valid_v], color='C1', marker='o', label='Table V')
ax.scatter(veh_ids + 0.2,  acc_vi,        color='C0', marker='s', label='Table VI')

# 3) 95 % CI whiskers
ax.errorbar(veh_ids[valid_v], acc_v[valid_v],
            yerr=[err_low_v[valid_v], err_high_v[valid_v]],
            fmt='none', ecolor='C1', capsize=4, lw=2)
ax.errorbar(veh_ids + 0.2,  acc_vi,
            yerr=[err_low_vi, err_high_vi],
            fmt='none', ecolor='C0', capsize=4, lw=2)

# cosmetics
ax.set_title("Vehicle-wise Accuracy ±95 % CI\nOrange = Table V • Blue = Table VI")
ax.set_xlabel("Vehicle ID")
ax.set_ylabel("Accuracy")
ax.set_xticks(veh_ids)
ax.set_ylim(0.60, 1.02)
ax.grid(axis='y', alpha=0.3)
ax.legend(frameon=False)
plt.tight_layout()
plt.show()
