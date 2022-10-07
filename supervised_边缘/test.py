import numpy as np
def ramp_up(epoch, max_epochs, max_val, mult):
    if epoch == 0:
        return 0.
    elif epoch >= max_epochs:
        return max_val
    return max_val * np.exp(mult * (1. - float(epoch) / max_epochs) ** 2)
def weight_schedule(epoch, max_epochs, max_val, mult):
    # max_val = max_val * (float(n_labeled) / n_samples)
    return 1 - ramp_up(epoch, max_epochs, max_val, mult)


# plot Gaussian Function
# 注：正态分布也叫高斯分布
import matplotlib.pyplot as plt

x = np.arange(0,100,1)

y = []
for i in x:
    y.append(np.exp(-5 * (  i / 100 ) ** 2))
    # y.append(np.exp(-5 * (1 - (i + 100) / 100) ** 2))

plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决pythonmatplotlib绘图无法显示中文的问题
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# plt.subplot(121)
plt.plot(x, y, 'b-', linewidth=2)
plt.title("高斯分布函数图像")

plt.show()