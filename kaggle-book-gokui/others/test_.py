import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


np.random.seed(42)


w = 4.5
b = 7.0

x_array = np.array([-1.5, -1.0, -0.1, 0.9, 1.8, 2.2, 3.1])
y_array = w * x_array + np.random.normal(size=x_array.shape[0])
x = torch.tensor(x_array).float()  # 入力
y = torch.tensor(y_array).float()  # 正解


# Tensorだけで学習させてみる
# param_w = torch.tensor([1.0], requires_grad=True)
# param_b = torch.tensor([0.0], requires_grad=True)

# epochs = 300
# lr = 0.01

# for epoch in range(1, epochs + 1):
#     # 予測
#     y_p = param_w * x + param_b
#     # 誤差評価
#     loss = torch.mean((y_p - y)**2)
#     # 勾配計算
#     loss.backward()
#     # パラメータ更新
#     param_w = (param_w - param_w.grad * lr).detach().requires_grad_()
#     param_b = (param_b - param_b.grad * lr).detach().requires_grad_()

#     if epoch % 10 == 0:
#         print(f'Epoch {epoch}: loss={loss}, param_w={float(param_w)}, param_b={float(param_b)}')


# optim.SGDを使って学習させてみる
param_w = torch.tensor([1.0], requires_grad=True)
param_b = torch.tensor([0.0], requires_grad=True)

epochs = 300
lr = 0.01
optimizer = optim.SGD([param_w, param_b], lr=lr)

for epoch in range(1, epochs + 1):
    # 予測
    y_p = param_w * x + param_b
    # 誤差評価
    loss = torch.mean((y_p - y)**2)
    # 勾配をゼロにリセット
    optimizer.zero_grad()
    # 勾配計算
    loss.backward()
    # パラメータ更新
    optimizer.step()

    if epoch % 10 == 0:
        print(f'Epoch {epoch}: loss={loss}, param_w={float(param_w)}, param_b={float(param_b)}')
