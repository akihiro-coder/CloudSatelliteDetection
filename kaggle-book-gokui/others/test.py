import torch 
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# 一次関数モデルを考える
w = 4.5
b = 7.0
x_array = np.array([-1.5, -1.0, -0.1, 0.9, 1.8, 2.2, 3.1])
y_array = w * x_array + np.random.normal(size=x_array.shape[0])
x = torch.tensor(x_array).float() # input
y = torch.tensor(y_array).float() # 教師データ

"""
誤差逆伝播は次の3ステップで行う
step1. 勾配計算が必要なパラメータ(required_grad = True)を使ってモデルで計算
step2. 得られた出力とground truthから誤差を計算
step3. 誤差関数の出力(Tensorオブジェクト)のbackwardメソッドを呼び出す
"""

# パラメータのセット　
## w=1.0, b=0.0で初期化
param_w = torch.tensor([1.0], requires_grad=True)
param_b = torch.tensor([0.0], requires_grad=True)

# step1: 一次関数を仮定して計算
y_p = param_w * x + param_b
"""
# torch.tensor.grad_fnプロパティ
- 勾配計算に使う関数オブジェクトがセットされている
    - (ex)print(y_p.grad_fn) : AddBackward0 object
- Tensorの演算関数ごとに定義されている
- __call__で実行できる
    - (ex) print(y_p.grad_fn(torch.tensor([2]))) -> (tensor([2]), tensor([2]))
"""

# step2: 平均2乗誤差を計算
loss = torch.mean((y_p - y)**2)
"""
# torch.tensor.grad_fn.next_functionsプロパティ
- 誤差を伝播する先の関数オブジェクトへの参照が入っている
- 実験
    print(loss.grad_fn) # MeanBackward0
    print(loss.grad_fn.next_functions[0][0]) # PowBackward0
    print(loss.grad_fn.next_functions[0][0].next_functions[0][0]) # SubBackward0
    print(loss.grad_fn.next_functions[0][0].next_functions[0][0].next_functions[0][0]) # AddBackward0
"""

# step3: 誤差を伝播する
# print('before: ', param_w.grad, param_b.grad)
loss.backward()
# print('after: ', param_w.grad, param_b.grad)


"""
実験
backwardメソッドは勾配が累積される仕様になっている
以下のように３回続けてbackwardを呼び出すと勾配も３倍になる
for i in range(3):
    y_p = param_w * x  + param_b
    loss = torch.mean((y_p - y)**2)
    loss.backward()
    print(param_w.grad, param_b.grad)
>>
tensor([-23.0909]) tensor([-6.4400])
tensor([-46.1818]) tensor([-12.8801])
tensor([-69.2727]) tensor([-19.3201])

勾配をゼロクリアにする処理を書くと、勾配は累積されない。
optimパッケージのzero_grad()メソッドでやっていることと同じ
for i in range(3):
    if param_w.grad: param_w.grad.zero_()
    if param_b.grad: param_b.grad.zero_()
    y_p = param_w * x  + param_b
    loss = torch.mean((y_p - y)**2)
    loss.backward()
    print(param_w.grad, param_b.grad)
>> 
tensor([-23.0909]) tensor([-6.4400])
tensor([-23.0909]) tensor([-6.4400])
tensor([-23.0909]) tensor([-6.4400])
"""


"""
# with torch.no_grad()と書くブロックでは勾配の計算をしないようになっている
# メモリ消費を減らすことが出来るらしい
with torch.no_grad():
    print(param_w.requires_grad, param_b.requires_grad)
    # True, True

    y_p = param_w * x + param_b
    print(y_p, y_p.requires_grad, y_p.grad)
    # tensor([-1.5000, -1.0000, -0.1000,  0.9000,  1.8000,  2.2000,  3.1000]) False None

    loss = torch.mean((y_p - y)**2)
    print(loss, loss.requires_grad, loss.grad)
    # tensor(43.5394) False None
"""
