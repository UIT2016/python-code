import numpy as np

# 1. 创建矩阵和向量
A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
B = np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]])
v = np.array([1, 2, 3])

print("矩阵 A:")
print(A)
print("\n向量 v:", v)

# 2. 基本矩阵运算
print("\n--- 基本运算 ---")
print("A + B =")
print(A + B)

print("\nA * 2 =")
print(A * 2)

print("\nA 和 B 的 Hadamard 积 (元素级乘法):")
print(A * B)

print("\nA 与 B 的矩阵乘法:")
print(A @ B)  # 或 np.dot(A, B)

# 3. 矩阵属性
print("\n--- 矩阵属性 ---")
print("A 的转置:")
print(A.T)

print("\nA 的行列式:", np.linalg.det(A))  # 注意：此矩阵奇异，行列式接近0

# 4. 求解线性方程组 Ax = b
b = np.array([14, 32, 50])
try:
    x = np.linalg.solve(A, b)
    print("\n方程组 Ax = b 的解:", x)
    print("验证 A@x =", A @ x)  # 应等于 b
except np.linalg.LinAlgError as e:
    print("\n无法求解方程组:", e)
    # 使用伪逆处理奇异矩阵
    x_pseudo = np.linalg.pinv(A) @ b
    print("使用伪逆求解:", x_pseudo)

# 5. 特征值分解
print("\n--- 特征值分解 ---")
# 创建可逆矩阵（原A奇异）
C = np.array([[2, 1], [1, 3]])
eigenvalues, eigenvectors = np.linalg.eig(C)

print("矩阵 C:\n", C)
print("特征值:", eigenvalues)
print("特征向量:\n", eigenvectors)
print("验证 Cv = λv (第一特征向量):")
v1 = eigenvectors[:, 0]
print("C @ v1 =", C @ v1)
print("λ1 * v1 =", eigenvalues[0] * v1)

# 6. 奇异值分解 (SVD)
print("\n--- 奇异值分解 ---")
U, S, Vt = np.linalg.svd(A)  # A = U @ diag(S) @ Vt
print("U 矩阵:\n", U)
print("奇异值:", S)
print("V 转置矩阵:\n", Vt)

# 重构验证
Sigma = np.zeros(A.shape)
np.fill_diagonal(Sigma, S)
A_reconstructed = U @ Sigma @ Vt
print("重构矩阵与原矩阵差值范数:", np.linalg.norm(A - A_reconstructed))
