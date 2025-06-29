import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.special import jn
from fenics import *
from mshr import *


##############################################################
# 创建网格和函数空间
##############################################################
domain = Circle(Point(0, 0), 1)  # 创建一个单位圆
Nh = 100
mesh = generate_mesh(domain, Nh)  # 生成网格
num_points = 100
r_vals = np.linspace(0, 1, num_points)  # r 从 0 到 1
theta_vals = np.linspace(0, 2 * np.pi, num_points)  # θ 从 0 到 2π
V = FunctionSpace(mesh, 'P', 1)  # 定义函数空间
# 定义一个示例函数 u_finite

jmn_values = np.loadtxt(os.path.join(os.path.dirname(__file__), 'jmn_values.txt'))
j21 = jmn_values[2, 0]
u_finite = Expression(
    " jn(2, j21*sqrt(x[0]*x[0]+x[1]*x[1])) * (exp(-x[1])+exp(x[1]))",
    degree=4,
    j21=j21
)

u_finite_projected = project(u_finite, V)




##############################################################
# 读取系数
##############################################################
coe_four = np.loadtxt(os.path.join(os.path.dirname(__file__), 'Tcoe.txt'))
input_index = np.loadtxt(os.path.join(os.path.dirname(__file__), 'input_index.txt'))
jmn_values = np.loadtxt(os.path.join(os.path.dirname(__file__), 'jmn_values.txt'))
normal_coeff = np.loadtxt(os.path.join(os.path.dirname(__file__), 'normal_coeff.txt'))


print(input_index.shape)

#找出每一列的最大值
max_values_per_column = np.max(input_index, axis=0)

# # 打印每一列的最大值
# print("最大值 per column:", max_values_per_column)
M = int(max_values_per_column[0])+1
N = int(max_values_per_column[1])
a_coeffs = np.zeros((M, N))
b_coeffs = np.zeros((M,N))
for j in range(input_index.shape[0]):
    [i_m, i_n] = input_index[j]
    m_int = int(i_m)
    n_int = int(i_n)
    if m_int >= 0:
        a_coeffs[m_int, n_int-1] = normal_coeff[m_int, n_int-1]*coe_four[j] / np.abs(jmn_values[m_int, n_int-1])
        b_coeffs[-m_int, n_int-1] = normal_coeff[m_int, n_int-1]*coe_four[j] / np.abs(jmn_values[-m_int, n_int-1])

# print(a_coeffs)
# print(b_coeffs)



###########################################################################
##################           重构函数                      #################
###########################################################################

def reconstruct_function(r, theta):
    # 创建一个空的数组来存储重构结果
    f_reconstructed = np.zeros_like(theta)
    for m in range(M):
        # 使用预计算的 jmn 值
        jmn = jmn_values[m]  # 把m行都提取出来

        # 计算 cos 和 sin 的值
        cos_values = a_coeffs[m, :] * np.cos(m * theta)
        sin_values = b_coeffs[m, :] * np.sin(m * theta)

        # 计算 Bessel 函数值
        for n in range(N):
            f_reconstructed += (cos_values[n] + sin_values[n]) * jn(m, jmn[n] * r)

    return f_reconstructed

# 计算重构函数在直角坐标系中的值

Z_reconstructed = np.zeros((num_points, num_points))
for i in range(num_points):
    for j in range(num_points):
        Z_reconstructed[i, j] = reconstruct_function(r_vals[i], theta_vals[j])

Z_vector = Z_reconstructed.ravel()
print("shape of Z_vector:", Z_vector.shape)
# print("value of Z:",Z_reconstructed)
# np.savetxt('Z_data.txt', Z_reconstructed, fmt='%d', delimiter=' ',  header='Z_data', comments='# ')
# 计算直角坐标系中的 X 和 Y
X = r_vals[:, None] * np.cos(theta_vals)  # X 坐标
X_vector = X.ravel()
# print("value of r_vals:",r_vals)
# print("value of cos:",np.cos(theta_vals))
# print("value of X:",X)
# np.savetxt('X_data.txt', X, fmt='%d', delimiter=' ',  header='X_data', comments='# ')
Y = r_vals[:, None] * np.sin(theta_vals)  
print(Y.shape)# Y 坐标
Y_vector = Y.ravel()
# print("value of Y:", Y)
# np.savetxt('Y_data.txt', Y, fmt='%d', delimiter=' ',  header='Y_data', comments='# ')

data_plot_ex2H1 = np.column_stack((X_vector,Y_vector,Z_vector))
np.savetxt('data_plot_recons.txt', data_plot_ex2H1)
###########################################################################
##################           绘图并且比较两个图             #################
###########################################################################


plt.figure(figsize=(12, 6))

# 绘制投影后的原函数
plt.subplot(1, 2, 1)
plot(u_finite_projected, mesh=mesh)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Original Function')

# 绘制重构函数
plt.subplot(1, 2, 2)
contour = plt.contourf(X, Y, Z_reconstructed, levels=50, cmap='viridis')
plt.colorbar(contour)  # 添加 colorbar
plt.title('Reconstructed Function')
plt.xlabel('X')
plt.ylabel('Y')
plt.axis('equal')

# 保存图像
current_dir = os.path.dirname(__file__)
# 构建完整的文件路径
image_path = os.path.join(current_dir, "reconstruct.png")
# 保存图像到程序所在的文件夹
plt.savefig(image_path, dpi=300)  # 保存为PNG文件，300dpi质量
plt.tight_layout()
plt.show()