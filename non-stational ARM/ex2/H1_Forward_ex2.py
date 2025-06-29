import numpy as np
from mshr import *
from scipy.special import *
from fenics import *
from dolfin import *
from math import cos, sin, pi, sqrt
import matplotlib.pyplot as plt
import os
import time  # 导入 time 模块
# 这个程序的目的是从另一个角度建立波方程的解法
# 记录开始时间
start_time = time.time()
N = 10  # 贝塞尔函数的阶数最大值
N = 10  # 选择的零点个数


##############################################################
# 创建网格和函数空间
##############################################################
domain = Circle(Point(0, 0), 1)  # 创建一个单位圆
Nh = 50
mesh = generate_mesh(domain, Nh)  # 生成网格
num_points = 200
r_vals = np.linspace(0, 1, num_points)  # r 从 0 到 1
theta_vals = np.linspace(0, 2 * np.pi, num_points)  # θ 从 0 到 2π
V = FunctionSpace(mesh, 'P', 1)  # 定义函数空间


# Bessel函数的零点
jmn_values = np.loadtxt(os.path.join(os.path.dirname(__file__), 'jmn_values.txt'))
T =1 
num_step = 100
dt =T/num_step


# ms = 5  # 例如，选择 J_0
# ns = 4
# abs_ms = np.abs(ms)
# jmns = jmn_values[abs_ms,ns-1] 
# bessel_expression = Expression('jn(n, jnk*sqrt(x[0]*x[0] + x[1]*x[1]))', degree=4, n=ms, jnk=jmns)


# if ms == 0:
#     normal_constant = np.sqrt(1) / np.abs(jn(ms+1, jmns)) / np.sqrt(pi)
# else:
#     normal_constant = np.sqrt(2) / np.abs(jn(ms+1, jmns)) / np.sqrt(pi)

# # 根据 ms 的值选择合适的三角函数
# if ms >= 0:
#     polar_expression = Expression('cos(n * atan2(x[1], x[0]))', degree=4, n=ms)
# else:
    
#     polar_expression = Expression('sin(n * atan2(x[1], x[0]))', degree=4, n=abs_ms)

# 将两个表达式相乘

# u_finite  =  bessel_expression * polar_expression * normal_constant

j21 = jmn_values[2, 0]
u_finite = Expression(
    " jn(2, j21*sqrt(x[0]*x[0]+x[1]*x[1])) * (exp(-x[1])+exp(x[1]))",
    degree=4,
    j21=j21
)

# def g(t):
#     return  2.0+np.power(t,2)*np.power(jmns,2)
def g(t):
       return  np.power(t,2)+1



#######################################################
###################生成指标输入输出  ####################
#######################################################

def bessel_index(N):
    index_matrix = []  # 在这里初始化矩阵
    for j in range(1, N + 1):
        index_matrix.append([0, j])  # 添加第一列为0，第二列为j
    for i in range(1, N + 1):
        for j in range(1, N + 1):
            index_matrix.append([i, j])  # 添加i和j的组合
            index_matrix.append([-i, j]) 
    return np.array(index_matrix)

# 设置N


def four_index(N):
    index_matrix = []  # 在这里初始化矩阵
    index_matrix.append([0])
    for j in range(1,N + 1):
        index_matrix.append([j])  # 添加第一列为0，第二列为j
        index_matrix.append([-j])
    return np.array(index_matrix)
  

input_index = bessel_index(N)
### 输出“输入指标”#######
# input_index_file = os.path.join(os.path.dirname(__file__), 'input_index.txt') 
# np.savetxt(input_index_file, input_index)
output_index = four_index(N) 
# output_index_file = os.path.join(os.path.dirname(__file__), 'output_index.txt') 
# np.savetxt(output_index_file, output_index)

#########################################################################################
############################ 计算傅里叶系数                              #################
#########################################################################################
M = N+1
a_coeffs = np.zeros((M, N))
b_coeffs = np.zeros((M, N))
normal_coeff = np.zeros((M,N))


# 预先定义三角函数表达式
cos_expressions = [Expression('cos(m * atan2(x[1], x[0]))', degree=4, m=m) for m in range(M)]
sin_expressions = [Expression('sin(m * atan2(x[1], x[0]))', degree=4, m=m) for m in range(M)]
# 预先定义 Bessel 表达式
bessel_expressions = [[Expression('jn(m, jmn * sqrt(x[0]*x[0] + x[1]*x[1]))', degree=4, m=m, jmn=jn_zeros(m, n + 1)[n]) for n in range(N)] for m in range(M)]
# 预先计算 Bessel 函数的零点

# 计算Bessel基下的傅里叶系数
# m 阶系数存在第m行中，m从0开始
# 定义积分测度时指定 quadrature_degree
dx = Measure('dx', domain=mesh, metadata={'quadrature_degree': 8})
# 为了计算精度，提高了积分的格式，这会拖慢进度 

for m in range(M):
    for n in range(N):
        jmn = jmn_values[m][n]  # 使用预计算的零点, jmn是m阶Bessel函数的第n+1个零点
        bessel_expression = bessel_expressions[m][n]  # 使用预定义的 Bessel 表达式
        bessel_cos = cos_expressions[m] * bessel_expression
        bessel_sin = sin_expressions[m] * bessel_expression
        if m == 0:
            normal_coeff[m][n] = 1 / np.abs(jn(m + 1, jmn))/ np.sqrt(pi)
        else: 
            normal_coeff[m][n] = np.sqrt(2) /np.abs(jn(m + 1, jmn))/np.sqrt(pi)
        a_coeffs[m, n] = normal_coeff[m][n] * assemble(u_finite * bessel_cos * dx(mesh))
        b_coeffs[m, n] = normal_coeff[m][n] * assemble(u_finite * bessel_sin * dx(mesh))

normal_coeff_file = os.path.join(os.path.dirname(__file__), 'normal_coeff.txt') 
np.savetxt(normal_coeff_file, normal_coeff)
a_coeffs_file = os.path.join(os.path.dirname(__file__), 'a_coeffs.txt') 
np.savetxt(a_coeffs_file, a_coeffs)
b_coeffs_file = os.path.join(os.path.dirname(__file__), 'b_coeffs.txt') 
np.savetxt(b_coeffs_file, b_coeffs)


#################################################################################################
 ## 把傅里叶系数录入到输入变量 ： coe_four是基于L2的系数，coe_H1_four是基于H1的系数
#################################################################################################

coe_four = np.zeros((input_index.shape[0],1)) # 构造输入向量并且按照input_index的顺序存入傅里叶系数
coe_H1_four = coe_four.copy()
for j in range(input_index.shape[0]):
    [m, n] = input_index[j]
    m_int = int(m)
    n_int = int(n)
    if m_int >= 0:
        coe_four[j]= a_coeffs[m_int, n_int-1]
        coe_H1_four[j]= coe_four[j] * np.abs(jmn_values[m_int, n-1])
    else:
        coe_four[j]= b_coeffs[-m_int, n_int-1]
        coe_H1_four[j]= coe_four[j] * np.abs(jmn_values[-m_int, n-1])

#print(coe_four)
coe_four_file = os.path.join(os.path.dirname(__file__), 'coe_four.txt') 
np.savetxt(coe_four_file, coe_four)

coe_H1_four_file = os.path.join(os.path.dirname(__file__), 'coe_H1_four.txt') 
np.savetxt(coe_H1_four_file, coe_H1_four)
# real_value = np.sqrt(2) / np.abs(jn(ms+1, jmns))/ np.sqrt(pi)
# print(real_value)




#########################################################################
#生成时间算子M_mn，每一列是一个特征函数对应的时间算子在时间上的离散向量####
#########################################################################

input_size = input_index.shape[0]

# 欧拉方法求解二阶常微分方程
def solve_ode(eigen):
    t_vals = np.linspace(0, T, num_step + 1)  # 时间网格
    u_vals = np.zeros(num_step + 1)  # 存储 u(t) 的数组
    v_vals = np.zeros(num_step + 1)  # 存储 v(t) 的数组
    
    # 初始条件
    u_vals[0] = 0
    v_vals[0] = 0
    # 迭代法（欧拉方法）
    for i in range(num_step):
        u_vals[i + 1] = u_vals[i] + dt * v_vals[i]  # 更新 u(t)
        v_vals[i + 1] = v_vals[i] - eigen* dt * u_vals[i] + dt*g(t_vals[i])  # 更新 v(t)
    return  u_vals  

# 构造时间矩阵（i，j）元表示某个ode的i_th 时刻的，这个ode的系数依赖于j所在的特征值
M_time = np.zeros((num_step, input_size)) 
for i in range(input_size):
    m = np.abs(input_index[i,0])
    n = input_index[i,1]
    eigen = np.power(jmn_values[m,n-1],2)
    Sol =solve_ode(eigen)
    M_time[:,i] = Sol[1:]


##########################定义源项的时间部分################################
###########################################################################
##################        计算方程解在最终时刻的傅里叶系数   #################
###########################################################################

output_index_size = output_index.shape[0]


# 计算m阶的Bessel函数在jmn点
# （m阶Bessel函数的的n个零点）的微分的值
def coe_diff(m, n):
    #这里的m,n都是他们本身的值，不是指标顺序的值
    # 
    jmn = jmn_values[m, n-1]
    if m == 0:
        coe_value = jmn / 2 * (jn(m-1, jmn) - jn(m+1, jmn)) * normal_coeff[m][n-1]*np.sqrt(2*pi)
    else:
        coe_value = jmn / 2 * (jn(m-1, jmn) - jn(m+1, jmn)) * normal_coeff[m][n-1]*np.sqrt(pi)
    return coe_value

tol = 1e-4

# 构造时空矩阵
output_index_temp = np.tile(output_index, (num_step, 1))
replication_indices = dt * np.repeat(np.arange(1, num_step + 1), output_index.shape[0]).reshape(-1, 1)
output_index_whole = np.hstack((output_index_temp,replication_indices))
output_index_fullsize = output_index_whole.shape[0]


# output_index_whole_file = os.path.join(os.path.dirname(__file__), 'whole_index.txt') 
# np.savetxt(output_index_whole_file, output_index_whole)

M_final = np.zeros((output_index_fullsize, input_size))
for j in range(output_index_fullsize):
   for i in range(input_size):
         m = int(input_index[i,0])
         n = int(input_index[i,1])
         mp = int(output_index_whole[j,0])
         tp = output_index_whole[j,1]
         if np.abs(mp-m) < tol:
             m_ab = np.abs(m)
             t_step = int(tp/dt)-1 
             M_final[j,i] = coe_diff(m_ab, n) * M_time[t_step,i]


M_final_file = os.path.join(os.path.dirname(__file__), 'M_final.txt') 
np.savetxt(M_final_file, M_final) 


# change the bais from L2 norm to H1 norm, creat the matrix of the forward operator  
# 计算基于H1半范数的矩阵      
M_H1_final = M_final.copy()
for i in range(input_size):
    m = np.abs(int(input_index[i,0]))
    n = int(input_index[i,1])
    divisor = np.abs(jmn_values[m, n-1])
    M_H1_final[:,i] = M_H1_final[:,i] / divisor
  
M_H1_final_file = os.path.join(os.path.dirname(__file__), 'M_H1_final.txt') 
np.savetxt(M_H1_final_file, M_H1_final) 

    

# a0_val = 0 


# ###########################################################################
# ##################        提取 t 时刻数据                  #################
# ###########################################################################
u_coeffs = np.dot(M_final, coe_four)
u_H1_coefs = np.dot(M_H1_final, coe_H1_four)
# 计算 u_coeffs 和 u_H1_coefs 之间的 L2 范数
L2_error = np.linalg.norm(u_coeffs - u_H1_coefs)
print(f"L2 norm of the error: {L2_error}")


t = 1 
u_coeffs_file = os.path.join(os.path.dirname(__file__), 'u_coeffs.txt') 
np.savetxt(u_coeffs_file, u_coeffs) 
def find_time(output_index_whole, t):
    indices = np.where(output_index_whole[:, 1] == t)[0]
    if len(indices) == 0:
        return None, None
    return indices[0], indices[-1]

start_idx, end_idx = find_time(output_index_whole, t)
u_coeffs_t = u_coeffs[start_idx:end_idx+1]
u_H1_coefs_t = u_H1_coefs[start_idx:end_idx+1]
# print(u_coeffs_t)


# 提取边界傅里叶系数并且重构出函数

# a0_val = 0 
# a_circ_coeffs = np.zeros((N, 1))
# b_circ_coeffs = np.zeros((N, 1))

# for i in range(u_coeffs_t.shape[0]):
#      k = int(output_index[i,0])
#      if k == 0:
#          a0_val = u_coeffs_t[i, 0]
#      if k > 0:
#          a_circ_coeffs[k-1,0] = u_coeffs_t[i, 0]
#      if k < 0:
#          b_circ_coeffs[-k-1,0] = u_coeffs_t[i, 0]

a0_H1_val = 0 
a_circ_H1_coeffs = np.zeros((N, 1))
b_circ_H1_coeffs = np.zeros((N, 1))
for i in range(u_H1_coefs_t.shape[0]):
     k = int(output_index[i,0])
     if k == 0:
         a0_val = u_H1_coefs_t[i, 0]
     if k > 0:
         a_circ_H1_coeffs[k-1,0] = u_H1_coefs_t[i, 0]
     if k < 0:
         b_circ_H1_coeffs[-k-1,0] = u_H1_coefs_t[i, 0]

def fourier_series(theta):
    series = a0_val/np.sqrt(2*pi)
    for n in range(1, N + 1):
        series += a_circ_H1_coeffs[n - 1,0] * np.cos(n * theta)/np.sqrt(pi) + b_circ_H1_coeffs[n - 1,0] * np.sin(n * theta)/np.sqrt(pi)
    return series




x_true_vals = np.linspace(-np.pi, np.pi, 100)
fs_vals = fourier_series(x_true_vals) 

# eigen1 = pow(jmns,2)
# sol1 = solve_ode(eigen1)
# un1 = sol1[-1]


# def  f_exact(theta):
#     if ms ==0: 
#         temp1 = np.sqrt(1)/abs(jn(ms+1,jmns)) * jmns / 2 * (jn(ms-1,jmns)-jn(ms+1,jmns))*un1
#     else: 
#         temp1 = np.sqrt(2)/abs(jn(ms+1,jmns)) * jmns / 2 * (jn(ms-1,jmns)-jn(ms+1,jmns))*un1
#     if ms >=0:
#         trigo = np.cos(ms*theta) /np.sqrt(pi) 
#     else:
#         trigo = -np.sin(ms*theta) /np.sqrt(pi) 
# #       temp2 = np.sqrt(2)/abs(jn(ms2+1,jmns2)) * jmns2 / 2 * (jn(ms2-1,jmns2)-jn(ms2+1,jmns2))*un2
#     return temp1 * trigo
# # # print(un)
# f_vals = f_exact(x_true_vals)


plt.plot(x_true_vals, fs_vals, label='Approximate curve', color='blue', linestyle='-')
# plt.plot(x_true_vals, f_vals, label='True curve', color='red', linestyle='--')
plt.legend()
plt.title(r'$\frac{\partial u}{\partial \nu}\mid_{t=1}$')
plt.xlabel(r'$\theta$')
plt.grid()


# 保存图像
current_dir = os.path.dirname(__file__)
# 构建完整的文件路径
image_path = os.path.join(current_dir, "Wave_normal.png")
# 保存图像到程序所在的文件夹
plt.savefig(image_path, dpi=300)  # 保存为PNG文件，300dpi质量
plt.show()





