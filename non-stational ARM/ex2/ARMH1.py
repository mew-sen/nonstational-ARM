import numpy as np
import matplotlib.pyplot as plt
import os
import multiprocessing as mp
from itertools import product
import time
from scipy.sparse import csr_matrix, eye, diags
from scipy.sparse.linalg import spsolve, factorized
from matplotlib.ticker import LogFormatter, LogLocator, LogFormatterSciNotation

# ========================
# 参数配置
# ========================
# np.random.seed(42)

# 数据加载（使用稀疏矩阵格式）
def load_sparse_matrix(file_path):
    dense_matrix = np.loadtxt(file_path)
    return csr_matrix(dense_matrix)

A = load_sparse_matrix(os.path.join(os.path.dirname(__file__), 'M_H1_final.txt'))
u_true = np.loadtxt(os.path.join(os.path.dirname(__file__), 'coe_H1_four.txt'))
jmn_values = np.loadtxt(os.path.join(os.path.dirname(__file__), 'jmn_values.txt'))
input_index = np.loadtxt(os.path.join(os.path.dirname(__file__), 'input_index.txt'))

input_dim = A.shape[1]
output_dim = A.shape[0]

n_samples = 100
m = np.zeros(input_dim)

# 稀疏矩阵预处理
# 构造Omega^{-1}， 在这个例子中就是Laplace算子 
Omega_inv = eye(input_dim, format='csr') # 单位矩阵的逆是其本身

for i in range(Omega_inv.shape[0]):
    i_m = int(np.abs(input_index[i,0]))
    i_n = int(input_index[i,1])
    eigen = np.power(jmn_values[i_m,i_n-1],2)
    Omega_inv[i,i] = eigen



A_T = A.T.tocsr()  # 提前计算转置矩阵

# ========================
# 优化后的ARM计算
# ========================
def ARM_sparse(A, A_T, Omega_inv, u, m, input_dim, output_dim, n_samples, alpha, T):
    N = 100
    dt = T/N
    samples = np.zeros((n_samples, N + 1, input_dim))
    samples[:, 0, :] = m
    # 预计算固定项
    E = A_T.dot(A)  # 保持稀疏性
    U_fix = E.dot(u)
    
    # 预分解矩阵以提高求解速度
    time_steps = np.arange(1, N + 1) * dt
    solvers = [factorized(Omega_inv * alpha + t * E) for t in time_steps]
    
    for k in range(n_samples):
        x_curr = m.copy()
        for i in range(N):
            t_idx = i
            solver = solvers[t_idx]
            
            # 生成随机项
            dW = np.sqrt(dt) * np.random.randn(output_dim)
            
            # 计算右侧项
            rhs = dt*(U_fix - E.dot(x_curr)) + A_T.dot(dW)
            
            # 求解线性系统
            d_error = solver(rhs)
            # 更新状态
            x_next = x_curr + d_error
            samples[k, i+1, :] = x_next
            x_curr = x_next
    
    # 计算RMSE
    errors = np.linalg.norm(samples[:, -1, :] - u_true, axis=1)
    return np.sqrt(np.mean(errors**2))

def ARM_sparse_after(A, A_T, Omega_inv, u, m, input_dim, output_dim, n_samples, alpha, T):
    N = 100
    dt = T/N
    samples = np.zeros((n_samples, N + 1, input_dim))
    samples[:, 0, :] = m
    # 预计算固定项
    E = A_T.dot(A)  # 保持稀疏性
    U_fix = E.dot(u)
    
    # 预分解矩阵以提高求解速度
    time_steps = np.arange(1, N + 1) * dt
    solvers = [factorized(Omega_inv * alpha + t * E) for t in time_steps]
    
    for k in range(n_samples):
        x_curr = m.copy()
        for i in range(N):
            t_idx = i
            solver = solvers[t_idx]
            
            # 生成随机项
            dW = np.sqrt(dt) * np.random.randn(output_dim)
            
            # 计算右侧项
            rhs = dt*(U_fix - E.dot(x_curr)) + A_T.dot(dW)
            
            # 求解线性系统
            d_error = solver(rhs)
            # 更新状态
            x_next = x_curr + d_error
            samples[k, i+1, :] = x_next
            x_curr = x_next
    
    # 计算RMSE
    mean_over_samples = np.mean(samples[:,-1,:], axis=0)
    return mean_over_samples
# ========================
# 并行优化部分
# ========================
def process_task(params):
    T, alpha = params
    return T, alpha, ARM_sparse(A, A_T, Omega_inv, u_true, m, 
                               input_dim, output_dim, n_samples, alpha, T)

def parallel_optimization():
    # 参数配置
    alpha_series = np.array([0.05 * (2 ** i) for i in range(20)])
    T_series = np.array([10* (2 ** i) for i in range(13)])
    param_grid = list(product(T_series, alpha_series))
    
    # 优化并行参数
    # n_cores = min(mp.cpu_count(), 8)  # 限制核心数防止内存溢出
    n_cores = 56
    chunk_size = max(1, len(param_grid) // (n_cores*4))  # 优化任务分块
    
    with mp.Pool(processes=n_cores) as pool:
        results = pool.imap(process_task, param_grid, chunksize=chunk_size)
        
        # 实时处理结果节省内存
        dtype = [('T', float), ('alpha', float), ('rmse', float)]
        results_array = np.zeros(len(param_grid), dtype=dtype)
        
        for idx, result in enumerate(results):
            results_array[idx] = result
    
    # 按T分组寻找最优解
    Talpha = []
    for T in np.unique(results_array['T']):
        mask = results_array['T'] == T
        subset = results_array[mask]
        optimal = subset[np.argmin(subset['rmse'])]
        Talpha.append([optimal['T'], optimal['alpha'], optimal['rmse']])
    
    return np.array(Talpha)

    
# ========================
# 可视化部分（保持不变）
# ========================
def visualize_results(Talpha):
    T_vals = Talpha[:, 0]
    alpha_vals = Talpha[:, 1]
    rmse_vals = Talpha[:, 2]

    plt.figure(figsize=(10, 6))
    
    # ========================
    # 第一个子图：Optimal α vs T
    # ========================
    plt.subplot(1, 2, 1)
    plt.loglog(T_vals, alpha_vals, 'ro-', base=2)  # 改为以2为底
    plt.xlabel('T'), plt.ylabel('Optimal α'), plt.grid(True)
    plt.title('Optimal α vs T')
    
    ax1 = plt.gca()
    from matplotlib.ticker import LogFormatterSciNotation, LogLocator
    
    # 设置2为底的对数坐标
    ax1.set_xscale('log', base=2)
    ax1.set_yscale('log', base=2)
    
    # 自定义刻度定位器
    def custom_loglocator(base, numticks):
        return LogLocator(base=base, subs=(1.0,), numticks=numticks)
    
    # 设置三个主刻度
    ax1.yaxis.set_major_locator(custom_loglocator(base=2, numticks=7))
    
    # 自动调整坐标范围
    ymin = 2**np.floor(np.log2(np.min(alpha_vals)))
    ymax = 2**np.ceil(np.log2(np.max(alpha_vals)))
    ax1.set_ylim(ymin, ymax)
    
    # ========================
    # 第二个子图： RMSEs vs T
    # ========================
    plt.subplot(1, 2, 2)
    plt.loglog(T_vals, rmse_vals, 'bo-')
    plt.xlabel('T'), plt.ylabel('RMSEs'), plt.grid(True)
    plt.title('RMSEs vs T')
    
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), 'optimization_results.png'), dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    start_time = time.time()
    
    # 预计算优化项
    print("Precomputing sparse matrices...")
    A.sort_indices()  # 优化稀疏矩阵访问速度
    A_T.sort_indices()
    
    # 执行优化
    print("Starting parallel optimization...")
    Talpha = parallel_optimization()
    # row_Talpha = Talpha.shape[0]
    
    # m_N = ARM_sparse_after(A, A_T, Omega_inv, u_true, m, input_dim, output_dim, n_samples, Talpha[0,1], Talpha[0,0])
    # print(f"m_N saved to {output_file}")
    m_di_N = np.zeros((input_dim,Talpha.shape[0]))
    for num in range(Talpha.shape[0]):
        Time = Talpha[num,0]
        Tal = Talpha[num,1]
        m_di_N[:,num] = ARM_sparse_after(A, A_T, Omega_inv, u_true, m, input_dim, output_dim, n_samples, Tal, Time)
    Tcoe = m_di_N[:,-1]
    output_file = os.path.join(os.path.dirname(__file__), 'Tcoe.txt')
    np.savetxt(output_file, Tcoe)


    # 保存 Talpha 到文本文件
    output_file = os.path.join(os.path.dirname(__file__), 'Talpha.txt')
    np.savetxt(output_file, Talpha)
    print(f"Talpha saved to {output_file}")

    # 结果处理
    visualize_results(Talpha)
    end_time = time.time()
    print(f"Total running time: {end_time - start_time:.2f} seconds")