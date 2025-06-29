Talpha = load("TalphaH12.txt");
T = Talpha(:,1);
alpha = Talpha(:,2);
rmse = Talpha(:,3);

% 分屏绘制
figure;

% ================== 左图优化 ==================
subplot(1,2,1);
loglog(T, alpha, 'r-', 'LineWidth', 1.0);
grid on;
xlabel('T'); ylabel('\eta', 'Interpreter','tex');
title('optimal \eta vs T');

% 动态生成对数刻度（示例范围10^1到10^3）
y_min = -1;  % log10(min(alpha))
y_max = 1;  % log10(max(alpha))
yticks(10.^(y_min:0.2:y_max));  % 每半个数量级一个刻度
yticklabels(cellstr(num2str((y_min:0.2:y_max)', '10^{%.1f}')));  % 5个刻度

% ================== 右图优化 ==================
subplot(1,2,2);
loglog(T, rmse, 'b-', 'LineWidth', 1.0);
set(gca, 'YScale', 'log');
grid on;
xlabel('T'); ylabel('RMSE');
title('RMSE vs T');

% 自定义科学计数刻度（覆盖3e-2到1.3e-1）
yticks([0.2, 0.4, 0.6, 0.8, 2.0]);  % 5个关键点
yticklabels({'\fontsize{9}2×10^{-1}', '4×10^{-1}', '6×10^{-1}','8×10^{-1}','2×10^{0}',});

% 统一美化设置
set(findall(gcf,'Type','axes'), 'FontName','Arial', 'FontSize',15)
saveas(gcf,'opreH1ex1','jpg')