% 双图并排程序 (严格保持原图风格)
figure('Position', [100 100 800 600], 'Color', 'w'); % 宽度调整为双倍

% ================== 左图：数据1 ==================
subplot(1,2,1);

% 加载数据
data = load("data_plot_recons_H1_ex1.txt");
x = data(:,1);
y = data(:,2);
gh_value = data(:,3);

% Delaunay三角剖分
tri = delaunay(x,y);
[r, c] = size(tri);
disp(['左图三角形数量: ', num2str(r)]);

% 绘制三角网线
plot(x,y,'-', 'Color', [0.2 0.2 0.2], 'LineWidth', 0.5); 

% 绘制三维曲面
h = trisurf(tri,x,y,gh_value);
axis tight;
view(-37.5, 30); % 固定视角

% 光照设置，调整光照位置
l = light('Position', [-50 -50 40], 'Style', 'infinite'); % 调整光照位置
shading interp;
material dull; % 原始材质效果

% 图形样式
set(gca, 'FontSize', 18, 'LineWidth', 1.5, 'XColor', [0.3 0.3 0.3], 'YColor', [0.3 0.3 0.3]);
colorbar off;
title('Reconstructed Solution', 'FontSize', 20);

% ================== 右图：数据2 ================== 
subplot(1,2,2);

% 加载第二组数据
data = load("true_sol_H1_ex1.txt");
x = data(:,1);
y = data(:,2);
gh_value = data(:,3);

% Delaunay三角剖分
tri = delaunay(x,y);
[r, c] = size(tri);
disp(['右图三角形数量: ', num2str(r)]);

% 绘制三角网线
plot(x,y,'-', 'Color', [0.2 0.2 0.2], 'LineWidth', 0.5);

% 绘制三维曲面
h = trisurf(tri,x,y,gh_value);
axis tight;
view(-37.5, 30); % 完全相同的视角

% 同步光照设置，调整光照位置
l = light('Position', [-50 -50 40], 'Style', 'infinite'); % 调整光照位置
shading interp;
material dull;

% 样式同步
set(gca, 'FontSize', 18, 'LineWidth', 1.5, 'XColor', [0.3 0.3 0.3], 'YColor', [0.3 0.3 0.3]);
colorbar off;
title('True Solution', 'FontSize', 20);

% ================== 专业输出 ==================
exportgraphics(gcf, 'dual_plot.jpg', 'Resolution', 600, 'BackgroundColor', 'white');