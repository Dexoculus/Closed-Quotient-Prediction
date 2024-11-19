clear all
close all

xx = 0:0.01:1;
yy = xx;

x = load("C:\Personal_Folder\Vocal_CQ\Source_Code\x_raw_data.txt");
y = load("C:\Personal_Folder\Vocal_CQ\Source_Code\y_predict_data.txt");

figure
scatter(x, y, 20)
hold on
plot(xx,yy);

h=legend('Predict','Raw');
set(h,'FontSize',16, 'FontWeight', 'bold', 'fontname', 'Times new roman', 'box', 'off');

xlim([0.3 0.8])
ylim([0.3 0.8])

set(gca,'fontsize',13);
h=xlabel('                                                  Exact CQ');
set(h,'FontSize',16, 'FontWeight', 'bold', 'fontname', 'Times new roman');
h=ylabel('                                       Predicted CQ');
set(h,'FontSize',16, 'FontWeight', 'bold', 'fontname', 'Times new roman'); 
box off