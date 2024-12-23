clear all
close all

a=load("C:\Users\rayce\Downloads\plot\Model_eval\Pitch\model_2_train.txt");
b=load("C:\Users\rayce\Downloads\plot\Model_eval\Pitch\model_2_valid.txt");
epoch=1:1:60

figure
plot(epoch, a, '-k', 'LineWidth', 2)
hold on 
plot(epoch, b, '-.', 'color',[100/256, 100/256, 100/256],'LineWidth', 2)

h=legend('Training','Validation');
set(h,'FontSize',16, 'FontWeight', 'bold', 'fontname', 'Times new roman', 'box', 'off');

xlim([0 60])
ylim([0 0.03])

set(gca,'fontsize',13);
h=xlabel('                                                           Epoch');
set(h,'FontSize',16, 'FontWeight', 'bold', 'fontname', 'Times new roman');
h=ylabel('                                          Loss');
set(h,'FontSize',16, 'FontWeight', 'bold', 'fontname', 'Times new roman'); 
box off
