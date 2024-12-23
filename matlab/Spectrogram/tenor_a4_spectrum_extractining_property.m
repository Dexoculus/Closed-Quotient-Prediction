clear all
clc
close all
% info = audioinfo('1_BrunoMars.wav');
% [y,Fs] = audioread('1_BrunoMars.wav');

info = audioinfo('TenorA4.wav');
[y,Fs] = audioread('TenorA4.wav');
size_=size(y);

y1=y(:,1);
t = 0:seconds(1/Fs):seconds(info.Duration);
t = t(1:end-1);

% 5~6√ ∏∏
dt=info.Duration/size_(1); 
time_step_1sec_=size_(1)/info.Duration;

t5(time_step_1sec_)=0;
z5(time_step_1sec_)=0;
for i=1:time_step_1sec_
    t5(i)=i*dt;
    z5(i)=y1(22*time_step_1sec_+i) ;
end
  
% % 
% figure
% plot(t,y1)
% xlabel('Time')
% ylabel('Audio Signal')
% xlim([seconds(22) seconds(23)])
% 
%  
% % 
figure
plot(t5,z5)
xlabel('Time')
ylabel('Audio Signal') 


figure
s = spectrogram(z5);
spectrogram(z5,'yaxis');
%s have size of  8193  x  8


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 1) Harmonicity
% s=8193 times 8
Nf=8193;
for i=1:Nf
    for k=1:10
        
    end
end

