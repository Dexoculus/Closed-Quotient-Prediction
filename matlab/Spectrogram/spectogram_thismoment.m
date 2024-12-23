clear all
clc
close all
% info = audioinfo('1_BrunoMars.wav');
% [y,Fs] = audioread('1_BrunoMars.wav');

info = audioinfo('ThisMoment.m4a');
[y,Fs] = audioread('ThisMoment.m4a');
size_=size(y);

y1=y(:,1);
t = 0:seconds(1/Fs):seconds(info.Duration);
t = t(1:end-1);

% 5~6√ ∏∏
dt=info.Duration/size_(1);
time_step_1sec_=size_(1)/info.Duration;
time_step_1sec_=time_step_1sec_-mod(time_step_1sec_,1);

t5(time_step_1sec_)=0;
z5(time_step_1sec_)=0;
for i=1:time_step_1sec_
    t5(i)=i*dt;
    z5(i)=y1(161*time_step_1sec_+i) ;
end
 
% figure
% plot(t,y1)
% xlabel('Time')
% ylabel('Audio Signal')
% xlim([seconds(5) seconds(6)])

 
% 
figure
plot(t5,z5)
xlabel('Time')
ylabel('Audio Signal') 


figure
s = spectrogram(z5);
spectrogram(z5,'yaxis');






