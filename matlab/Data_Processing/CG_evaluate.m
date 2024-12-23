function [mean_CQ] = CG_evaluate(str_, range, rate)

info = audioinfo(str_);
[y,Fs] = audioread(str_);
size_=size(y,1);

% y1=y(:,1);
y=y(:,2);
t = 0:seconds(1/Fs):seconds(info.Duration);
t = t(1:end-1);

a = size_ / 10 * 2;
b = size_ / 10 * 8;

a = a - mod(a,1);
b = b - mod(b,1);


y = y(a:b);
n = size(y,1);
y_index=1:1:n;
min_ = min(y);
max_ = max(y);

amplitude_ = max_ - min_;
%{
figure
plot(y)
title(str_)
hold on
%}

count_interval=1;
xx_interest(10)=0; 
yy_interest(10)=0;
clear mark_

for i=range:n-range
    
    for j=1:range*2
        xx_interest(j) = i + j - range;
        yy_interest(j) = y(xx_interest(j));
    end
   [min_value, index_]=min(yy_interest);
   if (i==xx_interest(index_) && min_value < min_+amplitude_*rate)
        mark_(count_interval) = i;
        %local_max(count_interval)=max_;
        count_interval = count_interval + 1;
        %plot(i, min_value, 'ok');
   end
    
end

ninterval = size(mark_, 2) - 1;

for i=1:ninterval
    sub_length = mark_(i+1) - mark_(i);
    
    local_min_ = y(mark_(i));
    local_max_ = max(y(mark_(i):mark_(i+1)));
    contact_creteria = local_min_ + 0.5 *(local_max_- local_min_);

    count_ = 0; 
    for t =1:sub_length
        if y( mark_(i)+t) > contact_creteria 
            count_ = count_ + 1;
        end
    end
    CQ(i) = count_ / sub_length;

end

mean_CQ = mean(CQ);
 