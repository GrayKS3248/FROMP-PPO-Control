% Set thresholding limit
limit = 5.0;

% Import data
color_grad = csvread('Color_Grad.txt', 0,0);
temp = 35.0 - 5.0 * color_grad(:,1);
red = color_grad(:,2);
green = color_grad(:,3);
blue = color_grad(:,4);

% Plot experimental data
plot(temp,red,'r','linewidth',2.0)
hold on
plot(temp,green,'g','linewidth',2.0)
plot(temp,blue,'b','linewidth',2.0)
hold off
xlabel("Temperature [C]")
ylabel("Spectrum Intensity [-]")
xlim([30.0,35.0])
ylim([0.0,255.0])
title("Temperature - RGB",'FontSize',14)
saveas(gcf, "Experimental_Data.png")
close all

% Get red fit
red(red<=limit) = 0.0;
non_zero_red = [0;red(red~=0.0);0];
front=false;
while length(non_zero_red) > length(red)
    if front
        non_zero_red = non_zero_red(2:end);
    else
        non_zero_red = non_zero_red(1:end-1);
    end
    front = ~front;
end
non_zero_red_temp = zeros(length(red),1);
for i = 1:length(red)
    if red(i) ~= 0.0
        non_zero_red_temp(i) = 1.0;
    elseif i+1<=length(red) && red(i)==0.0 && red(i+1)~=0.0
        non_zero_red_temp(i) = 1.0;
    elseif red(i-1)~=0.0 && red(i)==0.0
        non_zero_red_temp(i) = 1.0;
    end
end
non_zero_red_temp = temp(logical(non_zero_red_temp));
red_fit = fit(non_zero_red_temp, non_zero_red, 'fourier2');
x_val = linspace(non_zero_red_temp(1),non_zero_red_temp(end),100);
y_val = red_fit(x_val);
y_val(y_val<=0.0)=0.0;
plot(x_val, y_val,'k','linewidth',2.0)
hold on
plot(temp,red,'r','linewidth',2.0)
xlim([30.0,35.0])
ylim([0.0,255.0])
xlabel("Temperature [C]")
ylabel("Spectrum Intensity [-]")
title("Red Fit",'FontSize',14)
legend('Fit','Data')
saveas(gcf, "Red_Fit.png")
close all

% Get green fit
green(green<=limit) = 0.0;
non_zero_green = [0;green(green~=0.0);0];
front=false;
while length(non_zero_green) > length(green)
    if front
        non_zero_green = non_zero_green(2:end);
    else
        non_zero_green = non_zero_green(1:end-1);
    end
    front = ~front;
end
non_zero_green_temp = zeros(length(green),1);
for i = 1:length(green)
    if green(i) ~= 0.0
        non_zero_green_temp(i) = 1.0;
    elseif i+1<=length(green) && green(i)==0.0 && green(i+1)~=0.0
        non_zero_green_temp(i) = 1.0;
    elseif green(i-1)~=0.0 && green(i)==0.0
        non_zero_green_temp(i) = 1.0;
    end
end
non_zero_green_temp = temp(logical(non_zero_green_temp));
green_fit = fit(non_zero_green_temp, non_zero_green, 'fourier2');
x_val = linspace(non_zero_green_temp(1),non_zero_green_temp(end),100);
y_val = green_fit(x_val);
y_val(y_val<=0.0)=0.0;
plot(x_val, y_val,'k','linewidth',2.0)
hold on
plot(temp,green,'g','linewidth',2.0)
xlim([30.0,35.0])
ylim([0.0,255.0])
xlabel("Temperature [C]")
ylabel("Spectrum Intensity [-]")
title("Green Fit",'FontSize',14)
legend('Fit','Data')
saveas(gcf, "Green_Fit.png")
close all

% Get blue fit
blue(blue<=limit) = 0.0;
blue(blue<=limit) = 0.0;
non_zero_blue = [0;blue(blue~=0.0);0];
front=false;
while length(non_zero_blue) > length(blue)
    if front
        non_zero_blue = non_zero_blue(2:end);
    else
        non_zero_blue = non_zero_blue(1:end-1);
    end
    front = ~front;
end
non_zero_blue_temp = zeros(length(blue),1);
for i = 1:length(blue)
    if blue(i) ~= 0.0
        non_zero_blue_temp(i) = 1.0;
    elseif i+1<=length(blue) && blue(i)==0.0 && blue(i+1)~=0.0
        non_zero_blue_temp(i) = 1.0;
    elseif blue(i-1)~=0.0 && blue(i)==0.0
        non_zero_blue_temp(i) = 1.0;
    end
end
non_zero_blue_temp = temp(logical(non_zero_blue_temp));
blue_fit = fit(non_zero_blue_temp, non_zero_blue, 'fourier4');
x_val = linspace(non_zero_blue_temp(1),non_zero_blue_temp(end),100);
y_val = blue_fit(x_val);
y_val(y_val<=0.0)=0.0;
plot(x_val, y_val,'k','linewidth',2.0)
hold on
plot(temp,blue,'b','linewidth',2.0)
xlim([30.0,35.0])
ylim([0.0,255.0])
xlabel("Temperature [C]")
ylabel("Spectrum Intensity [-]")
title("Blue Fit",'FontSize',14)
legend('Fit','Data')
saveas(gcf, "Blue_Fit.png")
close all