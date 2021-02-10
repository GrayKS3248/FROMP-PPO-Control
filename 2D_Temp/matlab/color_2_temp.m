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
red_fit_1 = fit(temp(1:7), red(1:7), 'fourier1');
x_val_1 = linspace(30.0,temp(7),100);
y_val_1 = red_fit_1(x_val_1);
y_val_1(y_val_1<=0.0)=0.0;
red_fit_2 = fit(temp(12:23), red(12:23), 'fourier2');
x_val_2 = linspace(temp(12),36.0,100);
y_val_2 = red_fit_2(x_val_2);
y_val_2(y_val_2<=0.0)=0.0;
plot(x_val_1, y_val_1,'k','linewidth',2.0)
hold on
plot(temp,red,'r','linewidth',2.0)
plot(x_val_2, y_val_2,'k','linewidth',2.0)
xlim([29.0,36.0])
ylim([0.0,255.0])
xlabel("Temperature [C]")
ylabel("Spectrum Intensity [-]")
title("Red Fit",'FontSize',14)
legend('Fit','Data')
saveas(gcf, "Red_Fit.png")
close all

% Get green fit
green_fit = fit(temp, green, 'fourier3');
x_val = linspace(29.0,36.0,100);
y_val = green_fit(x_val);
y_val(y_val<=0.0)=0.0;
plot(x_val, y_val,'k','linewidth',2.0)
hold on
plot(temp,green,'g','linewidth',2.0)
xlim([29.0,36.0])
ylim([0.0,255.0])
xlabel("Temperature [C]")
ylabel("Spectrum Intensity [-]")
title("Green Fit",'FontSize',14)
legend('Fit','Data')
saveas(gcf, "Green_Fit.png")
close all

% Get blue fit
blue_fit = fit(temp, blue, 'fourier4');
x_val = linspace(29.0,36.0,100);
y_val = blue_fit(x_val);
y_val(y_val<=0.0)=0.0;
plot(x_val, y_val,'k','linewidth',2.0)
hold on
plot(temp,blue,'b','linewidth',2.0)
xlim([29.0,36.0])
ylim([0.0,255.0])
xlabel("Temperature [C]")
ylabel("Spectrum Intensity [-]")
title("Blue Fit",'FontSize',14)
legend('Fit','Data')
saveas(gcf, "Blue_Fit.png")
close all