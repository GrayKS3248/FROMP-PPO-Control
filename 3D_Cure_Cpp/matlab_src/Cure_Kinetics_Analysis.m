%% Startup
%=================================================================================================================%
close all
clear


%% Physical Parameters
%=================================================================================================================%
R = 8.314;      % Joules / Mol * Kelvin
k = 0.152;      % Watts / Meter * Kelvin
p = 980.0;      % Kilograms / Meter ^ 3
Hr = 350000.0;  % Joules / Kilogram
Cp = 1600.0;    % Joules / Kilogram * Kelvin
A = 8.55e15;    % 1 / Seconds
E = 110750.0;   % Joules / Mol
n = 1.72;       % Unitless
m = 0.77;       % Unitless
C = 14.48;      % Unitless
ac = 0.41;      % Decimal Percent


%% Solution Space Parameters
%=================================================================================================================%
a0 = 0.05;
T0 = 293.15;


%% Plotting Options
%=================================================================================================================%
use_log_plot = true;
n_levels = 12;


%% Define and Solve Optimization Problem
%=================================================================================================================%
enforcement_point = 0.1;
T_max = (Hr * (1 - a0) / Cp) + T0;

T = @(vf,gt,bt,x,t) -(T0+(T_max-T0)*((1+exp(-gt.*(x-vf.*t))).^(bt.^-1)).^(-1)) + T0 + T_max;
syms vf gt bt x t;
dT_dt = matlabFunction(diff(T(vf,gt,bt,x,t),t),'Vars',[vf,gt,bt,x,t]);
laplace_T = matlabFunction(diff(T(vf,gt,bt,x,t),x,2),'Vars',[vf,gt,bt,x,t]);
clear vf gt bt x t

alpha = @(vf,ga,ba,da,x,t)  -(a0+(1-a0)*((1+exp(-ga.*(x+da-vf.*t))).^(ba.^-1)).^(-1)) + a0 + 1;
syms vf ga ba da x t;
dalpha_dt = matlabFunction(diff(alpha(vf,ga,ba,da,x,t),t),'Vars',[vf,ga,ba,da,x,t]);
clear vf ga ba da x t;
dalpha_dt_kinetics = @(vf,gt,bt,ga,ba,da,x,t) A*exp(-E*(R*T(vf,gt,bt,x,t)).^(-1.0)) .* (1-alpha(vf,ga,ba,da,x,t)).^(n) .* (alpha(vf,ga,ba,da,x,t)).^(m) .* (1+exp(C*(alpha(vf,ga,ba,da,x,t)-ac))).^(-1.0);

heat_equation_residual = @(vf,gt,bt,ga,ba,da,x,t) 1e-30*trapz(t, abs(k*laplace_T(vf,gt,bt,x,t) + p*Hr*dalpha_dt(vf,ga,ba,da,x,t) - p*Cp*dT_dt(vf,gt,bt,x,t)));
cure_kinetics_residual = @(vf,gt,bt,ga,ba,da,x,t) trapz(t, abs(dalpha_dt(vf,ga,ba,da,x,t) - dalpha_dt_kinetics(vf,gt,bt,ga,ba,da,x,t)));
          
total_error = @(x) heat_equation_residual(abs(x(1)),abs(x(2)),abs(x(3)),abs(x(4)),abs(x(5)),abs(x(6)),enforcement_point,((abs(x(2))*enforcement_point+log((1000/999).^(abs(x(3)))-1)).*((abs(x(1)).*abs(x(2))).^-1)):0.01:((abs(x(2))*enforcement_point+log((1000).^(abs(x(3)))-1)).*((abs(x(1)).*abs(x(2))).^-1))) + ...
                   cure_kinetics_residual(abs(x(1)),abs(x(2)),abs(x(3)),abs(x(4)),abs(x(5)),abs(x(6)),enforcement_point,((abs(x(2))*enforcement_point+log((1000/999).^(abs(x(3)))-1)).*((abs(x(1)).*abs(x(2))).^-1)):0.01:((abs(x(2))*enforcement_point+log((1000).^(abs(x(3)))-1)).*((abs(x(1)).*abs(x(2))).^-1)));
               
%     1       2       3     4        5     6
%     vf      gt      bt    ga       ba    da
x0 = [0.001,  10000,  5,   10000,   5,    0.0];
ms = MultiStart('FunctionTolerance',0,'UseParallel',true);
gs = GlobalSearch(ms,'MaxWaitCycle',500,'NumStageOnePoints',200,'NumTrialPoints',1000000,'XTolerance',1e-14);
problem = createOptimProblem('fmincon','x0',x0,'objective',total_error,'lb',[0.0001,1000,1,1000,1,0],'ub',[0.01,50000,50,50000,50,0.001]);
[soln,fg,exitflag,output,solns] = run(gs,problem);

min_t = ((abs(soln(2))*enforcement_point+log((1000/999)^(abs(soln(3)))-1))*((abs(soln(1))*abs(soln(2)))^-1));
max_t = ((abs(soln(2))*enforcement_point+log((1000)^(abs(soln(3)))-1))*((abs(soln(1))*abs(soln(2)))^-1));
t = min_t:0.01:max_t;
min_pos = ((abs(soln(1))*abs(soln(2))*mean(t)-log((1000)^(abs(soln(3)))-1))*((abs(soln(2)))^-1));
max_pos = ((abs(soln(1))*abs(soln(2))*mean(t)-log((1000/999)^(abs(soln(3)))-1))*((abs(soln(2)))^-1));
pos = min_pos:0.0001:max_pos;


%% Save Optimization Solution as Table
%=================================================================================================================%
Param = [ "V_f"; "k_T"; "v_T"; "k_a"; "v_a"; "d_a" ];
Unit = [ "[mm/s]"; "[1/m]"; "[-]"; "[1/m]"; "[-]"; "[um]" ];
Value = [round(1e3*soln(1),3); round(soln(2),3); round(soln(3),3); round(soln(4),3); round(soln(5),3); round(1e6*soln(6),3)];
soln_table = table(Param,Unit,Value);
writetable(soln_table,'soln.csv');


%% Plot Optimization Results
%=================================================================================================================%
figure(1)
plot(t-min_t,dalpha_dt(soln(1),soln(4),soln(5),soln(6),enforcement_point,t),'LineWidth',3.0,'color','r')
hold on
plot(t-min_t,dalpha_dt_kinetics(soln(1),soln(2),soln(3),soln(4),soln(5),soln(6),enforcement_point,t),'LineWidth',3.0,'LineStyle',':','color','k')
legend({'Ansatz','Kinetics'},'Location','northeastoutside')
title('Cure Rate Adherence at Enforcement Point')
xlabel('Time [s]')
ylabel('Cure rate [1/s]')
hold off
saveas(gcf, "Cure_Rate_Adherence.png")

figure(2)
plot(t-min_t,(p*Cp)^-1*(k*laplace_T(soln(1),soln(2),soln(3),enforcement_point,t) + p*Hr*dalpha_dt(soln(1),soln(4),soln(5),soln(6),enforcement_point,t)),'LineWidth',3.0,'color','r')
hold on
plot(t-min_t,dT_dt(soln(1),soln(2),soln(3),enforcement_point,t),'LineWidth',3.0,'LineStyle',':','color','k')
legend({'LHS','RHS'},'Location','northeastoutside')
title('Heat Equation Adherence at Enforcement Point')
xlabel('Time [s]')
ylabel('Temperature rate [K/s]')
hold off
saveas(gcf, "Heat_Eqn_Adherence.png")

figure(3)
yyaxis left
plot(t-min_t,alpha(soln(1),soln(4),soln(5),soln(6),enforcement_point,t),'LineWidth',3.0,'color','r')
ylabel('Degree cure [-]')
yyaxis right
plot(t-min_t,(T(soln(1),soln(2),soln(3),enforcement_point,t)-T0)/(T_max-T0),'LineWidth',3.0,'color','b')
ylabel('\Theta [-]')
title('Temperature and Cure at Enforcement Point')
xlabel('Time [s]')
ax = gca;
ax.YAxis(1).Color = 'r';
ax.YAxis(2).Color = 'b';
saveas(gcf, "T_a_Time.png")

figure(4)
yyaxis left
plot(1000.0*(pos-enforcement_point),alpha(soln(1),soln(4),soln(5),soln(6),pos,mean(t)),'LineWidth',3.0,'color','r')
ylabel('Degree cure [-]')
yyaxis right
plot(1000.0*(pos-enforcement_point),(T(soln(1),soln(2),soln(3),pos,mean(t))-T0)/(T_max-T0),'LineWidth',3.0,'color','b')
ylabel('\Theta [-]')
title('Temperature and Cure Around Front')
xlabel('X Location [mm]')
ax = gca;
ax.YAxis(1).Color = 'r';
ax.YAxis(2).Color = 'b';
saveas(gcf, "T_a_Position.png")


%% Calculate Cure Rate in Parameter Space
%=================================================================================================================%
T_linspace = T0:1:T_max;
a_linspace = a0:0.001:1.0;
[a_grid, T_grid] = meshgrid(a_linspace, T_linspace);
da_dt = zeros([length(T_linspace), length(a_linspace)]);
temperature_index = 1;
for curr_temperature = T_linspace
    alpha_index = 1;
    for curr_alpha = a_linspace
        da_dt(temperature_index, alpha_index) = A*exp(-E/(R*curr_temperature)) * (1-curr_alpha)^n * curr_alpha^m * (1.0 + exp(C*(curr_alpha-ac)))^(-1.0);
        alpha_index = alpha_index + 1;
    end 
    temperature_index = temperature_index + 1;
end


%% Plot Cure Curve in Parameter Space
%=================================================================================================================%
if use_log_plot
    da_dt = log10(da_dt);
end
level_spacing = (max(max(da_dt(da_dt~=Inf))) - min(min(da_dt(da_dt~=-Inf)))) / (n_levels);
levels = min(min(da_dt(da_dt~=-Inf))):level_spacing:max(max(da_dt(da_dt~=Inf)));
if ~use_log_plot
    levels = round(levels,0);
else
    levels = round(levels,3);
end
figure('Renderer', 'painters', 'Position', [600 300 850 650])
[M,c] = contourf(T_grid-273.15,a_grid,(da_dt),levels);
hold on
plot(T(soln(1),soln(2),soln(3),enforcement_point,t)-273.15, alpha(soln(1),soln(4),soln(5),soln(6),enforcement_point,t),'LineWidth',3.0,'color','k');
hold off
colormap(parula(n_levels))
cbar = colorbar();
cbar.Ticks = c.LevelList;
cbar.TickLength = 0.04;
cbar.Label.String = "log_{10}(\partial\alpha/\partialt)";
cbar.FontSize = 16;
cbar.Label.FontSize = 16;
xlabel("Temperature [C]",'FontSize',16)
ylabel("\alpha [-]",'FontSize',16)
ax = gca;
ax.FontSize = 16;
axis square
ax_pos = get(gca, 'Position');
xoffset = -0.04;
ax_pos(1) = ax_pos(1) + xoffset;
set(gca, 'Position', ax_pos)
title("Cure Kinetics for DCPD 100ppm G2")
saveas(gcf, "Cure_Kinetics.png")