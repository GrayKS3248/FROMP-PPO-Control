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
T0 = 293.15:2.0:309.15;
a0 = 0.05:0.02:0.09;
alpha_colors = ['b' 'k' 'r'];

%% Solve Speed and Temperature Fields
%=================================================================================================================%
v_f_e0_list = zeros([length(T0), length(a0)]);
t_max_e0_list = zeros([length(T0), length(a0)]);
e_list = zeros([length(T0), length(a0)]);
v_f_list = zeros([length(T0), length(a0)]);
t_max_list = zeros([length(T0), length(a0)]);

T0_count = 1;
a0_count = 1;
for curr_T0=T0
    a0_count = 1;
    for curr_a0=a0
        [v_f_e0, v_f, e] = get_vf(curr_T0, curr_a0, R, k, p, Hr, Cp, A, E, n, m, C, ac);
        
        v_f_e0_list(T0_count, a0_count) = v_f_e0;
        v_f_list(T0_count, a0_count) = v_f;
        e_list(T0_count, a0_count) = e;
        
        t_max_e0_list(T0_count, a0_count) = get_t_max(curr_T0, curr_a0, 0.0, Hr, Cp) - 273.15;
        t_max_list(T0_count, a0_count) = get_t_max(curr_T0, curr_a0, e, Hr, Cp) - 273.15;
        
        a0_count = a0_count + 1;
    end
    T0_count = T0_count + 1;
end


%% Plot Solution
%=================================================================================================================%
figure(1)
for i=1:length(a0)
    plot(T0-273.15, t_max_e0_list(:,i),'LineWidth',2.0,'Color',alpha_colors(i))
    hold on
end
hold off
title("Temperature of Polymerization",'FontSize',14)
xlabel("Initial Temperature [C]",'FontSize',12)
ylabel("Temperature [C]",'FontSize',12)
xlim([T0(1)-273.15, T0(end)-273.15])
legend("$\alpha_0$="+string(a0), 'Location','northeastoutside', 'Interpreter', 'LaTeX','FontSize',12)
grid on
saveas(gcf(),"temp_of_rxn.png")
close()

figure(2)
for i=1:length(a0)
    plot(T0-273.15, 1000.0*v_f_list(:,i),'LineWidth',2.0,'Color',alpha_colors(i))
    hold on
end
hold off
title("Steady State Front Speed",'FontSize',14)
xlabel("Initial Temperature [C]",'FontSize',12)
ylabel("Speed [mm/s]",'FontSize',12)
xlim([T0(1)-273.15, T0(end)-273.15])
legend("$\alpha_0$="+string(a0), 'Location','northeastoutside', 'Interpreter', 'LaTeX','FontSize',12)
grid on
saveas(gcf(),"front_speed.png")
close()

figure(3)
for i=1:length(a0)
    plot(T0-273.15, e_list(:,i),'LineWidth',2.0,'Color',alpha_colors(i))
    hold on
end
hold off
title("ε Parameter",'FontSize',14)
xlabel("Initial Temperature [C]",'FontSize',12)
ylabel("ε [-]",'FontSize',12)
xlim([T0(1)-273.15, T0(end)-273.15])
legend("$\alpha_0$="+string(a0), 'Location','northeastoutside', 'Interpreter', 'LaTeX','FontSize',12)
grid on
saveas(gcf(),"epsilon_parameter.png")
close()

%% Functions
%=================================================================================================================%
function t_max = get_t_max(T0, a0, e, Hr, Cp)
    t_max = T0 + ((1-a0-e) * Hr) / (Cp);
end

function PI = get_PI(a0, e, n, m, C, ac)
    integrand = @(y) (y .* (( (y+e).^n .* (1-e-y).^m ).^-1.0)) .* (1 + exp( C*(1-e-ac-y) ) );
    PI = integral(integrand,0,1-e-a0);
end

function [v_f_e0, v_f, e_max] = get_vf(T0, a0, R, k, p, Hr, Cp, A, E, n, m, C, ac)
    t_max = @(e) get_t_max(T0, a0, e, Hr, Cp);
    PI = @(e) get_PI(a0, e, n, m, C, ac);
    v_f = @(e) -sqrt((A*k)/(p*Hr) * (R*t_max(e)^2)/(E) * exp((-E)/(R*t_max(e))) * 1.0/PI(e));
    e0 = 0.3;
    v_f_e0 = -v_f(e0);
    [e_min,vf_min] = fminsearch(v_f,e0);
    v_f = -vf_min;
    e_max = e_min;
end