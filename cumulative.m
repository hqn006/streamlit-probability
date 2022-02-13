% FOR REFERENCE ONLY
%
% Cumulative probability of "at least r successes in n trials"
%
% P(at least r successes) = 1 - P(at most r-1 successes)
%                         = 1 - P(exactly r-1 successes) - P(exactly r-2
%                         successes) ... - P(exactly 0 successes)

clearvars
close all
clc


%% INPUT

P_des = 0.5;  % desired cumulative probability
p = 0.008  ;  % probability of 1 success in a trial
r = 2;        % # of successes
n_max = 1000; % max # of trials


%% MAIN

P_percent_str = [num2str(P_des*100) '%'];


r_minus = r - 1;
N = r_minus:n_max;
P = zeros(size(N));
n_found = -1;
for i = 1:length(N)
    
    sumExactly = 0;
    for j = r_minus:-1:0
        exactly = nchoosek(N(i),j) * p^j * (1-p)^(N(i)-j);
        sumExactly = sumExactly + exactly;
    end
    
    P(i) = 1 - ( sumExactly );
    
    % Store data of desired probability
    if P(i) > P_des && n_found == -1
        n_found = N(i);
        P_found = P(i);
    end
    
end

plot(N,P)
hold on
title(['Cumulative Probability of At Least ' num2str(r) ' Successes in n Trials'])
xlabel('Number of Trials')
ylabel('Cumulative Probability')

yline(P_des, 'r--', P_percent_str, 'HandleVisibility', 'off')
plot(n_found, P_found, 'bo')

legend('P(n)', 'closest pt. to desired', 'Location', 'southeast')


%% EOF
