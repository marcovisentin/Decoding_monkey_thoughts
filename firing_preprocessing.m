% -- FIRING RATE CALCULATION --

function spikes_train = make_spikes_matrix(data,index1,index2,bin_size, n_neurons,lag, bias)

% Initialize
n_bins = floor(length(data(index1,index2).spikes)/20);  % keep them all
spikes_train = zeros(n_neurons,n_bins);
j=0;

for i = 2:n_bins
    j=j+1;
    % Sum spikes across 20 ms
    spikes_train(:, j) = sum(data(index1,index2).spikes(:, (bin_size*(i-1)):((i)*bin_size)),2);
end
% Take only spikes from 320ms to the end. Apply lag.
spikes_train = spikes_train(:,bias-lag+1:n_bins-lag); 
end