% -- POSITIONS --

function xy_positions_train = make_position_matrix(data,index1,index2,bias)
% Initialize
bin_size = 20;
n_bins = floor(length(data(index1,index2).handPos)/bin_size); 

% Positions form 320ms to end
xy_positions_train = zeros(2,n_bins+1);
xy_positions_train(:,1:n_bins) = data(index1,index2).handPos(1:2,(1:n_bins)*bin_size);    
xy_positions_train = xy_positions_train(:,bias+1:n_bins); 
end