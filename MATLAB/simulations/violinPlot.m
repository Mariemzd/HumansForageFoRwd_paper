function violinPlot(data, varargin)
% violinPlot - Create a violin plot to visualize data distribution.
%
% A violin plot is a combination of a box plot and a kernel density plot.
% It shows the probability density of the data at different values.
%
% SYNTAX:
%   violinPlot(data)
%   violinPlot(data, 'param', value, ...)
%
% INPUTS:
%   data - A numeric matrix. Each column of the matrix will be plotted as a
%          separate violin.
%
% OPTIONAL PARAMETERS:
%   'Colors'     - A matrix of RGB triplets for the violin colors. If not
%                  specified, it will use MATLAB's default color order.
%                  Example: [1 0 0] for red.
%   'Width'      - The maximum width of each violin. Default is 0.4.
%   'ShowData'   - (true/false) If true, overlays the raw data points as a
%                  scatter plot. Default is true.
%   'ShowMedian' - (true/false) If true, shows the median as a white dot.
%                  Default is true.
%
% EXAMPLE:
%   % Generate some sample data
%   data1 = randn(100, 1);
%   data2 = [randn(50, 1) - 2; randn(50, 1) + 2];
%   data_matrix = [data1, data2];
%
%   % Create a violin plot
%   figure;
%   violinPlot(data_matrix, 'Colors', [0 0.4470 0.7410; 0.8500 0.3250 0.0980]);
%   set(gca, 'xtick', [1 2], 'xticklabel', {'Group 1', 'Group 2'});
%   title('Example Violin Plot');
%
% Author: Gemini
% Date: 2025-09-22

% --- Input Parser ---
p = inputParser;
addOptional(p, 'Colors', get(gca, 'ColorOrder'), @isnumeric);
addOptional(p, 'Width', 0.4, @isscalar);
addOptional(p, 'ShowData', true, @islogical);
addOptional(p, 'ShowMedian', true, @islogical);
parse(p, varargin{:});

% Get parsed inputs
colors = p.Results.Colors;
maxWidth = p.Results.Width;
showData = p.Results.ShowData;
showMedian = p.Results.ShowMedian;

% --- Main Plotting Loop ---
hold on;
num_violins = size(data, 2);

for i = 1:num_violins
    % Get data for the current violin and remove NaNs
    current_data = data(:, i);
    current_data = current_data(~isnan(current_data));
    
    % Use ksdensity to get the kernel density estimate
    [f, xi] = ksdensity(current_data);
    
    % Scale the density to the desired maximum width
    f_scaled = f / max(f) * maxWidth;
    
    % --- Draw the Violin Patch ---
    % Get the color for the current violin
    if size(colors, 1) < i
        violin_color = colors(end, :); % Reuse last color if not enough are specified
    else
        violin_color = colors(i, :);
    end
    
    % Draw the right half of the violin
    patch(i + f_scaled, xi, violin_color, 'FaceAlpha', 0.5, 'EdgeColor', 'none');
    % Draw the left half of the violin
    patch(i - f_scaled, xi, violin_color, 'FaceAlpha', 0.5, 'EdgeColor', 'none');
    
    % --- Overlay Data Points and Median ---
    if showData
        % Add a small amount of random jitter for better visualization
        jitter = (rand(size(current_data)) - 0.5) * maxWidth * 0.2;
        scatter(i + jitter, current_data, 15, 'filled', ...
            'MarkerFaceColor', [0.5 0.5 0.5], 'MarkerFaceAlpha', 0.6);
    end
    
    if showMedian
        median_val = median(current_data);
        plot([i - maxWidth*0.2, i + maxWidth*0.2], [median_val, median_val], ...
            'Color', 'w', 'LineWidth', 2);
    end
end

% --- Final Axis Formatting ---
hold off;
box on;
xlim([0.5, num_violins + 0.5]); % Set x-axis limits
set(gca, 'xtick', 1:num_violins); % Set x-axis ticks to center of violins

end
