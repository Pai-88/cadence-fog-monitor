%% ─────────────────────────────────────────────────────────────────────────
%  plot_capture.m  —  ENGF0031 Accuracy Worksheet, Task 3
%  Plots ONE capture from cpx_accel_logger.ino with the protocol phases shaded.
%  CSV columns:  idx, t_s, ax_mg, ay_mg, az_mg, phase
%  (the `phase` column is set by the RIGHT button on the board, so shading is
%   automatic — no need to type in region times like Appendix C).
%% ─────────────────────────────────────────────────────────────────────────

%% 1. Pick the CSV you saved from the Serial Monitor
[file, path] = uigetfile('*.csv', 'Select a capture CSV');
if isequal(file, 0); disp('No file selected.'); return; end
fn = fullfile(path, file);

%% 2. Read robustly — keep only the 6-number data rows
%    (skips the ---, # , === and header lines the board prints around the data)
raw  = readlines(fn);
rows = [];
for i = 1:numel(raw)
    parts = str2double(split(strtrim(raw(i)), ','));
    if numel(parts) == 6 && all(~isnan(parts))
        rows(end+1, :) = parts'; %#ok<AGROW>
    end
end
if isempty(rows); error('No data rows found in %s', file); end

t   = rows(:, 2);              % time (s)
acc = rows(:, 3:5) / 1000;    % milli-g -> g  (ax, ay, az)
ph  = rows(:, 6);             % phase id (0,1,2,...)
mag = sqrt(sum(acc.^2, 2));   % magnitude (g) — orientation-invariant

%% 3. >>> EDIT to match the protocol you ran (one label per phase, in order) <<<
phaseNames = ["Still", "Walk", "Freeze", "Walk"];              % Capture 1
% phaseNames = ["Still", "Foot-tap", "Turn/shuffle", "Free"];  % Capture 2

%% 4. Magnitude with shaded + labelled phases  (this is your Task 3 figure)
figure('Color', 'w'); hold on;
yl   = [min(mag) - 0.1, max(mag) + 0.1];
cmap = lines(max(ph) + 1);
for p = 0:max(ph)
    k = find(ph == p);
    if isempty(k); continue; end
    x0 = t(k(1)); x1 = t(k(end));
    patch([x0 x1 x1 x0], [yl(1) yl(1) yl(2) yl(2)], cmap(p+1, :), ...
          'FaceAlpha', 0.12, 'EdgeColor', 'none', 'HandleVisibility', 'off');
    nm = "Phase " + p;
    if p + 1 <= numel(phaseNames); nm = phaseNames(p + 1); end
    text((x0 + x1) / 2, yl(2) - 0.06 * (yl(2) - yl(1)), nm, ...
         'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'Interpreter', 'none');
end
plot(t, mag, 'k', 'LineWidth', 1.2);
xlabel('Time (s)'); ylabel('Acceleration magnitude (g)');
title('Capture — accel magnitude with protocol phases');
ylim(yl); grid on; box on; hold off;

%% 5. (optional) per-axis x / y / z
figure('Color', 'w');
plot(t, acc(:, 1), t, acc(:, 2), t, acc(:, 3), 'LineWidth', 1.0);
legend('a_x', 'a_y', 'a_z'); xlabel('Time (s)'); ylabel('Acceleration (g)');
title('Capture — x / y / z'); grid on;

fprintf('Loaded %d samples, %.1f s, %d phases from %s\n', ...
        size(rows, 1), t(end), max(ph) + 1, file);
