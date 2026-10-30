
clear; clc; close all;
% Resolve data relative to this script, independent of MATLAB's current folder.
script_dir = fileparts(mfilename('fullpath'));
cd(script_dir);


FIG_W = 7.94;   % 英寸（A4 竖版宽）
FIG_H = 9.25;   % 英寸（A4 竖版高）

c_lo  = [0.93 0.96 0.99];
c_m1  = [0.49 0.74 0.88];
c_m2  = [0.13 0.47 0.71];
c_hi  = [0.03 0.14 0.27];
N_CM  = 512;
t1 = linspace(0,1,N_CM/2)';
t2 = linspace(0,1,N_CM/2)';
CMAP = [ c_lo + t1.*(c_m1-c_lo);
         c_m1 + t2.*(c_hi-c_m1) ];
CMAP = max(0, min(1, CMAP));

ACCENT_COLOR = [0.85 0.33 0.10];

if ismac
    FONT = 'Times New Roman';
else
    FONT = 'Times New Roman';
end

fig = figure('Units','inches','Position',[0.5 0.5 FIG_W FIG_H],'Color','w');
colormap(fig, CMAP);


ds_names = {'Financial Dataset', 'Medical Dateset', 'Traffic Dataset'};

panel_labels_map = containers.Map( ...
    {'Financial Dataset','Traffic Dataset','Medical Dateset'}, ...
    {'(b)  Financial Dataset','(a)  Traffic Dataset','(c)  Medical Dataset'});


% Symmetric triangular layout with a separate full-width shared legend.
title_h = 0.030;
top_bot = 0.550; top_h = 0.380; top_w = 0.74; top_left = 0.13;
bot_bot = 0.185; bot_h = 0.300;
bot_w_fin = 0.44; bot_w_med = 0.44;
bot_left1 = 0.03; bot_left2 = 0.53;
ax_pos = {[bot_left1 bot_bot bot_w_fin bot_h], ...
          [bot_left2 bot_bot bot_w_med bot_h], ...
          [top_left top_bot top_w top_h]};
audit = struct([]);
for ds_idx = 1:3
    ds = ds_names{ds_idx};
    is_large = (ds_idx == 3);   % Traffic 为大图（上排单图）

    candidates_adj  = {fullfile(ds,'adj.csv')};
    candidates_nmi  = {fullfile(ds,'nmi.csv')};
    candidates_rank = {fullfile(ds,'rank.csv')};

    file_adj=''; file_nmi=''; file_rank='';
    for k=1:numel(candidates_adj); if isfile(candidates_adj{k}),  file_adj  = candidates_adj{k};  break; end; end
    for k=1:numel(candidates_nmi); if isfile(candidates_nmi{k}),  file_nmi  = candidates_nmi{k};  break; end; end
    for k=1:numel(candidates_rank); if isfile(candidates_rank{k}), file_rank = candidates_rank{k}; break; end; end

    if isempty(file_adj) || isempty(file_nmi) || isempty(file_rank)
        error('Missing real CSV data for %s. Export stopped.', ds);
    else
        adj_table  = readtable(file_adj);
        nmi_table  = readtable(file_nmi);
        rank_table = readtable(file_rank);
        source_features = string(adj_table{:,1});
        features = canonical_feature_names(ds, source_features);
        nmi_keys = canonical_feature_names(ds, string(nmi_table.Feature));
        score_keys = canonical_feature_names(ds, string(rank_table.Feature));
        fid=fopen(file_adj,'r','n','UTF-8');
        header_names=string(strsplit(fgetl(fid),',')); fclose(fid);
        header_keys=canonical_feature_names(ds,header_names(2:end)');
        if ~isequal(features,header_keys), fprintf('Mismatch dataset: %s\n', ds); disp(table(features,header_keys)); error('Adjacency row/column names differ.'); end
        assert(numel(unique(features))==numel(features),'Duplicate feature keys.');
        assert(numel(unique(nmi_keys))==numel(nmi_keys),'Duplicate NMI keys.');
        assert(numel(unique(score_keys))==numel(score_keys),'Duplicate score keys.');
        A_raw      = table2array(adj_table(:,2:end));
        A_raw(isnan(A_raw)) = 0;
        num_nodes  = numel(features);
        [found_nmi,idx_nmi]=ismember(features,nmi_keys);
        [found_score,idx_score]=ismember(features,score_keys);
        assert(all(found_nmi),'A feature is missing from nmi.csv; export stopped.');
        assert(all(found_score),'A feature is missing from rank.csv; export stopped.');
        node_nmi=nmi_table.NMI(idx_nmi);
        node_score=rank_table.Score(idx_score);
    end

    pos_vals = A_raw(A_raw > 0);
    if numel(pos_vals) > 10
        pct    = max(65, 100 - 700/num_nodes);
        thresh = prctile(pos_vals, pct);
    else
        thresh = 0;
    end
    A = A_raw;
    A(A < thresh) = 0;
    A = A - diag(diag(A));

    [node_nmi, sidx] = sort(node_nmi, 'descend');
    features   = features(sidx);
    node_score = node_score(sidx);
    A          = A(sidx, sidx);

    nmi_n  = (node_nmi  - min(node_nmi))  / (max(node_nmi)  - min(node_nmi)  + eps);
    score_n= (node_score- min(node_score))/ (max(node_score)- min(node_score)+ eps);

    r_node_min = 0.028;
    r_node_max = 0.065;
    node_r = r_node_min + score_n * (r_node_max - r_node_min);

    R_node = 1.00;
    theta  = pi/2 - (2*pi)*(0:num_nodes-1)'/num_nodes;
    Xn = R_node*cos(theta);
    Yn = R_node*sin(theta);

    if is_large
        if num_nodes <= 20
            FS_lbl = 9.0;
            R_lbl  = 1.23;
        elseif num_nodes <= 30
            FS_lbl = 7.5;
            R_lbl  = 1.26;
        else
            FS_lbl = 6.5;
            R_lbl  = 1.29;
        end
    else
        if num_nodes <= 18
            FS_lbl = 7.5;
            R_lbl  = 1.26;
        elseif num_nodes <= 22
            FS_lbl = 7.0;
            R_lbl  = 1.28;
        else
            FS_lbl = 6.5;
            R_lbl  = 1.30;
        end
    end

    ax = axes('Position', ax_pos{ds_idx});
    hold on; axis equal off;
    VIEW_R = R_lbl + 0.20;
    xlim([-VIEW_R, VIEW_R]);
    ylim([-VIEW_R, VIEW_R]);

    max_w   = max(A(:)) + eps;
    N_curve = 100;
    t_vec   = linspace(0,1,N_curve)';

    [ri, ci, wv] = find(triu(A,1));
    if ~isempty(wv)
        [~, ord] = sort(wv);
        ri = ri(ord); ci = ci(ord); wv = wv(ord);
        for ek = 1:numel(wv)
            r = ri(ek); c_ = ci(ek); w = wv(ek);
            P0 = [Xn(r), Yn(r)];
            P2 = [Xn(c_), Yn(c_)];
            chord_len = norm(P0-P2);
            pull = 0.30 + 0.52*(chord_len/2.0);
            pull = min(pull, 0.85);
            P1 = ((P0+P2)/2) * (1 - pull);
            curve = (1-t_vec).^2.*P0 + 2*(1-t_vec).*t_vec.*P1 + t_vec.^2.*P2;
            nmi_avg = (nmi_n(r) + nmi_n(c_)) / 2;
            c_idx   = max(1, min(N_CM, round(1 + (N_CM-1)*nmi_avg)));
            rgb     = CMAP(c_idx, :);
            wt_n    = w / max_w;
            lw      = 0.3 + 1.4*wt_n;
            alpha_v = 0.15 + 0.30*wt_n;
            line(curve(:,1), curve(:,2), ...
                'Color', [rgb, alpha_v], 'LineWidth', lw, ...
                'LineStyle', '-', 'Clipping', 'off');
        end
    end

    for j = 1:num_nodes
        c_idx = max(1, min(N_CM, round(1 + (N_CM-1)*nmi_n(j))));
        fc    = CMAP(c_idx, :);
        if score_n(j) > 0.80
            ec = ACCENT_COLOR; lw_e = 0.9;
        else
            ec = [0.20 0.20 0.22]; lw_e = 0.5;
        end
        r_pix = node_r(j);
        rectangle('Position', [Xn(j)-r_pix, Yn(j)-r_pix, 2*r_pix, 2*r_pix], ...
            'Curvature', [1,1], 'FaceColor', fc, 'EdgeColor', ec, 'LineWidth', lw_e);
    end

    for j = 1:num_nodes
        ang     = theta(j);
        ang_deg = rad2deg(ang);

        x_tick0 = (R_node + node_r(j) + 0.025)*cos(ang);
        y_tick0 = (R_node + node_r(j) + 0.025)*sin(ang);
        x_tick1 = (R_lbl - 0.04)*cos(ang);
        y_tick1 = (R_lbl - 0.04)*sin(ang);
        line([x_tick0, x_tick1], [y_tick0, y_tick1], ...
            'Color', [0.65 0.65 0.65], 'LineWidth', 0.4, 'Clipping','off');

        xL = R_lbl*cos(ang);
        yL = R_lbl*sin(ang);

        if ang_deg >= -90 && ang_deg <= 90
            rot_ang = ang_deg;
            hal = 'left';
        else
            rot_ang = ang_deg + 180;
            hal = 'right';
        end

        lbl = char(features(j));
        label_interpreter='none';
        if strcmp(lbl,'Actual_Number_of_Lanes'),lbl='Actual_No_of_Lanes';end
        if strcmp(lbl,'38_PD≥4MV')
            % Use MATLAB's TeX >= glyph to avoid missing Unicode on R2019a.
            lbl='38\_PD\geq4MV';label_interpreter='tex';
        end

        if nmi_n(j) > 0.75
            fw = 'bold';
        else
            fw = 'normal';
        end

        text(xL, yL, {lbl}, ...
            'Interpreter',label_interpreter, ...
            'Rotation', rot_ang, ...
            'HorizontalAlignment', hal, ...
            'VerticalAlignment', 'middle', ...
            'FontName', FONT, 'FontSize', FS_lbl, ...
            'FontWeight', fw, ...
            'Color', [0.15 0.15 0.15], ...
            'Clipping', 'off');
    end

    th_ring = linspace(0, 2*pi, 360);
    line(R_node*cos(th_ring), R_node*sin(th_ring), ...
        'Color', [0.80 0.80 0.82], 'LineWidth', 0.4, 'LineStyle', ':');

    lbl_str = panel_labels_map(ds);

    if is_large
        fig_x_ctr = top_left + top_w * 0.5;
        fig_y_ttl = 0.475;
    else
        fig_y_ttl = 0.130;
        if ds_idx == 1
            fig_x_ctr = bot_left1 + bot_w_fin * 0.5;
        else
            fig_x_ctr = bot_left2 + bot_w_med * 0.5;
        end
    end

    annotation('textbox', [fig_x_ctr - 0.14, fig_y_ttl, 0.28, 0.035], ...
        'String', lbl_str, ...
        'FontName', FONT, 'FontSize', 11, 'FontWeight', 'bold', ...
        'Color', [0.10 0.10 0.10], ...
        'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
        'EdgeColor', 'none', 'BackgroundColor', 'none', 'FitBoxToText', 'off');

    audit(ds_idx).dataset = ds;
    audit(ds_idx).features = cellstr(features);
    audit(ds_idx).nmi = node_nmi;
    audit(ds_idx).score = node_score;
end  % 子图循环结束


% Shared encoding band: it belongs to the figure, never to one dataset.
annotation(fig,'line',[0.065 0.935],[0.112 0.112], ...
    'Color',[0.75 0.79 0.83],'LineWidth',0.6);
annotation(fig,'textbox',[0.15 0.083 0.70 0.025], ...
    'String','Shared legend for all three networks','FontName',FONT, ...
    'FontSize',9,'FontWeight','bold','HorizontalAlignment','center', ...
    'EdgeColor','none','Margin',0);
ax_key = axes('Position',[0.065 0.006 0.87 0.076]);
hold(ax_key,'on');axis(ax_key,'off');
xlim(ax_key,[0 FIG_W*0.87]);ylim(ax_key,[0 FIG_H*0.076]);
% Axes use physical-inch coordinates so node keys are true circles.
image(ax_key,[0.08 2.68],[0.22 0.28],repmat(reshape(CMAP,[1 size(CMAP,1) 3]),[2 1 1]));
set(ax_key,'YDir','normal');
text(ax_key,1.38,0.49,'Within-dataset NMI (min-max)', ...
    'FontName',FONT,'FontSize',8,'HorizontalAlignment','center');
for kk=0:2
 text(ax_key,0.08+kk*1.30,0.12,sprintf('%.1f',kk/2), ...
     'FontName',FONT,'FontSize',7,'HorizontalAlignment','center');
end
text(ax_key,4.10,0.49,'Node size: normalized score', ...
    'FontName',FONT,'FontSize',8,'HorizontalAlignment','center');
for ss=1:3
 rr=0.030+(ss-1)*0.014; xx=3.68+(ss-1)*0.42;
 rectangle(ax_key,'Position',[xx-rr 0.245-rr 2*rr 2*rr], ...
     'Curvature',[1 1],'FaceColor',[0.65 0.65 0.65], ...
     'EdgeColor',[0.35 0.35 0.35],'LineWidth',0.5);
 names={'Low','Mid','High'};
 text(ax_key,xx,0.10,names{ss},'FontName',FONT,'FontSize',7, ...
     'HorizontalAlignment','center');
end
rectangle(ax_key,'Position',[5.45 0.20 0.10 0.10], ...
    'Curvature',[1 1],'FaceColor',c_m1,'EdgeColor',ACCENT_COLOR, ...
    'LineWidth',1.0);
text(ax_key,5.68,0.245,{'Orange outline:','score > 0.80'}, ...
    'FontName',FONT,'FontSize',8,'VerticalAlignment','middle');
drawnow;
save('Figure6_feature_audit.mat','audit');
fid=fopen('Figure6_feature_audit.json','w','n','UTF-8');
fprintf(fid,'%s',jsonencode(audit));fclose(fid);
out_name = 'Fig6_SharedLegend';

set(fig, 'PaperUnits','inches','PaperSize',[FIG_W FIG_H],...
    'PaperPosition',[0 0 FIG_W FIG_H],'PaperPositionMode','manual');

print(fig, out_name, '-dpdf', '-painters', '-r0');
print(fig, out_name, '-dpng', '-r600');

fprintf('Saved Fig6_SharedLegend.pdf and .png using real CSV data.\n');

function names=canonical_feature_names(ds,names)
names=strtrim(string(names));
if strcmp(ds,'Financial Dataset')
    % Unique schema alias: adjacency uses Insolvency; UCI/NMI/rank use default.
    names(names=="Insolvency")="default";
elseif strcmp(ds,'Traffic Dataset')
    % Restore the official dwelling-vehicle category; numeric CSV values unchanged.
    % Dataset-specific names are handled by the CSV alias mapping.
end
end
