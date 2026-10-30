% Each folder must contain sensitivity_stats.csv from run_alpha_sensitivity.py.
% Shaded regions show mean +/- sample SD across validation folds, not CI.
clear; close all; clc;
root=getenv('CNFI_RESULTS_DIR');
if isempty(root) || ~isfolder(root),error('Set CNFI_RESULTS_DIR to your experiment result root.');end
cd(root);
folders={'Traffic Dataset','Financial Dataset','Medical Dateset'};
labels={'Traffic','Financial','Medical'}; metrics={'accuracy','f1','precision','recall','specificity','auc'};
names={'Accuracy','F1-Score','Precision','Recall','Specificity','AUC'};
colors={[.8 .2 .2],[0 .65 .75],[.2 .6 .3]}; marks={'o','s','^'};
D=cell(1,3);
for i=1:3
 file=fullfile(folders{i},'sensitivity_stats.csv');
 if ~isfile(file),error('Missing actual sensitivity data: %s',file);end
 D{i}=readtable(file); assert(all(ismember([metrics strcat(metrics,'_sd')],D{i}.Properties.VariableNames)), 'Missing metrics');
 assert(height(D{i})==11 && all(abs(sort(D{i}.alpha)-(0:.1:1)')<1e-8),'Expected exactly alpha 0:.1:1');
 D{i}=sortrows(D{i},'alpha');
end
fig=figure('Visible','off','Color','w','Units','inches','Position',[1 1 11 7]);
for j=1:6
 subplot(2,3,j);hold on;
 for i=1:3
  x=D{i}.alpha;y=D{i}.(metrics{j});sd=D{i}.([metrics{j} '_sd']);
  assert(all(isfinite(y)) && all(isfinite(sd)) && all(sd>=0),'Invalid mean or SD');
  fill([x;flipud(x)],[max(0,y-sd);flipud(min(1,y+sd))],colors{i},'FaceAlpha',.14,'EdgeColor','none','HandleVisibility','off');
  plot(x,y,'-','Color',colors{i},'Marker',marks{i},'LineWidth',1.2,'MarkerSize',5);
 end
 grid on;xlabel('Alpha');ylabel(names{j});xlim([0 1]);title(sprintf('(%c) %s',char(96+j),names{j}));
 if j==1,legend(labels,'Location','best');end
end
annotation(fig,'textbox',[.28 .955 .44 .03],'String','Shading: mean +/- 1 SD across folds','HorizontalAlignment','center','EdgeColor','none','FontSize',10);
set(fig,'PaperUnits','inches','PaperSize',[11 7],'PaperPosition',[0 0 11 7],'PaperPositionMode','manual');
print(fig,'Fig5_sensitivity','-dpng','-r300');print(fig,'Fig5_sensitivity','-dpdf','-painters');
