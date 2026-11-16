clear;close all;clc;
root=getenv('CNFI_RESULTS_DIR');assert(isfolder(root));
D=readtable(fullfile(root,'Traffic Dataset','ablation_summary.csv'));
metrics={'accuracy','precision','recall','f1','auc'};
labels={'Accuracy','Precision','Recall','F1-score','AUC'};
short={'PGI','+ Multi','+ Dynamic','+ Eigen','CNFI','SDMI','NMI'};
assert(height(D)==7);
fig=figure('Visible','off','Color','w','Units','inches','Position',[1 1 12 7]);
for i=1:5
 subplot(2,3,i);errorbar(1:7,D.(metrics{i}),D.([metrics{i} '_sd']),'o-','Color',[.13 .47 .71],'LineWidth',1.2,'MarkerFaceColor',[.49 .74 .88]);
 set(gca,'XTick',1:7,'XTickLabel',short,'XTickLabelRotation',35,'FontName','Times New Roman','FontSize',10);grid on;xlim([.5 7.5]);ylabel(labels{i});title(sprintf('(%c)',96+i),'FontWeight','normal');
end
subplot(2,3,6);axis off;
text(0,.85,{'Component settings', ...
 'PGI: 3-WLR on the shared feature graph', ...
 'Multi: prior midpoint applied to raw metrics', ...
 'Dynamic: confidence times (0.10 + 0.30 lambda)', ...
 'Eigen: raw Q and L2 iteration from the PDF', ...
 'Full CNFI: complete PDF formula', ...
 'See protocol.json for complete definitions.'}, ...
 'FontName','Times New Roman','FontSize',10,'VerticalAlignment','top','Interpreter','none');
annotation(fig,'textbox',[.25 .96 .5 .035],'String','Traffic dataset, top-4 features; mean +/- sample SD (5 folds)','HorizontalAlignment','center','EdgeColor','none','FontName','Times New Roman');
set(fig,'PaperUnits','inches','PaperSize',[12 7],'PaperPosition',[0 0 12 7]);
print(fig,fullfile(root,'Table5_ablation'),'-dpng','-r300');print(fig,fullfile(root,'Table5_ablation'),'-dpdf','-painters');
