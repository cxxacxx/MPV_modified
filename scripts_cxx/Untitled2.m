fig = figure;
ax = axes(fig);
tp = theaterPlot('XLim',[0 90],'YLim',[-35 35],'ZLim',[0 50],'Parent',ax);
radarPlotter = detectionPlotter(tp,'DisplayName','Radar Detections');
plotDetection(radarPlotter, [30 -5 5; 50 -10 10; 40 7 40])
clearPlotterData(radarPlotter);
grid on
view(90,0)