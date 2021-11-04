clc;close all;
data = load('../AB06/10_09_18/stair/markers/stair_1_l_01_01.mat').data;
data = table2array(data);
time = data(:,1);
markers=zeros(length(time),3,28);

for i = 0:27
    markers(:,:,i+1) = data(:,(2+i*3):(4+i*3));
end
figure;
plot_box();
hold on
i = 1;
position = reshape(markers(i,:,:),3,28)';
    h1 = scatter3(position(:,3),position(:,1),position(:,2),'blue');
    h2 = line(position(5:6,3),position(5:6,1),position(5:6,2),'LineWidth',2);
    h3 = line(position([9,12],3),position([9,12],1),position([9,12],2),'LineWidth',2);
    h4 =line(position([6,21],3),position([6,21],1),position([6,21],2),'LineWidth',2);
    h5 =line(position([12,25],3),position([12,25],1),position([12,25],2),'LineWidth',2);
    h6 =line(position(21:22,3),position(21:22,1),position(21:22,2),'LineWidth',2);
    h7 =line(position(25:26,3),position(25:26,1),position(25:26,2),'LineWidth',2);
    
for i = 2000:10:2800%1:5:length(time)%200:10:width
    
  position = reshape(markers(i,:,:),3,28)';
    delet_skeleton(h1,h2,h3,h4,h5,h6,h7);
    [h1,h2,h3,h4,h5,h6,h7]=plot_skeleton(position);
    
    xlabel("x");
    
    axis equal;
    axis([-4000,1000,-5500,-5000,0,2000]);
    pause(0);
    drawnow;
    
end

function [h1,h2,h3,h4,h5,h6,h7]=plot_skeleton(position)
h1 = scatter3(position(:,3),position(:,1),position(:,2),'blue');
h2 = line(position(5:6,3),position(5:6,1),position(5:6,2),'LineWidth',2);
h3 = line(position([9,12],3),position([9,12],1),position([9,12],2),'LineWidth',2);
h4 =line(position([6,21],3),position([6,21],1),position([6,21],2),'LineWidth',2);
h5 =line(position([12,25],3),position([12,25],1),position([12,25],2),'LineWidth',2);
h6 =line(position(21:22,3),position(21:22,1),position(21:22,2),'LineWidth',2);
h7 =line(position(25:26,3),position(25:26,1),position(25:26,2),'LineWidth',2);
end

function delet_skeleton(h1,h2,h3,h4,h5,h6,h7)
delete(h1);delete(h2);delete(h3);delete(h4);delete(h5);delete(h6);delete(h7);
end

function plot_box()
x=linspace(0,1,100);

y=linspace(0,1,100);

[X,Y]=meshgrid(x,y);
for i = 0:5
    hight = 101.6;
    base_h = 45;
    width = 320;
    h = fill3([-2200+i*width -2200+i*width -2200+i*width -2200+i*width],[-5500 -5500 -5000 -5000],[i*hight+base_h (i+1)*hight+base_h (i+1)*hight+base_h i*hight+base_h],'k', ...
    [-2200+i*width -2200+(i+1)*width -2200+(i+1)*width -2200+i*width],[-5500 -5500 -5000 -5000],[(i+1)*hight+base_h (i+1)*hight+base_h (i+1)*hight+base_h (i+1)*hight+base_h],'k');
set(h,'FaceAlpha',0.2) ;
hold on;
end
h = fill3([-2200+i*width 500 500 -2200+i*width],[-5500 -5500 -5000 -5000],[(i+1)*hight+base_h (i+1)*hight+base_h (i+1)*hight+base_h (i+1)*hight+base_h],'k');

set(h,'FaceAlpha',0.2) ;
    

end

