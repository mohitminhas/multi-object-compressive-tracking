clc;clear all;close all;
%----------------------------------
rand('state',0);
%----------------------------------
vid = aviread('.\test video\shaking.avi');          % Read video file
[dontneedthis,num]  = size(vid);
Frames={vid.cdata};              
img = Frames{1};
[aa,bb,cc,dd] = select(img);
%a,b,c,d
initstate = [aa,bb,cc,dd];
[aa,bb,cc,dd] = select(img);
initstate1 = [aa,bb,cc,dd];
img = double(img(:,:,1));
%----------------------------------------------------------------
trparams.init_negnumtrain = 50;%number of trained negative samples
trparams.init_postrainrad = 4.0;%radical scope of positive samples
trparams.initstate = initstate;% object position [x y width height]
trparams.srchwinsz = 20;% size of search window
% Sometimes, it affects the results.
trparams1.init_negnumtrain = 50;%number of trained negative samples
trparams1.init_postrainrad = 4.0;%radical scope of positive samples
trparams1.initstate = initstate1;% object position [x y width height]
trparams1.srchwinsz = 20;% size of search window
%-------------------------
% classifier parameters
clfparams.width = trparams.initstate(3);
clfparams.height= trparams.initstate(4);
clfparams1.width = trparams1.initstate(3);
clfparams1.height= trparams1.initstate(4);
%-------------------------
% feature parameters
% number of rectangle from 2 to 4.
ftrparams.minNumRect = 2;
ftrparams.maxNumRect = 4;
ftrparams1.minNumRect = 2;
ftrparams1.maxNumRect = 4;
%-------------------------
M = 50;% number of all weaker classifiers, i.e,feature pool
%-------------------------
posx.mu = zeros(M,1);% mean of positive features
negx.mu = zeros(M,1);
posx.sig= ones(M,1);% variance of positive features
negx.sig= ones(M,1);

posx1.mu = zeros(M,1);% mean of positive features
negx1.mu = zeros(M,1);
posx1.sig= ones(M,1);% variance of positive features
negx1.sig= ones(M,1);
%-------------------------Learning rate parameter
lRate = 0.85;
%-------------------------
%compute feature template
[ftr.px,ftr.py,ftr.pw,ftr.ph,ftr.pwt] = HaarFtr(clfparams,ftrparams,M);
[ftr1.px,ftr1.py,ftr1.pw,ftr1.ph,ftr1.pwt] = HaarFtr(clfparams1,ftrparams1,M);
%-------------------------
%compute sample templates
posx.sampleImage = sampleImg(img,initstate,trparams.init_postrainrad,0,100000);
negx.sampleImage = sampleImg(img,initstate,1.5*trparams.srchwinsz,4+trparams.init_postrainrad,50);
posx1.sampleImage = sampleImg(img,initstate1,trparams1.init_postrainrad,0,100000);
negx1.sampleImage = sampleImg(img,initstate1,1.5*trparams1.srchwinsz,4+trparams1.init_postrainrad,50);
%-----------------------------------
%--------Feature extraction
iH = integral(img);%Compute integral image
posx.feature = getFtrVal(iH,posx.sampleImage,ftr);
negx.feature = getFtrVal(iH,negx.sampleImage,ftr);
posx1.feature = getFtrVal(iH,posx1.sampleImage,ftr1);
negx1.feature = getFtrVal(iH,negx1.sampleImage,ftr1);
%--------------------------------------------------
[posx.mu,posx.sig,negx.mu,negx.sig] = classiferUpdate(posx,negx,posx.mu,posx.sig,negx.mu,negx.sig,lRate);% update distribution parameters
[posx1.mu,posx1.sig,negx1.mu,negx1.sig] = classiferUpdate(posx1,negx1,posx1.mu,posx1.sig,negx1.mu,negx1.sig,lRate);% update distribution parameters
%--------------------------------------------------------
x = initstate(1);% x axis at the Top left corner
y = initstate(2);
w = initstate(3);% width of the rectangle
h = initstate(4);% height of the rectangle
x1 = initstate1(1);% x axis at the Top left corner
y1 = initstate1(2);
w1 = initstate1(3);% width of the rectangle
h1 = initstate1(4);% height of the rectangle
%--------------------------------------------------------
for i = 2:num
    img = Frames{i};
    imgSr = img;% imgSr is used for showing tracking results.
    img = double(img(:,:,1));
    detectx.sampleImage = sampleImg(img,initstate,trparams.srchwinsz,0,100000);   
    detectx1.sampleImage = sampleImg(img,initstate1,trparams1.srchwinsz,0,100000);   
    iH = integral(img);%Compute integral image
    detectx.feature = getFtrVal(iH,detectx.sampleImage,ftr);
    detectx1.feature = getFtrVal(iH,detectx1.sampleImage,ftr1);
    %------------------------------------
    r = ratioClassifier(posx,negx,detectx);% compute the classifier for all samples
    clf = sum(r);% linearly combine the ratio classifiers in r to the final classifier
    r1 = ratioClassifier(posx1,negx1,detectx1);% compute the classifier for all samples
    clf1 = sum(r1);% linearly combine the ratio classifiers in r to the final classifier
    %-------------------------------------
    [c,index] = max(clf);
    [c1,index1] = max(clf1);
    %--------------------------------
    x = detectx.sampleImage.sx(index);
    y = detectx.sampleImage.sy(index);
    w = detectx.sampleImage.sw(index);
    h = detectx.sampleImage.sh(index);
    initstate = [x y w h];
    x1 = detectx1.sampleImage.sx(index1);
    y1 = detectx1.sampleImage.sy(index1);
    w1 = detectx1.sampleImage.sw(index1);
    h1 = detectx1.sampleImage.sh(index1);
    initstate1 = [x1 y1 w1 h1];
    %-------------------------------Show the tracking results
    imshow(uint8(imgSr));
    rectangle('Position',initstate,'LineWidth',4,'EdgeColor','r');
    rectangle('Position',initstate1,'LineWidth',4,'EdgeColor','r');
    hold on;
    text(5, 18, strcat('#',num2str(i)), 'Color','y', 'FontWeight','bold', 'FontSize',20);
    set(gca,'position',[0 0 1 1]); 
    pause(0.00001); 
    hold off;
    %------------------------------Extract samples 
    posx.sampleImage = sampleImg(img,initstate,trparams.init_postrainrad,0,100000);
    negx.sampleImage = sampleImg(img,initstate,1.5*trparams.srchwinsz,4+trparams.init_postrainrad,trparams.init_negnumtrain);
    %--------------------------------------------------Update all the features 
    posx.feature = getFtrVal(iH,posx.sampleImage,ftr);
    negx.feature = getFtrVal(iH,negx.sampleImage,ftr);
    %--------------------------------------------------
    [posx.mu,posx.sig,negx.mu,negx.sig] = classiferUpdate(posx,negx,posx.mu,posx.sig,negx.mu,negx.sig,lRate);% update distribution parameters
        
    posx1.sampleImage = sampleImg(img,initstate1,trparams1.init_postrainrad,0,100000);
    negx1.sampleImage = sampleImg(img,initstate1,1.5*trparams1.srchwinsz,4+trparams1.init_postrainrad,trparams1.init_negnumtrain);
    %--------------------------------------------------Update all the features 
    posx1.feature = getFtrVal(iH,posx1.sampleImage,ftr1);
    negx1.feature = getFtrVal(iH,negx1.sampleImage,ftr1);
    %--------------------------------------------------
    [posx1.mu,posx1.sig,negx1.mu,negx1.sig] = classiferUpdate(posx1,negx1,posx1.mu,posx1.sig,negx1.mu,negx1.sig,lRate);% update distribution parameters
end