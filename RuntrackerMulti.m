clc;clear all;close all;
%----------------------------------
rand('state',0);
%----------------------------------
vid = aviread('.\test video\shaking.avi');          % Read video file
[dontneedthis,num]  = size(vid);
Frames={vid.cdata};              
img = Frames{1};
nob = input('Enter number of objects to be tracked and then select them manually: '); %number of objects
initstate = zeros(nob,4);
for i=1:nob
    [aa,bb,cc,dd] = select(img);
    initstate(i,:) = [aa,bb,cc,dd];
end %initstate
img = double(img(:,:,1));
%----------------------------------------------------------------
M = 50;% number of all weaker classifiers, i.e,feature pool
%-------------------------
lRate = 0.85;
%-------------------------Learning rate parameter
trparams = struct;
clfparams = struct;
ftrparams = struct;
posx = struct;
negx = struct;
ftr = struct;
x = zeros(1,nob);
y = zeros(1,nob);
w = zeros(1,nob);
h = zeros(1,nob);
for i=1:nob
    trparams(i).init_negnumtrain = 50;%number of trained negative samples
    trparams(i).init_postrainrad = 4.0;%radical scope of positive samples
    trparams(i).initstate = initstate;% object position [x y width height]
    trparams(i).srchwinsz = 20;% size of search window
    % Sometimes, it affects the results.
    %-------------------------
    % classifier parameters
    clfparams(i).width = trparams(i).initstate(3);
    clfparams(i).height= trparams(i).initstate(4);
    %-------------------------
    % feature parameters
    % number of rectangle from 2 to 4.
    ftrparams(i).minNumRect = 2;
    ftrparams(i).maxNumRect = 4;
    %-------------------------
    posx(i).mu = zeros(M,1);% mean of positive features
    negx(i).mu = zeros(M,1);
    posx(i).sig= ones(M,1);% variance of positive features
    negx(i).sig= ones(M,1);
    %-------------------------
    %compute feature template
    [ftr(i).px,ftr(i).py,ftr(i).pw,ftr(i).ph,ftr(i).pwt] = HaarFtr(clfparams(i),ftrparams(i),M);
    %-------------------------
    %compute sample templates
    posx(i).sampleImage = sampleImg(img,initstate(i,:),trparams(i).init_postrainrad,0,100000);
    negx(i).sampleImage = sampleImg(img,initstate(i,:),1.5*trparams(i).srchwinsz,4+trparams(i).init_postrainrad,50);
    %--------Feature extraction
end
%-----------------------------------
iH = integral(img);%Compute integral image
for i = 1:nob
    posx(i).feature = getFtrVal(iH,posx(i).sampleImage,ftr(i));
    negx(i).feature = getFtrVal(iH,negx(i).sampleImage,ftr(i));
    %--------------------------------------------------
    [posx(i).mu,posx(i).sig,negx(i).mu,negx(i).sig] = classiferUpdate(posx(i),negx(i),posx(i).mu,posx(i).sig,negx(i).mu,negx(i).sig,lRate);% update distribution parameters
    %-------------------------------------------------
    x(i) = initstate(i,1);% x axis at the Top left corner
    y(i) = initstate(i,2);
    w(i) = initstate(i,3);% width of the rectangle
    h(i) = initstate(i,4);% height of the rectangle
end
detectx = struct;
for i = 2:num
    img = Frames{i};
    imgSr = img;% imgSr is used for showing tracking results.
    img = double(img(:,:,1));
    for j = 1:nob
        detectx(j).sampleImage = sampleImg(img,initstate(j,:),trparams(j).srchwinsz,0,100000);   
    end
    iH = integral(img);%Compute integral image
    for j = 1:nob
        detectx(j).feature = getFtrVal(iH,detectx(j).sampleImage,ftr(j));
        %------------------------------------
        clear r
        r = ratioClassifier(posx(j),negx(j),detectx(j));% compute the classifier for all samples
        clear clf
        clf = sum(r);% linearly combine the ratio classifiers in r to the final classifier
        %-------------------------------------
        [c(j),index(j)] = max(clf);
        %--------------------------------
        x(j) = detectx(j).sampleImage.sx(index(j));
        y(j) = detectx(j).sampleImage.sy(index(j));
        w(j) = detectx(j).sampleImage.sw(index(j));
        h(j) = detectx(j).sampleImage.sh(index(j));
        initstate(j,:) = [x(j),y(j),w(j),h(j)];
    end
    %-------------------------------Show the tracking results
    imshow(uint8(imgSr));
    for j = 1:nob
        rectangle('Position',initstate(j,:),'LineWidth',4,'EdgeColor','r');
    end
    hold on;
    text(5, 18, strcat('#',num2str(i)), 'Color','y', 'FontWeight','bold', 'FontSize',20);
    set(gca,'position',[0 0 1 1]);
    pause(0.00001); 
    hold off;
    %------------------------------Extract samples
    for j = 1:nob
        posx(j).sampleImage = sampleImg(img,initstate(j,:),trparams(j).init_postrainrad,0,100000);
        negx(j).sampleImage = sampleImg(img,initstate(j,:),1.5*trparams(j).srchwinsz,4+trparams(j).init_postrainrad,trparams(j).init_negnumtrain);
        %--------------------------------------------------Update all the features 
        posx(j).feature = getFtrVal(iH,posx(j).sampleImage,ftr(j));
        negx(j).feature = getFtrVal(iH,negx(j).sampleImage,ftr(j));
        %--------------------------------------------------
        [posx(j).mu,posx(j).sig,negx(j).mu,negx(j).sig] = classiferUpdate(posx(j),negx(j),posx(j).mu,posx(j).sig,negx(j).mu,negx(j).sig,lRate);% update distribution parameters
    end
end