% This script generates compressed observations of a dynamic scene which is
% sparse in the canonical basis and uses DMD and DFS to reconstruct the
% scene (See arXiv 1307.5944, section 7.2)
% Written by Eric C. Hall, 15 Sept 2014

clear
clc

load Paramecium_image

%define real image size
[rows cols]=size(paramecium);
d=rows*cols;

%define observations dimension
k=50; %Number of observations per time
T=1000; %Number of time steps
sigma=.1; % Variance of observation noise

tau=2e-3*d;
l=@(x,theta,A) 1/(2*sigma*d)*(x-A*theta)'*(x-A*theta)+tau*sum(abs(theta))/d; %per pixel loss

alpha=[-1 0; 0 1]; %True dynamics
alpha_hat=[0 0; 0 1; -1 1; -1 0; -1 -1; 0 -1; 1 -1; 1 0; 1 1]; %Potential dynamics
N=size(alpha_hat,1);
m=1;
lambda=m/(T-1);

no_iters=3; %Number of trials
loss_DMD=zeros(no_iters,T,N);
loss_DFS=zeros(no_iters,T);

for iter=1:no_iters
    disp(['Trial number ',int2str(iter) ' out of ' int2str(no_iters)])
    theta_hat_DMD=zeros(d,T+1,N);
    theta_hat_DFS=zeros(d,T+1);
    theta_true=zeros(d,T+1);
    weights=ones(N,1)/N;
    
    for t=1:T;
        if mod(t,100)==0
            disp(['t=' int2str(t) ' out of ' int2str(T)])
        end
        
        if t==1
            theta_true(:,t)=paramecium(:);
        elseif t<550
            theta_true(:,t)=reshape(circshift(reshape(theta_true(:,t-1),rows,cols),alpha(1,:)),d,1);
        elseif t==550
            theta_true(:,t)=reshape(imrotate(reshape(theta_true(:,t-1),rows,cols),-90),d,1);
        else
            theta_true(:,t)=reshape(circshift(reshape(theta_true(:,t-1),rows,cols),alpha(2,:)),d,1);
        end
        
        A=randn(k,d); %Observation matrix
        eta=1/(sqrt(t)); %DMD Step size
        eta_p=sqrt((8*(2*log(N)+log(T)+1))/T); %DFS step size
        x_t=A*theta_true(:,t)+randn(k,1)*sqrt(sigma); %Data
        
        for n=1:N
            theta_t=theta_hat_DMD(:,t,n);
            loss_DMD(iter,t,n)=l(x_t,theta_t,A); %incur loss
            theta_t=theta_t-eta*A'*(A*theta_t-x_t)/(sigma*d); %update estimate
            theta_t=Soft_thresh(theta_t,eta*tau/d); %soft threshold
            theta_t=max(min(theta_t,1),0); %Ensure estimate is between 0 and 1
            theta_hat_DMD(:,t+1,n)=reshape(circshift(reshape(theta_t,rows,cols),alpha_hat(n,:)),d,1);
            weights(n)=weights(n)*exp(-eta_p*loss_DMD(iter,t,n)); %Update DFS weights
        end
        
        loss_DFS(iter,t)=l(x_t,theta_hat_DFS(:,t),A);
        weights=weights./(sum(weights));
        weights=lambda./N+(1-lambda)*weights;
        
        theta_hat_DFS(:,t+1)=squeeze(theta_hat_DMD(:,t+1,:))*weights;
    end
end

figure(1)
subplot(1,2,1), plot(1:T,mean(loss_DFS,1),1:T,squeeze(mean(loss_DMD(:,:,[1 4 2]),1))')
title(['Mean of instantaneous loss over ' int2str(no_iters) ' trials'])
legend('DFS','COMID','DMD Dynamic 1','DMD Dynamic 2')
subplot(1,2,2), plot(1:T,loss_DFS(no_iters,:),1:T,squeeze(loss_DMD(no_iters,:,[1 4 2]))')
title('Instantaneous loss of a single trial')
legend('DFS','COMID','DMD Dynamic 1','DMD Dynamic 2')

vid_bool=lower(input('Create and store result video (about 1-1.5 GB) (y/n)? ','s'));
if strcmp(vid_bool,'y') || strcmp(vid_bool,'yes')
    disp('Creating video')
    h=figure(2);
    set(gcf,'position',[1 840 1020 840])
    aviobj=avifile('CompressedSensingResults.avi');
    for t=1:T
        subplot(2,2,1), imagesc(reshape(theta_true(:,t),rows,cols),[0 .5]);
        axis off; colormap bone
        title('Ground Truth')
        subplot(2,2,2), imagesc(reshape(theta_hat_DMD(:,t,1),rows,cols),[0 .5])
        axis off;
        title('COMID Prediction')
        subplot(2,2,3), imagesc(reshape(theta_hat_DFS(:,t),rows,cols),[0 .5])
        axis off; 
        title('DFS Prediction')
        subplot(224), plot(1:T,loss_DFS(no_iters,:),1:T,squeeze(loss_DMD(no_iters,:,[1 4 2]))',...
            t,loss_DFS(no_iters,t),'ko',repmat(t,1,3),squeeze(loss_DMD(no_iters,t,[1 4 2])),'ko')
        title('Instantaneous loss of a single trial')
        legend('DFS','COMID','DMD Dynamic 1','DMD Dynamic 2')
        aviobj=addframe(aviobj,h);
    end
    aviobj=close(aviobj);
end