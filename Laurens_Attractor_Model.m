%% Simulate a HD attractor based on Stringer et al. 2002

% Constants 
N_HD_Neurons = 100 ; % Number of neurons in the attractor
HD_Neurons_PD = (1:N_HD_Neurons)*2*pi/N_HD_Neurons - pi ; % The prefered directions of the neurons, distributed regularly from 0 to 360°
N_Velocity_Neurons=101 ; % Numbers of neurons that encode velocity
Velocity_Neurons_Offset = [-50:50]*0.02; % See below



% These constants determine the strengths of the connections
K_Recurrent = 52/N_HD_Neurons ; 
K_Inhibitory = 3*K_Recurrent; % winh
K_Excitatory = 12*K_Recurrent ; % f1_Chdrot
K_Visual = 17 ; % Strength of the visual input
tau = 1; % Time constant of the neurons, in simulation steps
alpha = 30;beta = 0.022; % gives a very 'all or nothing' profile
Recurrent_std = 0.15 ; % Each neuron activates its neighbors, and the strength of these connections follow a Gaussian profile. This is the std of the Gaussian
Visual_std = 0.262 ; % A visual landmark at a certain position activates the corresponding HD neurons. The strength of this activation follow a Gaussian profile. This is the std of the Gaussian
% We are going to compute the matrix "w^ROT_ijk" in eq. 10 of Stringer et al. 2002.
% The principle is the following:
% It is a 3D matrix that encodes the connexion between HD Neuron i and
% HD Neuron j when the velocity is encoded by Velocity_Neuron k

% If the head is not moving, then this matrix corresponds to the classical
% attractor model where each neuron excites its neighbors.
% The weight of these excitatory connections follows a Gaussian profile,
% whose width is encoded by the parameter Recurrent_std
% If the head rotates, the 'hill' is shifted by an offset factor, which is
% encoded by the variable Velocity_Neurons_Offset
Wrot = zeros(N_HD_Neurons,N_HD_Neurons,N_Velocity_Neurons) ;

% The loop will iterate through all values of j and k. In each iteration,
% it will compute a weight vector that corresponds to all possible values
% of i

for j = 1:N_HD_Neurons
    for k = 1:N_Velocity_Neurons
        % delta_pd represents the difference of PD between all HD_Neurons
        % and a certain HD_Neuron j, and is shifted by an amound that corresponds to 
        % the Velocity Neuron k. 
        delta_pd = HD_Neurons_PD - repmat(HD_Neurons_PD(j)+Velocity_Neurons_Offset(k),1,N_HD_Neurons) ;
        delta_pd = mod(delta_pd+pi,2*pi)-pi;
        Wrot(:,j,k) = normpdf(delta_pd',0,Recurrent_std) ;
    end
end

Wrot = Wrot/mvnpdf(0,0,Recurrent_std) ; % Sets the peak value of Wrot to 1

% Visual input: this vector is the pattern of excitation from the visual
% system to the HD Neurons when the visual system reports an azimuth of 0.
% If follows a hill of activation whose with is encoded by the parameter
% Visual_std
% If the visual azimuth is different, we will circularly shift this vector
% accordingly

Iv0 = normpdf(HD_Neurons_PD,0,Visual_std)'; Iv0=Iv0/max(Iv0)*K_Visual ;


%%

for CONDITION = 1:3
    switch CONDITION 
        case 1, RECCURENT_GAIN = 1; VELOCITY_GAIN = 1;
        case 2, RECCURENT_GAIN = 0.5; VELOCITY_GAIN = 1;
        case 3, RECCURENT_GAIN = 1; VELOCITY_GAIN = 0.5;     
    end
    
    % Generate a random trajectory by drawing head velocity Head_Velocity
    % (according to a Gaussian distribution) and integrating it over time
    % to obtain azimuth (Az).            
    Vneurons_Calibration = [-50:50]*7.55 ;
    dt = 0.1 ;
    time = 0:dt:240;
    Head_Velocity = randn(size(time))*240 ;
    [b,a]=butter(2,2*dt/2,'low');
    Vmax = max(Vneurons_Calibration) ;
    Head_Velocity = filter(b,a,Head_Velocity) ;Head_Velocity(Head_Velocity>Vmax)=Vmax;Head_Velocity(Head_Velocity<-Vmax)=-Vmax;
    Az = cumsum(Head_Velocity*dt) ;%Az=mod(Az+180,360)-180;
    
    % Head velocity will be encoded by a series of "head velocity neurons"
    % (see Stringer et al. 2002).
    % The model will use a total of 101 "head velocity neurons". At a given
    % time, only one of them is active.
    % When the head is immobile, the 51th head velocity neuron is active. If the
    % head rotates in one direction, a neuron with a higher index is active
    % If the head rotates in the other direction, a neuron with a lower
    % index is active.    
    % By construction, each velocity neurons causes the hill of activity to
    % slide at a certain velocity. We used stimulations to determine this
    % velocity empirically and map real head velocity (Head_Velocity) 
    % onto an index (Velocity Index)
    Velocity_Index = interp1(Vneurons_Calibration,1:101,VELOCITY_GAIN*Head_Velocity,'nearest') ;
    Velocity_Index(isnan(Velocity_Index))=51;
    
    
    % Initialize the activity of the HD neurons
    hHD = zeros(N_HD_Neurons,1) ; % Activation
    rHD = zeros(N_HD_Neurons,length(time)) ; % Firing rate
    decodedHD = zeros(length(time),1) ; % For visualization purpose, we use a population vector to decode HD from the HD neurons

    for t = 1:length(time)        
        r_ROT = zeros(1,N_Velocity_Neurons)+1 ;
        
        % Compute neurons visual inputs for a give x,y       
        Iv = circshift(Iv0,round(Az(t)*N_HD_Neurons/360)) ;

        % Compute inhibitory and excitatory recurrent inputs
        if t == 1
            Ii = zeros(N_HD_Neurons,1) ; 
            Irot = zeros(N_HD_Neurons,1) ; 
        else
            Ii = sum(K_Inhibitory*rHD(:,t-1)) ;
            Irot = K_Excitatory*Wrot(:,:,Velocity_Index(t))*rHD(:,t-1) ;
        end
        
        % Update neuronal activation (eq. 1)
        hHD = exp(-1/tau)*hHD + 1/tau*(Iv+(Irot-Ii)*RECCURENT_GAIN) ;
        rHD(:,t) = power(1+exp(-2*beta*(hHD-alpha)),-1) ;
                
        a = sum([rHD(:,t).*cos(HD_Neurons_PD') rHD(:,t).*sin(HD_Neurons_PD')])/sum(rHD(:,t)) ;
        decodedHD(t) = atan2d(a(2),a(1)) ;

        if mod(t,10)==1
        subplot(211) ;cla; hold on
        plot(HD_Neurons_PD*180/pi,rHD(:,t),'.r');
        plot(HD_Neurons_PD*180/pi,Iv/K_Visual,'.k')
        axis([-180 180 0 1])
        set(gca,'XTick',-180:45:180) ; xlabel('HD (°)') ;
        ylabel('a.u.')
        legend('HD Neurons firing','Direct visual input','Location','Eastoutside') ;
        
        subplot(212);cla;hold on
        plot(time(1:t),decodedHD(1:t) ,'or') ;
        plot(time(1:t),mod(Az(1:t)+180,360)-180) ;
        legend('Ring HD Signal','Actual HD','Location','Eastoutside') ;
                      
        
        pause(0.001)
        end
    end
  
    
end

%% Figure

% Plot the simulated response of a HD cell
IndexHD = 10 ; % Chose which HD to plot

% In the simulation, the firing of HD encoded by rHD varies from 0 to 1.
% This is an arbitrary scale. In reality, HD cells in the thalamus have
% peak firing rate in the range of 10 to 100 Hz
% So we will scale up the firing to a hand-picked value:
Peak_Firing = 60 ;

FR = rHD(IndexHD,:)*Peak_Firing ;

plot(time, FR) ;

