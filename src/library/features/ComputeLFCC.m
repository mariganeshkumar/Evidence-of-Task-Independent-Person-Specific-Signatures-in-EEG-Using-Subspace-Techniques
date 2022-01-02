function fbe = ComputeLFCC(data,windowFrames,windowShiftFrames,nFFT,...
    sampRate,noOfFilters,noOfCoeff,freqBegin,freqEnd,deltaRequired,delta2Required)
    %windowFrames=floor(sampRate*windowLength);
    %windowShiftFrames=floor(sampRate*windowOverLap);
    if (~exist('noOfFilters', 'var'))
        noOfFilters = 23;
    end
    if (~exist('freqBegin', 'var'))
        freqBegin = 300;
    end
    if (~exist('freqEnd', 'var'))
        freqEnd = 3200;
    end
    if (~exist('deltaRequired', 'var'))
        deltaRequired = false;
    end
    if (~exist('delta2Required', 'var'))
        delta2Required = false;
    end
    
    ind=1;
    fbe=[];
    while ind<length(data)
        if ind+windowFrames-1<length(data)
        windowData=data(ind:ind+windowFrames-1);
        else
        windowData=data(ind:end);
        end   
        fbe=[fbe ComputeWindowFBE(windowData,noOfFilters,noOfCoeff,freqBegin,freqEnd,sampRate,nFFT)];
        ind=ind+windowShiftFrames+1;
    end
    if deltaRequired==false
        return;
    end
    deltaFbe=[];
    denominator=1+4+9;
    for frame=1:size(fbe,2)
        delta=zeros(size(fbe,1),1);
        for k=1:3
            startInd=frame-k;
            endInd= frame+k;
            if startInd <=0
                startInd=1;
            end
            if endInd >size(fbe,2)
                endInd=size(fbe,2);
            end
            delta=delta+k*(fbe(:,startInd)-fbe(:,endInd));
        end
        deltaFbe=[deltaFbe delta/denominator];
    end
    if delta2Required==false
        fbe=[fbe; deltaFbe];
        return;
    end
    delta2Fbe=[];
    denominator=1+4+9;
    for frame=1:size(deltaFbe,2)
        delta=zeros(size(deltaFbe,1),1);
        for k=1:3
            startInd=frame-k;
            endInd= frame+k;
            if startInd <=0
                startInd=1;
            end
            if endInd >size(deltaFbe,2)
                endInd=size(deltaFbe,2);
            end
            delta=delta+k*(deltaFbe(:,startInd)-deltaFbe(:,endInd));
        end
        delta2Fbe=[delta2Fbe delta/denominator];
    end
    fbe=[fbe; deltaFbe; delta2Fbe];
end
 

function windowLfCC= ComputeWindowFBE(data,noOfFilters,noOfCoeff,freqBegin,freqEnd,sampRate,nFFT)
    %MF = @(f) 1127.*log10(1 + f./700);
    %invMF = @(m) 700.*(10.^(m/1127)-1);
    ceplifter = @( N, L )( 1+0.5*L*sin(pi*[0:N-1]/L) );
    dctm = @( N, M )( sqrt(2.0/M) * cos( repmat([0:N-1].',1,M).* repmat(pi*(([1:M])-0.5)/M,N,1) ) );
    noOfTriangeEdges = noOfFilters+2; % number of triangular filers
    mm = linspace(freqBegin,freqEnd,noOfTriangeEdges); % equal space in mel-frequency
    ff = mm; % convert mel-frequencies into frequency
    X = fft(data.*hamming(length(data)),nFFT);
    N2 = max([floor(nFFT+1)/2 floor(nFFT/2)+1]); %
    P = abs(X(1:N2,:));%.^2./nFFT; % NoFr no. of periodograms
    filterBanks = triangularFilterShape(ff,N2,sampRate); %
    windowLfCC = dctm(noOfCoeff,noOfFilters)*log(filterBanks'*P);
    lifter = ceplifter( noOfCoeff, 22 );
    %windowLfCC = diag( lifter ) * windowLfCC;
end
 
function [out,k] = triangularFilterShape(f,N2,fs)
    M = length(f);
    k = linspace(0,fs/2,N2);
    out = zeros(N2,M-2);
    for m=2:M-1
        I = k >= f(m-1) & k <= f(m);
        J = k >= f(m) & k <= f(m+1);
        out(I,m-1) = (k(I) - f(m-1))./(f(m) - f(m-1));
        out(J,m-1) = (f(m+1) - k(J))./(f(m+1) - f(m));
    end
end

function [out,k] = triangularEqualAUCFilterShape(f,N2,fs)
    M = length(f);
    k = linspace(0,fs/2,N2);
    out = zeros(N2,M-2);
    for m=2:M-1
        I = k >= f(m-1) & k <= f(m);
        J = k >= f(m) & k <= f(m+1);
        out(I,m-1) = 2*(k(I) - f(m-1))./((f(m+1) - f(m-1))*(f(m) - f(m-1)));
        out(J,m-1) = (f(m+1) - k(J))./((f(m+1) - f(m-1))*(f(m+1) - f(m)));
    end
end