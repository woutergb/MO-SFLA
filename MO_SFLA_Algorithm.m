function MO_SFLA_Algorithm
% This function is used to apply the MO SFLA to several benchmark problems
% in Matlab.
% The true Pareto fronts for the problems selected can be obtained form the
% EMOO home page (www.lania.mx/~ccoello/, cited on 20 November 2011).
% The Excel file called referring to these true Pareto fronts
% must be in the same folder as the code and an exact copy can be obtained
% at: https://docs.google.com/open?id=0B0h91W0RUck1UlZxcndtdVc5S2M

% This function calculates the Hypervolume and additive epsilon indicators.
% The averages and CI halfwidths are calulated and saved in an Excel file
% after every 100 repitions up to 1000. The Excel file is called: 
% '2_MOP1_1.xlsx'
% Just edit the name as you see fit. Note MOP 8,9 and 10 refers to 
% ZDT 1,2 and 3.

% Currently the code draws a figure meant for one repetition. 
% When more than one repition is executed, just comment out 
% the figure.(Search for 
% "Show the known Pareto front and the approximate front")

% Different acceleration factor parameter values can be used, which relate
% to different acceleration factor formulas. This is done by assigning
% the values 0, 40 or 600 to PPP. 

% Author: Wouter Gideon Bam 
% Goal: This code was originally developed to enable research for
% the final year project presented in partial fulfilment of the 
% requiremenets for the degree of Industrial Engineering at Stellenbosch
% University (BEng Industrial) - December 2012
% Date: 18 October 2012
% Study leader: James Bekker
% Stellenbosch University: Department of Industrial Engineering


%External archive preserving non-dominated solutions
global Elite;
%The best global solution; 
global Xg;
%The best frog in each memeplex; 
global Xb; 
%The worst frog in each memeplex
global Xw;
%All frogs
global Frogs;
%An array to arrange frogs in memeplex
global Memeplex;
%Global array for Pareto front from Coello
global ParFr

%The researcher can set this value to solve some standard MOO problems:
%1,2,3,4,6 in Coello Coello, and (8,9,10) = ZDT1, 2, 3.
MOP = 10
PPP = 40
Reps = 1;
%Array for results
Results = zeros(Reps,3);

%Initialize reference point
if MOP == 1
    Ref(1) = 4.5;
    Ref(2) = 4.5;
else
    if MOP == 2
        Ref(1) = 1;
        Ref(2) = 1;
    else
        if MOP == 3
            Ref(1) = -30;
            Ref(2) = -30;
        else
            if MOP == 4
                Ref(1) = -13;
                Ref(2) = 2;
            else
                if MOP == 6
                    Ref(1) = 1;
                    Ref(2) = 12;
                else
                    if MOP == 8
                        Ref(1) = 1;
                        Ref(2) = 7;
                    else
                        if MOP == 9
                            Ref(1) = 1;
                            Ref(2) = 7;
                        else
                            if MOP == 10
                                Ref(1) = 1;
                                Ref(2) = 7;
                            end
                        end
                    end
                end
            end
        end
    end       
end

%Problem parameters were coded in InitializeProblem to make the code
%more generic. "Limits" is used as a varying limit vector in the histogram
%assignments, while "L" is kept fixed. These initially have the same values
%(the limits of the decision variables).

%Clear all vectors:
WorkArea = []; 
Elite = [];
%Xg=[]; 
Xb=[]; 
Xw=[]; 
Xstep=[];

%Number of memeplexes
m = 10
%Number of frogs per memeplex
k = 100
%Number of frogs 
N = m*k
%Number of frogs per sub-memeplex
q = 66
%Number of evolutions before reshuffling frogs into new memeplexes
r = 40

%Try to achieve good solutions with "few" tries.
MaxEvaluations = 10000

[NumVars, NumObjectives, Limits, L, SheetName, ProblemN] = ...
InitializeProblem(MOP);

tic  %Start clock, not necessary.

for repitition = 1:Reps
    
    WorkArea = []; 
    Elite = []; 
    Xb=[]; 
    Xw=[]; 
    Xstep=[];
    K=[];
    Memeplex = [];
    Frogs = [];
    Choice = [];
    NumberXw = [];
    NumberXb = [];
    ProbMeme = [];
    Xtemp = [];
    Xstep = [];
    
    t = 0;
    
    % ~~~~~~~~~~  MAIN LOOP ~~~~~~~~~~  
    K = NumVars + NumObjectives; %Store value for later use
    Memeplex = zeros(N,K+2); %intilialize the size of the memeplex vector
    Frogs = zeros(N,K+1); % Initialize frogs
    
    %%initialise random frogs
    for LoopFrogs = 1:N
        for LoopVar = 1:NumVars
            Frogs(LoopFrogs, LoopVar) = L(LoopVar,1) + rand(1)*(L(LoopVar,2)-L(LoopVar,1));
        end
    end

    %%Get function values of frogs
    NumEvaluations = N;
    Frogs(1:N, NumVars+1) = f1(Frogs, NumVars, MOP);
    Frogs(1:N, NumVars+2) = f2(Frogs, NumVars, MOP);

    %Show frogs (optional)
    Frogs;

    %intialize the first column of the memplex vector.
    for meme = 1:m
        Memeplex((meme-1)*k+1:meme*k,1) = meme;
    end
    
    %Sort frogs with ranking contained in this file below the main function
    Frogs = Rank(Frogs, N, NumVars, NumObjectives, MOP);
    %Save globally best frog
    Xg = Frogs(1,:);

    while (NumEvaluations + m <= MaxEvaluations)  %repeat dividing into memeplex until max evaluations is reached.
        %Divide the frogs into memeplexes
        t = t+1;
        
        %Choose acceleration factors based on problem. Further research
        %into the choice of these factors are suggested
        if PPP == 0
            P = 1;
        else
            if PPP == 40
                P = PPP /t; 
            else
                if PPP == 600
                    P = PPP/(t^2);
                end
            end
        end
        
        %Divide frogs into memeplexes
        for meme = 1:m
            for nfrog = 1:k
                for Col = 1:K
                    Memeplex((meme-1)*k+nfrog,Col+1) = Frogs((nfrog-1)*m+meme,Col);
                end
            end
        end

        for evolution = 1:r %Repeat dividing into submemeplex and updating worst frog r times.
            
                        
            if NumEvaluations + m <= MaxEvaluations
                %%Array of probabilities of choosing an element in a memeplex for a
                %submemeplex
                %Could place this outside of the for loop to save on computational
                %time
                
                %%Calculate probability and cumulative probability that a
                %%position in a memeplex will be included in sub-memeplex 
                ProbMeme = zeros(k,1);
                for FrogNumberInMeme = 1:k
                    ProbMeme(FrogNumberInMeme,1) = 2*(k+1-FrogNumberInMeme)/(k*(k+1));
                    if FrogNumberInMeme == 1
                        ProbMeme(FrogNumberInMeme,2) = ProbMeme(FrogNumberInMeme,1);
                    end
                    if FrogNumberInMeme >=2
                        ProbMeme(FrogNumberInMeme,2) = ProbMeme(FrogNumberInMeme-1,2)+ ProbMeme(FrogNumberInMeme,1);
                    end
                end

                %%Divide frogs in memeplex into submemeplex - 
                % I noticed that it might happen that the same frog is 
                % selected twice, this should be changed, but will not 
                % undermine the algorithm as only the best and worst frog
                % in each sub-memeplex is used.
                Choice = rand(m,q); 
                NumberXw = zeros(m,1);
                NumberXb = m*ones(m,1);
                for meme = 1:m 
                    for qs = 1:q
                        if (0 <= Choice(meme,qs))&&(Choice(meme,qs)<ProbMeme(1,2)) %If first frog in a memeplex is chosen
                             %Submemeplex((meme-1)*q+qs,:) = Memeplex((meme-1)*m+1,:);  %Assign frog to submemeplex
                             %NumberXw(meme,1) = 1; Irrelevant as the best frog
                             %will never be the worst
                             NumberXb(meme,1) = 1;
                        end
                        for ks = 2:k %If any other frog is chosen
                            if (ProbMeme(ks-1,2) <= Choice(meme,qs))&&(Choice(meme,qs)<ProbMeme(ks,2))
                                %Submemeplex((meme-1)*q+qs,:) = Memeplex((meme-1)*m+ks,:);
                                %Keep track of worst frog from each memeplex selected for
                                %submemeplex
                                if ks > NumberXw(meme,1)
                                    NumberXw(meme,1) = ks;
                                end
                                %Keep track of best frog from each memeplex
                                %selected for submemeplex
                                if ks < NumberXb(meme,1)
                                    NumberXb(meme,1) = ks;
                                end
                            end
                        end
                    end
                end

                %%Update the worst frog in each sub-memeplex
                Xw = zeros(m,K+1); %initialize vector to hold values of worst frogs
                Xb = zeros(m,K+1); %initialize vector to hold values of best frogs
                for meme = 1:m
                    for col = 1:K
                        Xw(meme,col) = Memeplex( (meme-1) * m + NumberXw(meme,1), col + 1);
                    end
                end
                for meme = 1:m
                    for col = 1:K
                        Xb(meme,col) = Memeplex( (meme-1) * m + NumberXb(meme, 1), col + 1);
                    end
                end

                Xtemp = zeros(m,K); %Initialize array to keep all the proposed updates for the worst frogs in each vector.
                Xstep = zeros(m,NumVars); %Initialize array to keep all the proposed steps for the worst frogs in each vector.
                %Xdiff = zeros(m,NumVars);
                %Calculate step for the worst frog in every submemeplex
                Xdiff = Xb - Xw;

                for meme = 1:m
                    for variable = 1:NumVars
                        Xstep(meme,variable) = P * rand(1,1) * Xdiff(meme,variable); %Calculate step
                    end
                end
                
                %Create temp update but keep updates in boundaries
                for TestFrog = 1:m
                    for TestVar = 1:NumVars
                        if ((Xw(TestFrog,TestVar)+ Xstep(TestFrog,TestVar)) <= L(TestVar,2))&&((Xw(TestFrog,TestVar)+ Xstep(TestFrog,TestVar)) >= L(TestVar,1))
                            Xtemp(TestFrog,TestVar) = Xw(TestFrog,TestVar)+ Xstep(TestFrog,TestVar);
                        else 
                            if (Xw(TestFrog,TestVar)+ Xstep(TestFrog,TestVar)) >= L(TestVar,2)
                                Xtemp(TestFrog,TestVar) = L(TestVar,2);
                            else
                                Xtemp(TestFrog,TestVar) = L(TestVar,1);
                            end
                        end
                    end
                end
                
                %Get the fuction values for the new frog
                NumEvaluations = NumEvaluations + m;
                Xtemp(1:m, NumVars + 1) = f1(Xtemp, NumVars, MOP); 
                Xtemp(1:m, NumVars + 2) = f2(Xtemp, NumVars, MOP);

                %Test whether any of the 2 function values is better than the original
                %worst value

                if (NumEvaluations + m <= MaxEvaluations)
                    for meme = 1:m
                        if (Xtemp(meme, NumVars+1) < Xw(meme, NumVars+1)) || (Xtemp(meme, NumVars+2) < Xw(meme, NumVars+2)) % other way around for MOP 3            
                            for col = 1:K
                                Memeplex( (meme-1) * k + NumberXw(meme,1), col + 1) = Xtemp(meme,col);
                            end
                        else
                            for variable = 1:NumVars
                                Xstep(meme,variable) = P*rand(1,1)*(Xg(1,variable)-Xw(meme,variable));
                            end
                            
                            %Create new temp keeping boundaries in mind
                            for TestVar = 1:NumVars
                                if ((Xw(meme,TestVar)+ Xstep(meme,TestVar)) <= L(TestVar,2))&&((Xw(meme,TestVar)+ Xstep(meme,TestVar)) >= L(TestVar,1))
                                    Xtemp(meme,TestVar) = Xw(meme,TestVar)+ Xstep(meme,TestVar);
                                else 
                                    if (Xw(meme,TestVar)+ Xstep(meme,TestVar)) >= L(TestVar,2)
                                        Xtemp(meme,TestVar) = L(TestVar,2);
                                    else
                                        Xtemp(meme,TestVar) = L(TestVar,1);
                                    end
                                end
                            end

                            %Xtemp(meme,1:NumVars) = Xw(meme,1:NumVars)+Xstep(meme,1:NumVars);

                            NumEvaluations = NumEvaluations + 1;
                            Xtemp(meme, NumVars + 1) = f1(Xtemp(meme,:), NumVars, MOP); 
                            Xtemp(meme, NumVars + 2) = f2(Xtemp(meme,:), NumVars, MOP);

                            if (Xtemp(meme, NumVars+1) < Xw(meme, NumVars+1)) || (Xtemp(meme, NumVars+2) < Xw(meme, NumVars+2))  % other way around for MOP 3

                            for col = 1:K
                                Memeplex( (meme-1) * k + NumberXw(meme,1), col + 1) = Xtemp(meme,col);
                            end

                            else 
                                %Update temp by new random frog
                                for LoopVar = 1:NumVars
                                    Xtemp(meme,LoopVar) = L(LoopVar,1) + rand(1,1)*(L(LoopVar,2)-L(LoopVar,1));
                                end
                                NumEvaluations = NumEvaluations + 1;
                                Xtemp(meme, NumVars + 1) = f1(Xtemp(meme,:), NumVars, MOP); 
                                Xtemp(meme, NumVars + 2) = f2(Xtemp(meme,:), NumVars, MOP);

                                for col = 1:K
                                    Memeplex( (meme-1) * k + NumberXw(meme,1), col + 1) = Xtemp(meme,col);
                                end                       
                            end
                        end
                    end
                end

                %%Update Worst frog in each submemeplex in Frogs
                for meme = 1:m
                    for Col = 1:K
                        Frogs((NumberXw(meme,1)-1)*m + meme,Col) = Memeplex((meme-1)*k + NumberXw(meme,1),Col+1);
                    end
                end

                Temp = Rank(Frogs, 0, NumVars, NumObjectives, MOP);
                %Add to Elite:
                Elite = vertcat(Elite, Temp);
            end %end conditional statement ensuring only valid amount of obj evaluations.

        end %%Repeat dividing into submemeplex and updating worst frog r times.

        Frogs = Rank(Frogs, N ,NumVars, NumObjectives, MOP);
        Xg = Frogs(1,:);

    end %%Reshuffle frogs and repeat entire process until 10 000 evaluations are reached.
   
    AvgHyp = 0;
    AvgEps = 0;
    
    %%It's nice to see the progress, but it can slow down the algorithm:
    %ProblemN = 'Problem1';
    %PlotDetailProgress(NumVars, Frogs, ProblemN, t, t, N, repitition,
    %AvgHyp, AvgEps)
    
    %%Plot the intermediate Elite vector (approximation set), for
    %%information purposes only:
    %i = subplot(2,2,2);
    %hold on
    %scatter(Elite(:,NumVars+1), Elite(:, NumVars+2), 3, '*')
    %xlabel('f1')
    %ylabel('f2')
    %title(['Elite vector before final ranking and after ' int2str(NumEvaluations) ' evaluations'])
    %drawnow
    %grid on
    %hold off

    %Final ranking, non-dominated values, th=0:
    Elite=Rank(Elite, 0, NumVars, NumObjectives, MOP);

    %Calculate hyperarea
    hyperarea = HyperArea(Elite, NumVars, Ref);
    Results(repitition,1) = hyperarea;

    %Calculate unary additive epsilon indicator
    ParFr = xlsread('True_PFs_Coello.xls', SheetName);
    EpsInd = Eps(ParFr(:,1:2), Elite(:,NumVars+1:K));
    Results(repitition,2) = EpsInd;
    

    %%Show the known Pareto front and the approximate front':
    %Plot_WorkArea(Elite(:,NumVars+1), Elite(:,NumVars+2),...
    %    MOP, SheetName, NumEvaluations, hyperarea, EpsInd, Ref)

    Results(repitition,3) = NumEvaluations;
    
    %Show at what repition execution is currently at
    repitition
    
    %Occasionally save values for indicators in Excel file
    if repitition == 100
        [status, msginfo] = xlswrite('2_MOP1_1.xlsx',Results,'HypEnEps','A2')
        [status, msginfo] = xlswrite('2_MOP1_1.xlsx',Ref,'HypEnEps','D2')
        [status, msginfo] = xlswrite('2_MOP1_1.xlsx',MOP,'HypEnEps','F2')  
        [status, msginfo] = xlswrite('2_MOP1_1.xlsx',MOP,PPP,'G2')  
    end
    if repitition == 200
        [status, msginfo] = xlswrite('2_MOP1_1.xlsx',Results(101:200,:),'HypEnEps','A102')
        [status, msginfo] = xlswrite('2_MOP1_1.xlsx',Ref,'HypEnEps','D102')
    end
    if repitition == 300
        [status, msginfo] = xlswrite('2_MOP1_1.xlsx',Results(201:300,:),'HypEnEps','A202')
        [status, msginfo] = xlswrite('2_MOP1_1.xlsx',Ref,'HypEnEps','D202')  
    end
    if repitition == 400
        [status, msginfo] = xlswrite('2_MOP1_1.xlsx',Results(301:400,:),'HypEnEps','A302')
        [status, msginfo] = xlswrite('2_MOP1_1.xlsx',Ref,'HypEnEps','D302')
    end
    if repitition == 500
        [status, msginfo] = xlswrite('2_MOP1_1.xlsx',Results(401:500,:),'HypEnEps','A402')
        [status, msginfo] = xlswrite('2_MOP1_1.xlsx',Ref,'HypEnEps','D402')
    end
    if repitition == 600
        [status, msginfo] = xlswrite('2_MOP1_1.xlsx',Results(501:600,:),'HypEnEps','A502')
        [status, msginfo] = xlswrite('2_MOP1_1.xlsx',Ref,'HypEnEps','D502')  
    end
    if repitition == 700
        [status, msginfo] = xlswrite('2_MOP1_1.xlsx',Results(601:700,:),'HypEnEps','A602')
        [status, msginfo] = xlswrite('2_MOP1_1.xlsx',Ref,'HypEnEps','D602')  
    end
    if repitition == 800
        [status, msginfo] = xlswrite('2_MOP1_1.xlsx',Results(701:800,:),'HypEnEps','A702')
        [status, msginfo] = xlswrite('2_MOP1_1.xlsx',Ref,'HypEnEps','D702')
    end
    if repitition == 900
        [status, msginfo] = xlswrite('2_MOP1_1.xlsx',Results(801:900,:),'HypEnEps','A802')
        [status, msginfo] = xlswrite('2_MOP1_1.xlsx',Ref,'HypEnEps','D802')
    end
    if repitition == 1000
        [status, msginfo] = xlswrite('2_MOP1_1.xlsx',Results(901:1000,:),'HypEnEps','A902')
        [status, msginfo] = xlswrite('2_MOP1_1.xlsx',Ref,'HypEnEps','D902')
    end
end

AvgHyp = Results(1,1);
AvgEps = Results(1,2);
AvgIters = Results(1,3);

for repeat = 2:Reps
    AvgHyp = Results(repeat,1) + AvgHyp;
    AvgEps = Results(repeat,2) + AvgEps;
    AvgIters = Results(repeat,3) + AvgIters;
end

%Display results
Results;
%Display average values for indicators
AvgHyp = AvgHyp/Reps
AvgEps = AvgEps/Reps
AvgIters = AvgIters/Reps
    
    %%Show 'the known Pareto front and the approximate front':
    Plot_WorkArea(Elite(:,NumVars+1), Elite(:,NumVars+2),...
        MOP, SheetName, NumEvaluations, hyperarea, EpsInd, Ref)
    axis([0 0.9 -1 2.5])

toc %Timer off

end % Main function 


function Do_f1 = f1(X, NVars, MOP)
    if MOP == 1
       Do_f1=X(:,1).^2;
    elseif MOP == 2
       rt = 1/sqrt(NVars); 
       Do_f1=1-exp(-((X(:,1)-rt).^2+(X(:,2)-rt).^2+(X(:,3)-rt).^2));
    elseif MOP == 3
       A1 = 0.5*sin(1)-2*cos(1)+  sin(2) - 1.5*cos(2);
       A2 = 1.5*sin(1)-  cos(1)+2*sin(2) - 0.5*cos(2);
       B1 = 0.5*sin(X(:,1)) - 2* cos(X(:,1)) + sin(X(:,2)) - 1.5*cos(X(:,2));
       B2 = 1.5*sin(X(:,1)) -    cos(X(:,1)) + 2*sin(X(:,2)) - 0.5*cos(X(:,2));
       Do_f1 = -(1 + (A1 - B1).^2 + (A2 - B2).^2);
    elseif MOP == 4
       Do_f1 = -10*(exp(-0.2*sqrt(X(:,1).^2 + X(:,2).^2)) +...
           exp(-0.2*sqrt(X(:,2).^2 + X(:,3).^2))); 
    elseif MOP == 6
       Do_f1=X(:,1); 
    elseif MOP >= 8
       Do_f1 = X(:,1);  %ZDT1, 2 & 3 
    end   
end
function Do_f2 = f2(X, NVars, MOP)
    if MOP == 1
       Do_f2=(X(:,1)-2).^2; 
    elseif MOP == 2
        rt = 1/sqrt(NVars);
        Do_f2=1-exp(-((X(:,1)+rt).^2+(X(:,2)+rt).^2+(X(:,3)+rt).^2)); 
    elseif MOP == 3
       Do_f2 = -((X(:,1) + 3).^2 + (X(:,2) + 1).^2);
    elseif MOP == 4
       Do_f2   = abs(X(:,1)).^(0.8)+ 5*sin((X(:,1)).^3) + abs(X(:,2)).^(0.8)...
           + 5*sin((X(:,2)).^3) + abs(X(:,3)).^(0.8) + 5*sin((X(:,3)).^3);
    elseif MOP == 6
        x=X(:,1)./(1+10*X(:,2));
        y=x;
        x=x.^2;
        x=1-x;
        Do_f2=(1+10*X(:,2)).*(x - y.*sin(12*pi*X(:,1)));
    elseif MOP >= 8 && MOP <= 10 %ZDT1-3
        c = 9/(NVars-1);  
        x = transpose(sum(transpose(X(:,2:NVars))));
        gx = 1 + x.*c;
        gx_inv = 1./gx;
        if MOP == 8     %ZDT1
            Do_f2 = gx.*(1 - sqrt(gx_inv.*X(:,NVars+1)));
        elseif MOP == 9 %ZDT2
            Do_f2 = gx.*(1 - (gx_inv.*X(:,NVars+1)).^2);
        elseif MOP == 10 %ZDT3
            Ten_Pi = 10*pi;
            Do_f2 = gx.*(1 - sqrt(gx_inv.*X(:,1)) - ...
                gx_inv.*X(:,1).*sin(Ten_Pi*X(:,1)));
        end
    end
end

%PlotDetailProgress
function PlotDetailProgress(NumVars, WorkArea, ProblemN, NoOfLoops, ...
    k, N, Reps, HypAr, EpsInd)
  h = subplot(2,2,[1 3]);
  hold on    
  scatter(WorkArea(1:N, NumVars+1), WorkArea(1:N, NumVars+2), 3, '*');
  xlabel('f1')
  ylabel('f2')
  title(['All frogs at completion after ', num2str(Reps), ' independent repititions with Average Hyperarea of ', num2str(HypAr), ' and Average Epsilon Indicator of ', num2str(EpsInd)]);
  drawnow
  grid on
  hold off
end

function PPF = Plot_WorkArea(x,y, MOP, xlSheetName, NEval, hyperarea, epsInd, Ref)
%subplot(2,2,4);
hold on;
if MOP <= 4 || (MOP >= 6 && MOP < 11)
    PF=[]; %The true Pareto fronts provided by Coello Coello are in Excel
    PF = xlsread('True_PFs_Coello.xls', xlSheetName);
    scatter(PF(:,1), PF(:,2), 5, 'v', 'filled');
end 
scatter(x, y, 3, 'o');
hold off
grid on
xlabel('f1');
ylabel('f2');
legend('True front','MO SFLA','Location','NorthEast')
if MOP <=6
    title(['MO SFLA Pareto front for MOP', int2str(MOP)]);% ...
%    ' after ', int2str(NEval), ' evaluations']);
else
    title(['MO SFLA Pareto front for ZDT', int2str(MOP-7)]);
end

end

function RankIt = Rank(Pop, Threshold, NVars, NObj,MOP)  
K = NVars+NObj;
Pop(:,K+1)=0;
F=[];
Sp=[];
signe = -1; %+1 for MOP3

for z=NVars+1:K - 1
    if MOP == 3
        Pop=sortrows(Pop, -signe*(z));
    else
        Pop=sortrows(Pop, signe*(z));
    end
    for p=1:size(Pop,1)-1
        for q=p+1:size(Pop,1)
            if MOP == 3
                if Pop(p,z+1) <=  Pop(q, z+1)  %Turn around for MOP3, maxim.
                   %Rank is in last col.
                   Pop(p, K+1) = Pop(p, K+1) + 1;   
                    %No need to look further, this candidate is not making it
                   if Pop(p, K+1) > (NObj-1)*Threshold, break, end  
                end
            else
                if Pop(p,z+1) >=  Pop(q, z+1)  %Turn around for MOP3, maxim.
                   %Rank is in last col.
                   Pop(p, K+1) = Pop(p, K+1) + 1;   
                    %No need to look further, this candidate is not making it
                   if Pop(p, K+1) > (NObj-1)*Threshold, break, end  
                end
            end
        end
        if Pop(p, K+1) <= Threshold
           F = vertcat(F, Pop(p, :));
        end
    end    
    F = vertcat(F, Pop(size(Pop,1), :));   %Add the last vector to the set
    F = sortrows(F,K+1);
end
RankIt  = F;

end    

function [NumVars, NumObjectives, Limits, L, SheetName, ProblemN] = ...
    InitializeProblem(MOP)

MOP_Config = [1 1 2 -1E5 1E5, %MOP1   Number, NumVars, NumObj, LowerLimit, HigherLimit
              2 3 2 -4 4,     %MOP2
              3 2 2 -pi pi,   %MOP3
              4 3 2 -4 4,     %MOP4
              5 0 0 0 0,
              6 2 2 0 1,      %MOP6   
              7 0 0 0 0,
              8 30 2 0 1,     %ZDT1
              9 30 2 0 1,     %ZDT2
              10 30 2 0 1     %ZDT3
              ]; 
          
NumVars       = MOP_Config(MOP, 2);
NumObjectives = MOP_Config(MOP, 3);

for i=1:NumVars    %Set problem boundaries
    L(i,1) = MOP_Config(MOP, 4);  
    L(i,2) = MOP_Config(MOP, 5); 
end

ProblemName = ['MOP1 ', 'MOP2 ', 'MOP3 ', 'MOP4 ', 'MOP5 ', 'MOP6 ',...
    'MOP7 ', 'ZDT1 ', 'ZDT2 ', 'ZDT3 '];

for i=1:NumVars
    Limits(i,1) = L(i,1);
    Limits(i,2) = L(i,2);
end;    

ProblemN = ProblemName((MOP-1)*5+1:5*MOP);
SheetName = ProblemN(1:4);
end %function InitializeProblem



%Determine hyperarea of elite vector for minimization problem
function hyperarea = HyperArea(elite, NumVars, Ref)
    %Set the reference point from which to compute the hyperarea.
    Ref(1);
    Ref(2);
    %Determine distance from upper limit to first solution
    Edge1 = Ref(1)- elite(1, NumVars+1);
    Edge2 = Ref(2) - elite(1, NumVars+2);
    ha = Edge1*Edge2; %Determine area of first solution
    
    %Determine area for second solutions and further
    frogs = size(elite,1);
    for frog = 2:frogs
        Edge1 = elite(frog-1,NumVars+1) - elite(frog,NumVars+1);
        Edge2 = Ref(2) - elite(frog,NumVars+2);
        ha = ha + Edge1*Edge2;
    end
    
    hyperarea = ha;
end

%Determine unary additive epsilon indicator for minimization
function EpsInd = Eps(A,B)
eps = [];
eps_j = 0;
eps_k = 0;
NumObj = 2;
A;
B;
sizea = size(A,1);
sizeb = size(B,1);
    
for i = 1:size(A,1)
    for j = 1:size(B,1)
        for k = 1:NumObj
            eps_temp = B(j,k) - A(i,k);           
            if (k == 1)
                eps_k = eps_temp;
            else
                if (eps_k < eps_temp)
                    eps_k = eps_temp;
                end
            end
        end
        
        if (j == 1)
            eps_j = eps_k;
        else
            if (eps_j > eps_k)
                eps_j = eps_k;
            end
        end
    end
    
    if (i == 1)
        eps = eps_j;
    else
        if (eps < eps_j)
            eps = eps_j;
        end
    end
    
end
EpsInd = eps;
end %end function
  
%  * Returns the additive-epsilon value of the paretoFront. This method call to the
%  * calculate epsilon-indicator one
%  * @param paretoFront The pareto front
%  * @param paretoTrueFront The true pareto front
%  * @param numberOfObjectives Number of objectives of the pareto front



