%% Federated Learning Simulation: IID vs Non-IID Data (+ FedProx mitigation)
% Signal Processing for Machine Learning - Exam Project
%
% Storyline:
%   1) FedAvg converges well when client data is IID.
%   2) Under EXTREME Non-IID data (each client owns only 2 out of 10
%      classes, i.e. no client has ANY information about most classes),
%      FedAvg degrades. FedProx (Li et al., 2020) -- which penalizes
%      local weights drifting away from the global model -- does NOT
%      meaningfully recover this gap, because the problem here is
%      missing information per client, not excessive local drift: a
%      proximal term can limit how far a client moves, but it cannot
%      inject information the client never had access to.
%   3) Under MODERATE Non-IID data (Dirichlet-based label skew: every
%      client sees ALL classes, but in very unbalanced proportions),
%      FedProx is expected to actually help, because here the failure
%      mode of FedAvg genuinely is client drift -- exactly what FedProx
%      is designed to mitigate. This is also the regime under which
%      FedProx is evaluated in its original paper.
%
% Model: multinomial logistic regression (softmax regression), trained
% with FedAvg / FedProx (local gradient descent + weighted averaging).
%
% Dataset: MATLAB's built-in "Digit Dataset" (10 classes, 0-9, 28x28
% grayscale images), shipped as sample data with the Deep Learning
% Toolbox. Falls back to local MNIST CSV files, then to a synthetic toy
% dataset, if unavailable (see LOAD DATA section).

clear; clc; close all;
rng(1);  % reproducibility

%% ------------------ 1. LOAD DATA ------------------
% Tries, in order: (1) Deep Learning Toolbox's built-in Digit Dataset,
% (2) local MNIST CSV files (run download_mnist_csv.m once to get them),
% (3) synthetic toy dataset as last-resort fallback.
digitDatasetPath = fullfile(matlabroot,'toolbox','nnet','nndemos', ...
    'nndatasets','DigitDataset');

if isfolder(digitDatasetPath)
    fprintf('Using Digit Dataset (Deep Learning Toolbox sample data).\n');
    imds = imageDatastore(digitDatasetPath, ...
        'IncludeSubfolders', true, 'LabelSource', 'foldernames');
    [X, Y] = readAllDigitData(imds);
    numClasses = 10;
elseif isfile('mnist_train.csv') && isfile('mnist_test.csv')
    fprintf('Using MNIST CSV files found in the current folder.\n');
    [X, Y] = readMnistCSV('mnist_train.csv', 'mnist_test.csv');
    numClasses = 10;
else
    warning(['Neither the Digit Dataset nor mnist_train.csv/mnist_test.csv ' ...
        'were found. Falling back to a synthetic toy dataset. Run ' ...
        'download_mnist_csv.m first if you want the real digit data.']);
    [X, Y, numClasses] = generateToyData();
end

numFeatures = size(X,2);

% Feature standardization (zero mean, unit variance per pixel).
% Raw pixel values (0-1 after /255) are very ill-conditioned for plain
% fixed-step gradient descent: many border pixels are ~always 0 (near-
% zero gradient), while central pixels have high variance. Centering
% and rescaling greatly speeds up convergence for the SAME learning
% rate and number of iterations.
muX = mean(X, 1);
sdX = std(X, 1) + 1e-8;   % epsilon avoids division by zero on dead pixels
X = (X - muX) ./ sdX;

% Train/test split (80/20), stratified per class
[XTrain, YTrain, XTest, YTest] = stratifiedSplit(X, Y, numClasses, 0.8);

fprintf('Loaded dataset: %d samples, %d features, %d classes\n', ...
    size(X,1), numFeatures, numClasses);
fprintf('Train samples: %d | Test samples: %d\n', size(XTrain,1), size(XTest,1));

%% ------------------ 2. FEDERATED LEARNING SETUP ------------------
numClients      = 5;
numRounds       = 100;    % global communication rounds
localEpochs     = 20;     % local gradient steps per round -- kept high so
                           % that, under Non-IID data, plain FedAvg has
                           % enough local computation to drift noticeably
                           % before averaging (this is also the regime
                           % where FedProx has visible room to act).
localLR         = 0.5;    % local learning rate
lambda          = 1e-4;   % L2 regularization
muProx          = 0.3;    % FedProx proximal-term coefficient (0 = plain FedAvg)
alphaDirichlet  = 0.5;    % Dirichlet concentration for the moderate Non-IID
                           % split. Smaller alpha -> more skewed/heterogeneous
                           % (alpha -> 0 approaches the extreme case: all mass
                           % on one class per client). Larger alpha -> closer
                           % to IID (alpha -> inf gives uniform proportions).
                           % 0.5 is a standard "fairly heterogeneous but not
                           % pathological" choice in the FL literature.

%% ------------------ 3. BUILD CLIENT PARTITIONS ------------------
clientsIID      = partitionIID(XTrain, YTrain, numClients);
clientsExtreme  = partitionNonIIDExtreme(XTrain, YTrain, numClients, numClasses);
clientsDirichlet = partitionDirichlet(XTrain, YTrain, numClients, numClasses, alphaDirichlet);

%% ------------------ 4. RUN EXPERIMENTS ------------------
% --- 4a. IID (FedAvg) ---
fprintf('\n--- [1/5] FedAvg on IID partition ---\n');
[accIID, lossIID, Wiid, biid] = runFedAvg(clientsIID, XTest, YTest, numFeatures, ...
    numClasses, numRounds, localEpochs, localLR, lambda, 0);

% --- 4b. Extreme Non-IID (FedAvg) ---
fprintf('\n--- [2/5] FedAvg on EXTREME Non-IID partition ---\n');
[accExtFedAvg, lossExtFedAvg, Wext, bext] = runFedAvg(clientsExtreme, XTest, YTest, ...
    numFeatures, numClasses, numRounds, localEpochs, localLR, lambda, 0);

% --- 4c. Extreme Non-IID (FedProx) ---
fprintf('\n--- [3/5] FedProx on EXTREME Non-IID partition ---\n');
[accExtFedProx, lossExtFedProx, WextProx, bextProx] = runFedAvg(clientsExtreme, XTest, YTest, ...
    numFeatures, numClasses, numRounds, localEpochs, localLR, lambda, muProx);

% --- 4d. Moderate Non-IID / Dirichlet (FedAvg) ---
fprintf('\n--- [4/5] FedAvg on MODERATE (Dirichlet) Non-IID partition ---\n');
[accDirFedAvg, lossDirFedAvg, Wdir, bdir] = runFedAvg(clientsDirichlet, XTest, YTest, ...
    numFeatures, numClasses, numRounds, localEpochs, localLR, lambda, 0);

% --- 4e. Moderate Non-IID / Dirichlet (FedProx) ---
fprintf('\n--- [5/5] FedProx on MODERATE (Dirichlet) Non-IID partition ---\n');
[accDirFedProx, lossDirFedProx, WdirProx, bdirProx] = runFedAvg(clientsDirichlet, XTest, YTest, ...
    numFeatures, numClasses, numRounds, localEpochs, localLR, lambda, muProx);

% --- 4f. Centralized baseline (upper bound: all data pooled, no federation) ---
fprintf('\n--- [Centralized] Softmax regression trained on all data pooled together ---\n');
[Wcentral, bcentral] = localTrain(XTrain, YTrain, zeros(numFeatures, numClasses), ...
    zeros(1, numClasses), numClasses, numRounds*localEpochs, localLR, lambda, 0);
[accCentral, lossCentral] = evaluate(XTest, YTest, Wcentral, bcentral, numClasses);
fprintf('Centralized | Test acc: %.3f | Test loss: %.4f\n', accCentral, lossCentral);

%% ------------------ 5. VISUALIZE CLIENT DATA DISTRIBUTIONS ------------------
figure('Name','Client label distributions','Position',[100 100 1300 400]);
colors = turbo(10);   % oppure: parula(10), jet(10), cool(10), hsv(10), winter(10)
subplot(1,3,1);
plotClientDistribution(clientsIID, numClasses,colors);
title('IID');
subplot(1,3,2);
plotClientDistribution(clientsExtreme, numClasses,colors);
title('Extreme Non-IID (2 classes/client)');
subplot(1,3,3);
plotClientDistribution(clientsDirichlet, numClasses,colors);
title(sprintf('Moderate Non-IID (Dirichlet, \\alpha=%.2f)', alphaDirichlet));

%% ------------------ 6. CONVERGENCE: EXTREME NON-IID SCENARIO ------------------
figure('Name','Convergence - Extreme Non-IID','Position',[100 550 1300 400]);
subplot(1,2,1);
plot(1:numRounds, accIID, '-o', 'Color', '#272DA8','LineWidth', 1.5); hold on;
plot(1:numRounds, accExtFedAvg, '-s','Color', '#B01E29', 'LineWidth', 1.5);
plot(1:numRounds, accExtFedProx, '-^', 'Color', '#EB6F49','LineWidth', 1.5);
xlabel('Communication round'); ylabel('Test accuracy');
legend('IID','Extreme Non-IID (FedAvg)','Extreme Non-IID (FedProx)','Location','southeast');
title('Accuracy - Extreme Non-IID scenario'); grid on;

subplot(1,2,2);
plot(1:numRounds, lossIID, '-o','Color', '#272DA8', 'LineWidth', 1.5); hold on;
plot(1:numRounds, lossExtFedAvg, '-s','Color', '#B01E29', 'LineWidth', 1.5);
plot(1:numRounds, lossExtFedProx, '-^', 'Color', '#EB6F49', 'LineWidth', 1.5);
xlabel('Communication round'); ylabel('Test cross-entropy loss');
legend('IID','Extreme Non-IID (FedAvg)','Extreme Non-IID (FedProx)','Location','northeast');
title('Loss - Extreme Non-IID scenario'); grid on;

%% ------------------ 7. CONVERGENCE: MODERATE (DIRICHLET) NON-IID SCENARIO ------------------
figure('Name','Convergence - Moderate (Dirichlet) Non-IID','Position',[100 1000 1300 400]);
subplot(1,2,1);
plot(1:numRounds, accIID, '-o','Color', '#272DA8', 'LineWidth', 1.5); hold on;
plot(1:numRounds, accDirFedAvg, '-s','Color', '#085E0C', 'LineWidth', 1.5);
plot(1:numRounds, accDirFedProx, '-^','Color', '#1DDE23', 'LineWidth', 1.5);
xlabel('Communication round'); ylabel('Test accuracy');
legend('IID','Dirichlet Non-IID (FedAvg)','Dirichlet Non-IID (FedProx)','Location','southeast');
title(sprintf('Accuracy - Moderate Non-IID scenario (\\alpha=%.2f)', alphaDirichlet)); grid on;

subplot(1,2,2);
plot(1:numRounds, lossIID, '-o','Color', '#272DA8', 'LineWidth', 1.5); hold on;
plot(1:numRounds, lossDirFedAvg, '-s','Color', '#085E0C', 'LineWidth', 1.5);
plot(1:numRounds, lossDirFedProx, '-^','Color', '#1DDE23', 'LineWidth', 1.5);
xlabel('Communication round'); ylabel('Test cross-entropy loss');
legend('IID','Dirichlet Non-IID (FedAvg)','Dirichlet Non-IID (FedProx)','Location','northeast');
title(sprintf('Loss - Moderate Non-IID scenario (\\alpha=%.2f)', alphaDirichlet)); grid on;

%% ------------------ 8. CONFUSION MATRICES (extreme scenario) ------------------
% Shows WHY FedProx does not help in the extreme case: entire classes
% get systematically confused/missed, which a proximal term cannot fix.
predIID     = predictLabels(XTest, Wiid, biid);
predExtAvg  = predictLabels(XTest, Wext, bext);
predExtProx = predictLabels(XTest, WextProx, bextProx);

Ciid     = confusionMatrix(YTest, predIID, numClasses);
CextAvg  = confusionMatrix(YTest, predExtAvg, numClasses);
CextProx = confusionMatrix(YTest, predExtProx, numClasses);

figure('Name','Confusion matrices (extreme Non-IID scenario)','Position',[100 1450 1400 450]);
subplot(1,3,1);
plotConfusion(Ciid, sprintf('IID - FedAvg (acc=%.3f)', accIID(end)));
subplot(1,3,2);
plotConfusion(CextAvg, sprintf('Extreme Non-IID - FedAvg (acc=%.3f)', accExtFedAvg(end)));
subplot(1,3,3);
plotConfusion(CextProx, sprintf('Extreme Non-IID - FedProx (acc=%.3f)', accExtFedProx(end)));

%% ------------------ 9. SUMMARY: FINAL ACCURACY ACROSS ALL SCENARIOS ------------------
finalAcc = [accCentral, accIID(end), accExtFedAvg(end), accExtFedProx(end), ...
            accDirFedAvg(end), accDirFedProx(end)];
scenarioNames = {'Centralized', 'IID', ...
    sprintf('Extreme-FedAvg'), sprintf('Extreme-FedProx'), ...
    sprintf('Dirichlet-FedAvg'), sprintf('Dirichlet-FedProx')};

figure('Name','Summary: final test accuracy','Position',[1450 550 700 450]);
b=bar(finalAcc, 'FaceColor', 'flat');
b.CData(1,:) = hex2rgb('#555555');   % Centralized - grigio neutro (riferimento, non uno scenario FL)
b.CData(2,:) = hex2rgb('#272DA8');
b.CData(3,:) = hex2rgb('#B01E29');
b.CData(4,:) = hex2rgb('#EB6F49');
b.CData(5,:) = hex2rgb('#085E0C');
b.CData(6,:) = hex2rgb('#1DDE23');

set(gca, 'XTick', 1:6, 'XTickLabel', scenarioNames, 'FontSize', 15.5);
ylabel('Final test accuracy', 'FontSize', 18);
title('Final accuracy across all scenarios', 'FontSize', 20);
grid on;
ylim([0, max(finalAcc)*1.15]);
for i = 1:numel(finalAcc)
    text(i, finalAcc(i) + 0.01, sprintf('%.3f', finalAcc(i)), ...
        'HorizontalAlignment', 'center', 'FontSize', 16, 'FontWeight', 'bold');
end

fprintf('\n===================== FINAL RESULTS =====================\n');
fprintf('Centralized (upper bound) : acc = %.3f | loss = %.4f\n', accCentral, lossCentral);
fprintf('IID                       : acc = %.3f | loss = %.4f\n', accIID(end), lossIID(end));
fprintf('Extreme Non-IID  (FedAvg) : acc = %.3f | loss = %.4f\n', accExtFedAvg(end), lossExtFedAvg(end));
fprintf('Extreme Non-IID  (FedProx): acc = %.3f | loss = %.4f\n', accExtFedProx(end), lossExtFedProx(end));
fprintf('Dirichlet Non-IID(FedAvg) : acc = %.3f | loss = %.4f\n', accDirFedAvg(end), lossDirFedAvg(end));
fprintf('Dirichlet Non-IID(FedProx): acc = %.3f | loss = %.4f\n', accDirFedProx(end), lossDirFedProx(end));
fprintf('===========================================================\n');
%% ===================== LOCAL FUNCTIONS =====================

function rgb = hex2rgb(hexStr)
    hexStr = erase(hexStr, '#');
    rgb = [hex2dec(hexStr(1:2)), hex2dec(hexStr(3:4)), hex2dec(hexStr(5:6))] / 255;
end

function [X, Y] = readAllDigitData(imds)
    % Reads all images from the Digit Dataset image datastore,
    % flattens each 28x28 image to a 784-dim row vector, normalizes
    % to [0,1], and returns numeric labels (0-9) taken from folder names.
    n = numel(imds.Files);
    X = zeros(n, 784);
    for i = 1:n
        img = readimage(imds, i);
        X(i,:) = double(reshape(img, 1, [])) / 255;
    end
    Y = double(string(imds.Labels));  % folder names '0'..'9' -> numeric
end

function [X, Y] = readMnistCSV(trainFile, testFile)
    % Reads MNIST from CSV files (785 columns: label, then 784 pixel
    % values 0-255) and returns them merged into a single X, Y pair
    % (the script re-splits train/test itself with stratifiedSplit, so
    % it's fine to just concatenate everything here).
    fprintf('Reading %s and %s (this can take a bit)...\n', trainFile, testFile);
    Dtr = readmatrix(trainFile);
    Dte = readmatrix(testFile);
    D = [Dtr; Dte];
    Y = D(:,1);
    X = D(:,2:end) / 255;
end

function [X, Y, numClasses] = generateToyData()
    % Fallback synthetic dataset: 4 Gaussian blobs in 2D, used only if
    % the Digit Dataset is not available on this machine.
    numClasses = 4;
    nPerClass  = 500;
    centers = [ 0 0; 5 5; 0 5; 5 0 ];
    X = []; Y = [];
    for c = 1:numClasses
        pts = centers(c,:) + 1.2*randn(nPerClass, 2);
        X = [X; pts]; %#ok<AGROW>
        Y = [Y; (c-1)*ones(nPerClass,1)]; %#ok<AGROW>
    end
    numClasses = 4;
end

function [XTr, YTr, XTe, YTe] = stratifiedSplit(X, Y, numClasses, trainFrac)
    XTr = []; YTr = []; XTe = []; YTe = [];
    for c = 0:numClasses-1
        idx = find(Y == c);
        idx = idx(randperm(numel(idx)));
        nTr = round(trainFrac * numel(idx));
        XTr = [XTr; X(idx(1:nTr), :)]; %#ok<AGROW>
        YTr = [YTr; Y(idx(1:nTr))]; %#ok<AGROW>
        XTe = [XTe; X(idx(nTr+1:end), :)]; %#ok<AGROW>
        YTe = [YTe; Y(idx(nTr+1:end))]; %#ok<AGROW>
    end
    p = randperm(size(XTr,1)); XTr = XTr(p,:); YTr = YTr(p);
    p = randperm(size(XTe,1)); XTe = XTe(p,:); YTe = YTe(p);
end

function clients = partitionIID(X, Y, numClients)
    % Random, uniform split across clients: every client sees roughly
    % the same class distribution as the global dataset.
    n = size(X,1);
    idx = randperm(n);
    splits = round(linspace(0, n, numClients+1));
    clients = cell(numClients,1);
    for k = 1:numClients
        sel = idx(splits(k)+1 : splits(k+1));
        clients{k}.X = X(sel,:);
        clients{k}.Y = Y(sel);
    end
end

function clients = partitionNonIIDExtreme(X, Y, numClients, numClasses)
    % Pathological split: each client only owns a subset of classes,
    % with ZERO samples from every other class (e.g. with 10 classes /
    % 5 clients -> 2 consecutive classes each, no overlap).
    if mod(numClasses, numClients) ~= 0
        error(['numClasses must be divisible by numClients for this ' ...
            'simple pathological split.']);
    end
    classesPerClient = numClasses / numClients;
    clients = cell(numClients,1);
    for k = 1:numClients
        classIDs = (k-1)*classesPerClient : (k*classesPerClient - 1);
        sel = ismember(Y, classIDs);
        clients{k}.X = X(sel,:);
        clients{k}.Y = Y(sel);
    end
end

function clients = partitionDirichlet(X, Y, numClients, numClasses, alpha)
    % Moderate heterogeneity split (Hsu et al., 2019 style): for each
    % class, draw a proportion vector over clients from a Dirichlet(alpha)
    % distribution, and assign that class's samples to clients according
    % to those proportions. Every client ends up with ALL classes, but in
    % very unbalanced quantities when alpha is small. This models
    % realistic "label skew" heterogeneity, as opposed to the fully
    % disjoint pathological case in partitionNonIIDExtreme.
    clients = cell(numClients,1);
    for k = 1:numClients
        clients{k}.X = [];
        clients{k}.Y = [];
    end

    for c = 0:numClasses-1
        idx = find(Y == c);
        idx = idx(randperm(numel(idx)));
        n_c = numel(idx);

        p = sampleDirichlet(alpha, numClients);      % proportions, sum to 1
        counts = floor(p * n_c);
        remainder = n_c - sum(counts);
        % distribute leftover samples to the clients with the largest
        % fractional remainders, so the split still uses all n_c samples
        fracs = p * n_c - counts;
        [~, order] = sort(fracs, 'descend');
        for i = 1:remainder
            counts(order(i)) = counts(order(i)) + 1;
        end

        startIdx = 1;
        for k = 1:numClients
            n_k = counts(k);
            if n_k > 0
                sel = idx(startIdx : startIdx + n_k - 1);
                clients{k}.X = [clients{k}.X; X(sel,:)]; %#ok<AGROW>
                clients{k}.Y = [clients{k}.Y; Y(sel)]; %#ok<AGROW>
            end
            startIdx = startIdx + n_k;
        end
    end

    % shuffle each client's local data
    for k = 1:numClients
        p = randperm(size(clients{k}.X,1));
        clients{k}.X = clients{k}.X(p,:);
        clients{k}.Y = clients{k}.Y(p);
    end
end

function p = sampleDirichlet(alpha, numClients)
    % Samples a probability vector (sums to 1) from a symmetric
    % Dirichlet(alpha) distribution, via independent Gamma(alpha,1)
    % draws normalized to sum to one. Implemented from scratch (no
    % Statistics Toolbox dependency).
    g = zeros(1, numClients);
    for k = 1:numClients
        g(k) = sampleGamma(alpha);
    end
    p = g / sum(g);
end

function g = sampleGamma(a)
    % Samples from a Gamma(shape=a, scale=1) distribution.
    % Marsaglia & Tsang (2000) method for a >= 1; for a < 1 uses the
    % standard boosting trick Gamma(a) = Gamma(a+1) * U^(1/a).
    if a < 1
        g = sampleGamma(a + 1) * rand()^(1/a);
        return;
    end
    d = a - 1/3;
    c = 1/sqrt(9*d);
    while true
        x = randn;
        v = (1 + c*x)^3;
        if v > 0
            u = rand;
            if log(u) < 0.5*x^2 + d - d*v + d*log(v)
                g = d*v;
                return;
            end
        end
    end
end

function plotClientDistribution(clients, numClasses, colors)
    % colors: optional numClasses x 3 RGB matrix (values in [0,1]).
    % If omitted, defaults to a 10-color gradient (turbo colormap).
    if nargin < 3 || isempty(colors)
        colors = turbo(numClasses);
    end

    numClients = numel(clients);
    counts = zeros(numClients, numClasses);
    for k = 1:numClients
        for c = 0:numClasses-1
            counts(k, c+1) = sum(clients{k}.Y == c);
        end
    end
    b = bar(counts, 'stacked');
    for i = 1:numClasses
        b(i).FaceColor = colors(i,:);
    end
    xlabel('Client'); ylabel('# samples');
    legend(arrayfun(@(c) sprintf('%d',c), 0:numClasses-1, 'UniformOutput', false), ...
        'Location','eastoutside');
end

function [accHistory, lossHistory, W, b] = runFedAvg(clients, XTest, YTest, ...
        numFeatures, numClasses, numRounds, localEpochs, lr, lambda, mu)
    % Runs FedAvg (mu=0) or FedProx (mu>0): at each round every client
    % does `localEpochs` full gradient-descent steps starting from the
    % current global weights, then the server averages the resulting
    % weights (weighted by the number of local samples). When mu>0, a
    % proximal term mu/2*||w - w_global||^2 is added to each client's
    % local objective, penalizing local weights that drift too far from
    % the global model received at the start of the round.
    if nargin < 10
        mu = 0;
    end

    numClients = numel(clients);
    nk = cellfun(@(c) size(c.X,1), clients);   % samples per client
    totalN = sum(nk);

    W = zeros(numFeatures, numClasses);
    b = zeros(1, numClasses);

    accHistory  = zeros(numRounds,1);
    lossHistory = zeros(numRounds,1);

    for r = 1:numRounds
        Wagg = zeros(size(W));
        bagg = zeros(size(b));

        for k = 1:numClients
            [Wk, bk] = localTrain(clients{k}.X, clients{k}.Y, W, b, ...
                numClasses, localEpochs, lr, lambda, mu);
            weight = nk(k) / totalN;
            Wagg = Wagg + weight * Wk;
            bagg = bagg + weight * bk;
        end

        W = Wagg;
        b = bagg;

        [accHistory(r), lossHistory(r)] = evaluate(XTest, YTest, W, b, numClasses);

        if mod(r, 10) == 0 || r == 1
            fprintf('Round %3d | Test acc: %.3f | Test loss: %.4f\n', ...
                r, accHistory(r), lossHistory(r));
        end
    end
end

function [W, b] = localTrain(X, Y, W, b, numClasses, epochs, lr, lambda, mu)
    % Full-batch gradient descent on a single client's local data,
    % starting from the received global weights (W, b). If mu>0
    % (FedProx), an extra mu*(W - Wglobal) term pulls the local update
    % back towards the weights received from the server.
    if nargin < 9
        mu = 0;
    end

    n = size(X,1);
    Yoh = fullOneHot(Y, numClasses);

    Wglobal = W;   % snapshot of the weights received this round
    bglobal = b;   % (the anchor point for the proximal term)

    for e = 1:epochs
        scores = X*W + b;
        P = softmaxRows(scores);
        grad = (P - Yoh);

        gW = (X' * grad) / n + lambda * W + mu * (W - Wglobal);
        gb = mean(grad, 1) + mu * (b - bglobal);

        W = W - lr * gW;
        b = b - lr * gb;
    end
end

function [acc, loss] = evaluate(X, Y, W, b, numClasses)
    scores = X*W + b;
    P = softmaxRows(scores);
    [~, pred] = max(P, [], 2);
    pred = pred - 1;

    acc = mean(pred == Y);

    Yoh = fullOneHot(Y, numClasses);
    eps_ = 1e-12;
    loss = -mean(sum(Yoh .* log(P + eps_), 2));
end

function P = softmaxRows(S)
    S = S - max(S, [], 2);   % numerical stability
    E = exp(S);
    P = E ./ sum(E, 2);
end

function Yoh = fullOneHot(Y, numClasses)
    n = numel(Y);
    Yoh = zeros(n, numClasses);
    idx = sub2ind(size(Yoh), (1:n)', Y+1);
    Yoh(idx) = 1;
end

function pred = predictLabels(X, W, b)
    % Returns predicted (0-indexed) class labels for X under model (W,b).
    scores = X*W + b;
    P = softmaxRows(scores);
    [~, pred] = max(P, [], 2);
    pred = pred - 1;
end

function C = confusionMatrix(YTrue, YPred, numClasses)
    % C(i,j) = number of test samples with true label (i-1) predicted
    % as label (j-1). Built manually so it works without the Statistics
    % and Machine Learning Toolbox.
    C = zeros(numClasses, numClasses);
    for i = 1:numel(YTrue)
        r = YTrue(i) + 1;
        c = YPred(i) + 1;
        C(r,c) = C(r,c) + 1;
    end
end

function plotConfusion(C, titleStr)
    numClasses = size(C,1);
    imagesc(C);
    colormap(gca, 'jet'); % alternative built-in: 'parula', 'jet', 'hot', 'cool', 'turbo', 'gray', 'winter'
    colorbar;
    axis square;
    set(gca, 'XTick', 1:numClasses, 'XTickLabel', 0:numClasses-1, ...
             'YTick', 1:numClasses, 'YTickLabel', 0:numClasses-1);
    xlabel('Predicted label'); ylabel('True label');
    title(titleStr);

    if numClasses <= 12
        maxVal = max(C(:));
        for r = 1:numClasses
            for c = 1:numClasses
                val = C(r,c);
                if val > 0
                    if val > 0.5*maxVal
                        txtColor = 'w';
                    else
                        txtColor = 'k';
                    end
                    text(c, r, num2str(val), 'HorizontalAlignment', 'center', ...
                        'Color', txtColor, 'FontSize', 7);
                end
            end
        end
    end
end