/* 
 * File:   main.cpp
 * Author: heshan
 *
 * Created on June 15, 2018, 4:38 PM
 */

#include <iostream>
#include <LSTMCNnet.hpp>
#include <vector>

/**
 * Time Series = { t, t+1, t+2, .... t+x} 
 * Input  = { {t, t+1, .. t+m} ...., {t+q+1, t+q+2, .. t+n} }
 */
int conv2() {
    
    // Generating a convolutional network
    int width = 20;
    int height = 2;
    int iterations = 20;
    int inputSize = 20;
    int targetsC = 1;
    double learningRate = 1;
    
    std::string infiles[] = {"seaLevelPressure.txt","InternetTraff.txt","dailyMinimumTemperatures.txt"};
    
    std::string inFile = infiles[2];
    
    
    // network structure
    
    std::tuple<int, int, int> dimensions = std::make_tuple(1,height,width);
    
    struct::ConvLayStruct CL1;
    CL1.filterSize = 2; // filter size: N x N
    CL1.filters = 1; // No of filters
    CL1.stride = 1;
//    struct::ConvLayStruct CL2;
//    CL2.filterSize = 4; // filter size: N x N
//    CL2.filters = 3; // No of filters
//    CL2.stride = 1;
    
    struct::PoolLayStruct PL1;
    PL1.poolH = 1; // pool size: N x N
    PL1.poolW = 2;
//    struct::PoolLayStruct PL2;
//    PL2.poolH = 2; // pool size: N x N
//    PL2.poolW = 2;
    
    struct::FCLayStruct FCL1;
    FCL1.outputs = 60; // neurons in fully connected layer
//    FCL1.classes = 4; // target classes
    struct::FCLayStruct FCL2;
    FCL2.outputs = 10; // neurons in fully connected layer
//    FCL2.classes = 1; // target classes
    struct::FCLayStruct FCL3;
    FCL3.outputs = 1; // neurons in fully connected layer
//    FCL3.classes = 1; // target classes
    
    char layerOrder[] = {/*'C','P',*/'C','P','F','F','F'};
    struct::ConvLayStruct CLs[] = {CL1/*,CL2*/};
    struct::PoolLayStruct PLs[] = {PL1/*,PL2*/};
    struct::FCLayStruct FCLs[] = {FCL1,FCL2,FCL3};
    
    
    struct::NetStruct netStruct;
    netStruct.layers = 5;
    netStruct.layerOrder = layerOrder;
    netStruct.CL = CLs;
    netStruct.PL = PLs;
    netStruct.FCL = FCLs;
    
    // Generating input matrices
    Eigen::MatrixXd ** inMatArr;
    Eigen::MatrixXd * inLblArr;
    Eigen::MatrixXd inMat;
    Eigen::MatrixXd inLbl;
    inMatArr = new Eigen::MatrixXd * [inputSize];
    inLblArr = new Eigen::MatrixXd[inputSize];
    
    // Reading the file
    FileProcessor fp;
    DataProcessor dp;
    std::vector<double> timeSeries;
    timeSeries = fp.read("datasets/univariate/input/"+inFile,1);
    timeSeries =  dp.process(timeSeries,1);
        
    for (int i = 0; i < inputSize; i++) {
        // inputs
        inMatArr[i] = new Eigen::MatrixXd[1]; // image depth
        inMat = Eigen::MatrixXd(height,width);
        for (int a = 0; a < height; a++) {
            for (int b = 0; b < width; b++) {
                inMat(a,b) = timeSeries.at(i + ( a * width ) + b);
            }
        }
        inMatArr[i][0] = inMat;
        // labels
        inLbl = Eigen::MatrixXd::Zero(targetsC,1);
        for (int a = 0; a < targetsC; a++) {
//            inLbl(a,0) = timeSeries.at(((i + 1) * width) + a);
            inLbl(a,0) = timeSeries.at(i + (width*height));
        }
        inLblArr[i] = inLbl;
    }
    
    // Generating the network
    CNN cn(dimensions, netStruct);
    // Training the network
    cn.train(inMatArr, inLblArr, inputSize, iterations, learningRate);
    
    // Predictions
    std::cout<<"\n Predictions: \n";
    Eigen::MatrixXd prediction;
    // Open the file to write the time series predictions
    std::ofstream out_file;
    out_file.open("datasets/univariate/predictions/"+inFile,std::ofstream::out | std::ofstream::trunc);
    Eigen::MatrixXd tstMatArr[1];
    double errorSq = 0, MSE;
    double expected;
    double val;
    int predSize = 3000;//timeSeries.size() - matSize; // training size 500 points
    for (int i = 0; i < predSize; i++) {
        tstMatArr[0] = Eigen::MatrixXd::Zero(height,width);
        for (int a = 0; a < height; a++) {
            for (int b = 0; b < width; b++) {
                tstMatArr[0](a,b) = timeSeries.at(i + ( a * width ) + b);
            }
        }
        
        prediction = cn.predict(tstMatArr);
        std::cout<<prediction<<"\n"; 
        expected = timeSeries.at(i + (width*height));
        for (int i = 0; i < targetsC; i++) {
            val = prediction(i,0);
            errorSq += pow(val - expected,2);
            out_file<<val<<"\n"; 
        }
    }
    MSE = errorSq/predSize;
    std::cout<<"\nMean Squared Error: "<<MSE<<"\n\n";
    
    return 0;
}

/**
 * Time Series = { t, t+1, t+2, .... t+x} 
 * Input = { {t, t+1, t+2, ..., t+m},{t, t+1, t+2, ..., t+m}...,{t, t+1, t+2, ..., t+m} } 
 */
int conv() {
    
    // Generating a convolutional network
    
    int width = 20;
    int height = 2;
    int iterations = 20;
    int inputSize = 20;
    int targetsC = 1;
    double learningRate = 1;
    
    std::string infiles[] = {"seaLevelPressure.txt","InternetTraff.txt","dailyMinimumTemperatures.txt"};
    
    std::string inFile = infiles[0];
    
    
    // network structure
    
    std::tuple<int, int, int> dimensions = std::make_tuple(1,height,width);
    
    struct::ConvLayStruct CL1;
    CL1.filterSize = 2; // filter size: N x N
    CL1.filters = 1; // No of filters
    CL1.stride = 1;
//    struct::ConvLayStruct CL2;
//    CL2.filterSize = 4; // filter size: N x N
//    CL2.filters = 3; // No of filters
//    CL2.stride = 1;
    
    struct::PoolLayStruct PL1;
    PL1.poolH = 1; // pool size: N x N
    PL1.poolW = 2;
//    struct::PoolLayStruct PL2;
//    PL2.poolH = 2; // pool size: N x N
//    PL2.poolW = 2;
    
    struct::FCLayStruct FCL1;
    FCL1.outputs = 60; // neurons in fully connected layer
    struct::FCLayStruct FCL2;
    FCL2.outputs = 10; // neurons in fully connected layer
    struct::FCLayStruct FCL3;
    FCL3.outputs = 1; // neurons in fully connected layer
    
    char layerOrder[] = {/*'C','P',*/'C','P','F','F','F'};
    struct::ConvLayStruct CLs[] = {CL1/*,CL2*/};
    struct::PoolLayStruct PLs[] = {PL1/*,PL2*/};
    struct::FCLayStruct FCLs[] = {FCL1,FCL2,FCL3};
    
    
    struct::NetStruct netStruct;
    netStruct.layers = 5;
    netStruct.layerOrder = layerOrder;
    netStruct.CL = CLs;
    netStruct.PL = PLs;
    netStruct.FCL = FCLs;
    
    // Generating input matrices
    Eigen::MatrixXd ** inMatArr;
    Eigen::MatrixXd * inLblArr;
    Eigen::MatrixXd inMat;
    Eigen::MatrixXd inLbl;
    inMatArr = new Eigen::MatrixXd * [inputSize];
    inLblArr = new Eigen::MatrixXd[inputSize];
    
    // Reading the file
    FileProcessor fp;
    DataProcessor dp;
    std::vector<double> timeSeries;
    timeSeries = fp.read("datasets/univariate/input/"+inFile,1);
    timeSeries =  dp.process(timeSeries,1);
        
    for (int i = 0; i < inputSize; i++) {
        // inputs
        inMatArr[i] = new Eigen::MatrixXd[1]; // image depth
        inMat = Eigen::MatrixXd(height,width);
        for (int a = 0; a < height; a++) {
            for (int b = 0; b < width; b++) {
                inMat(a,b) = timeSeries.at(i + b);
            }
        }
        inMatArr[i][0] = inMat;
        // labels
        inLbl = Eigen::MatrixXd::Zero(targetsC,1);
        for (int a = 0; a < targetsC; a++) {
//            inLbl(a,0) = timeSeries.at(((i + 1) * width) + a);
            inLbl(a,0) = timeSeries.at(i + width);
        }
        inLblArr[i] = inLbl;
    }
    
    // Generating the network
    CNN cn(dimensions, netStruct);
    // Training the network
    cn.train(inMatArr, inLblArr, inputSize, iterations, learningRate);
    
    // Predictions
    std::cout<<"\n Predictions: \n";
    Eigen::MatrixXd prediction;
    // Open the file to write the time series predictions
    std::ofstream out_file;
    out_file.open("datasets/univariate/predictions/"+inFile,std::ofstream::out | std::ofstream::trunc);
    Eigen::MatrixXd tstMatArr[1];
    double errorSq = 0, MSE;
    double expected;
    double val;
    int predSize = 3000;//timeSeries.size() - matSize; // training size 500 points
    for (int i = 0; i < predSize; i++) {
        tstMatArr[0] = Eigen::MatrixXd::Zero(height,width);
        for (int a = 0; a < height; a++) {
            for (int b = 0; b < width; b++) {
                tstMatArr[0](a,b) = timeSeries.at(i + b);
            }
        }
        
        prediction = cn.predict(tstMatArr);
        std::cout<<prediction<<"\n"; 
        expected = timeSeries.at(i + width);
        for (int i = 0; i < targetsC; i++) {
            val = prediction(i,0);
            errorSq += pow(val - expected,2);
            out_file<<val<<"\n"; 
        }
    }
    MSE = errorSq/predSize;
    std::cout<<"\nMean Squared Error: "<<MSE<<"\n\n";
    
    return 0;
}

/**
 * univariate time series
 * @return 
 */
int lstm() {
    
    int memCells = 5; // number of memory cells
    int trainDataSize = 300; // train data size
    int inputVecSize = 60; // input vector size
    int timeSteps = 60; // unfolded time steps
    float learningRate = 0.01;
    int predictions = 1300; // prediction points
    int iterations = 10; // training iterations with training data
    
    // Adding the time series in to a vector and preprocessing
    DataProcessor * dataproc;
    dataproc = new DataProcessor();
    FileProcessor * fileProc;
    fileProc = new FileProcessor();
    std::vector<double> timeSeries;
    
    std::vector<double> timeSeries2;
    
    ////////// Converting the CVS ////////////////////////    
    
        
//    fileProc->writeUniVariate("datasets/internetTrafficData.csv","datasets/InternetTraff.txt",2,1);
//    fileProc->writeUniVariate("datasets/monthlyReturnsOfValueweighted.csv","datasets/monthlyReturnsOfValueweighted.txt",2,1);
//    fileProc->writeUniVariate("datasets/treeAlmagreMountainPiarLocat.csv","datasets/treeAlmagreMountainPiarLocat.txt",2,1);
//    fileProc->writeUniVariate("datasets/dailyCyclistsAlongSudurlandsb.csv","datasets/dailyCyclistsAlongSudurlandsb.txt",2,1);
//    fileProc->writeUniVariate("datasets/totalPopulation.csv","datasets/totalPopulation.txt",2,1);
//    fileProc->writeUniVariate("datasets/numberOfUnemployed.csv","datasets/numberOfUnemployed.txt",2,1);
//    fileProc->writeUniVariate("datasets/data.csv","datasets/data.txt",2,1);
//    fileProc->writeUniVariate("datasets/monthlySunspotNumbers.csv","datasets/monthlySunspotNumbers.txt",2,1);
//    fileProc->writeUniVariate("datasets/dailyMinimumTemperatures.csv","datasets/dailyMinimumTemperatures.txt",2,1);    
    
    
    ///////////// Data Sets //////////////////////////////
    
    std::string datasets[] = {
        /* 0*/ "seaLevelPressure.txt",
        /* 1*/ "InternetTraff.txt",
        /* 2*/ "monthlyReturnsOfValueweighted.txt",
        /* 3*/ "treeAlmagreMountainPiarLocat.txt",
        /* 4*/ "dailyCyclistsAlongSudurlandsb.txt",
        /* 5*/ "totalPopulation.txt",
        /* 6*/ "numberOfUnemployed.txt",
        /* 7*/ "data.txt",
        /* 8*/ "monthlySunspotNumbers.txt",
        /* 9*/ "dailyMinimumTemperatures.txt",
        /*10*/ "hr2.txt"
    };
    
    std::string datasets2[] = {
        /* 0*/ "seaLevelPressureAnml.txt",
        /* 1*/ "dailyMinimumTemperaturesAnml.txt"
    };
    
    std::string inFile = datasets[0];
    timeSeries2 = fileProc->read("datasets/univariate/input/"+inFile,1);
    timeSeries =  dataproc->process(timeSeries2,1);
    
    // Creating the input vector Array
    std::vector<double> * input;
    input = new std::vector<double>[trainDataSize];
    std::vector<double> inputVec;
    
    for (int i = 0; i < trainDataSize; i++) {
        inputVec.clear();
        for (int j = 0; j < inputVecSize; j++) {
            inputVec.push_back(timeSeries.at(i+j));
        }
        inputVec =  dataproc->process(inputVec,0);
        input[i] = inputVec; 
    }
    
    
    // Creating the target vector using the time series 
    std::vector<double>::const_iterator first = timeSeries.begin() + inputVecSize;
    std::vector<double>::const_iterator last = timeSeries.begin() + inputVecSize + trainDataSize;
    std::vector<double> targetVector(first, last);
    
    // Training the LSTM net
    LSTMNet lstm(memCells,inputVecSize);
    lstm.train(input, targetVector, trainDataSize, timeSteps, learningRate, iterations);
  
    // Open the file to write the time series predictions
    std::ofstream out_file;
    out_file.open("datasets/univariate/predictions/LSTM/predict_"+inFile,std::ofstream::out | std::ofstream::trunc);
    std::ofstream out_file2;
    out_file2.open("datasets/univariate/predictions/LSTM/expect_"+inFile,std::ofstream::out | std::ofstream::trunc);
    
    std::vector<double> inVec;
    input = new std::vector<double>[1];
    double result;
    double expected;
    double MSE = 0;
    
    int numPredPoints = 3;
    double predPoints[numPredPoints];
    
    for (int j = 0; j < numPredPoints; j++) {
        predPoints[j] = 0;
    }
    
    std::cout << std::fixed;
    
    for (int i = 0; i < numPredPoints-1; i++) {
        inVec.clear();
        for (int j = 0; j < inputVecSize; j++) {
            inVec.push_back(timeSeries2.at(i+j));
        }
        
        inVec = dataproc->process(inVec,0);
        input[0] = inVec;
        for (int j = 0; j < numPredPoints; j++) {          
            result = lstm.predict(input); 
            input[0] = std::vector<double>(inVec.begin(), inVec.begin()+inputVecSize-2);
            input[0].push_back(result);
            predPoints[((i+inputVecSize+j)%numPredPoints)] += result;     
        }
    }
    
    for (int i = numPredPoints-1; i < predictions; i++) {
        
        inVec.clear();
        for (int j = 0; j < inputVecSize; j++) {
            inVec.push_back(timeSeries2.at(i+j));
        }
        
        inVec = dataproc->process(inVec,0);
        input[0] = inVec;
        for (int j = 0; j < numPredPoints; j++) {          
            result = lstm.predict(input); 
            input[0] = std::vector<double>(inVec.begin(), inVec.begin()+inputVecSize-2);
            input[0].push_back(result);
            predPoints[((i+inputVecSize+j)%numPredPoints)] += result;     
        }
        
        result = predPoints[((i+inputVecSize)%numPredPoints)]/(double)numPredPoints;
        predPoints[((i+inputVecSize)%numPredPoints)] = 0;
        
        // calculating the Mean Squared Error
        expected = timeSeries.at(i+inputVecSize+1);
        MSE += std::pow(expected-result,2);
        
        result = dataproc->postProcess(result);
        out_file<<result<<"\n";
        out_file2<<timeSeries2.at(i+inputVecSize)<<"\n";
        
//        inVec = dataproc->process(inVec,0);
//        input[0] = inVec;
        
//        result = lstm.predict(input);
//        std::cout<<std::endl<<"result: "<<result<<std::endl;
        
//        result = dataproc->postProcess(result);
//        expected = timeSeries2.at(i+inputVecSize+1);
//        MSE += std::pow(expected-result,2);
        
//        result = dataproc->postProcess(result);
//        out_file<<result<<"\n";
//        std::cout<<"result processed: "<<result<<std::endl<<std::endl;
        
//        out_file2<<timeSeries2.at(i+inputVecSize)<<"\n";
    }
  
    out_file.close();
    out_file2.close();
    
    MSE /= predictions;
    std::cout<<"Mean Squared Error: "<<MSE<<"\n";
    std::cout << std::scientific;
    return 0;
}

/**
 * multivariate time series
 * @return 
 */
int lstm2() {

    ///////////////////////// Multivariate time series data prediction ////////////////////////////////////
    
    int memCells = 10; // number of memory cells
    int inputVecSize = 5; // input vector size
    int trainDataSize = 5000; // train data size
    int timeSteps = 1; // data points used for one forward step
    float learningRate = 0.0001;
    int iterations = 10; // training iterations with training data
    int lines = 5000;
    
    DataProcessor * dataproc;
    dataproc = new DataProcessor();
    FileProcessor * fileProc;
    fileProc = new FileProcessor();
    
    int colIndxs[] = {0,0,1,1,1,1,1};
    int targetValCol = 7;
    
    std::vector<double> * timeSeries;
    timeSeries = fileProc->readMultivariate("datasets/multivariate/input/occupancyData/datatraining.txt",lines,inputVecSize,colIndxs,targetValCol);
    
    // Creating the input vector Array
    std::vector<double> * input;
    input = new std::vector<double>[trainDataSize];    
    for (int i = 0; i < trainDataSize; i++) {
        input[i] = dataproc->process(timeSeries[i],0);
    }
    
    // Creating the target vector using the time series 
    std::vector<double>::const_iterator first = timeSeries[lines].begin();
    std::vector<double>::const_iterator last = timeSeries[lines].begin() + trainDataSize;
    std::vector<double> targetVector(first, last);
    for (std::vector<double>::iterator it = targetVector.begin(); it != targetVector.end(); ++it) {
        if (*it == 0) *it = -1;
    }    
    
    // Training the LSTM net
    LSTMNet * lstm = new LSTMNet(memCells,inputVecSize);    
    lstm->train(input, targetVector, trainDataSize, timeSteps, learningRate, iterations);
  
    // Predictions
    int predictions = 2000; // prediction points
    lines = 2000; // lines read from the files
    
    timeSeries = fileProc->readMultivariate("datasets/multivariate/input/occupancyData/datatest.txt",lines,inputVecSize,colIndxs,targetValCol);
    input = new std::vector<double>[1];
    double result;
    double min = 0, max = 0;
    std::vector<double> resultVec;
    for (int i = 0; i < predictions; i++) {    
        input[0] = dataproc->process(timeSeries[i],0);
        result = lstm->predict(input);
        resultVec.push_back(result);
        //std::cout<<std::endl<<"result: "<<result<<"  ==>  expected: "<<timeSeries[lines].at(i)<<std::endl;
        
        if (i == 0){
            min = result;
            max = result;
        } else {
        
            if (min > result) min = result;
            if (max < result) max = result;
        }
    }
    std::cout<<"min: "<<min<<std::endl;
    std::cout<<"max: "<<max<<std::endl;
    
    double line = 0; //(min + max)/2;
    std::cout<<"margin: "<<line<<std::endl<<std::endl;
    
    
    int occu = 0, notoccu = 0;
    
    int corr = 0;
    int incorr = 0;
    
    int truePos = 0;
    int falsePos = 0;
    int trueNeg = 0;
    int falseNeg = 0;
    
    int corrNwMgn = 0;
    int incorrNwMgn = 0;
    
    // Open the file to write the time series predictions
    std::ofstream out_file;
    std::ofstream out_file2;
    out_file.open("datasets/multivariate/predictions/occupancyData/multiResults.txt",std::ofstream::out | std::ofstream::trunc);
    out_file2.open("datasets/multivariate/predictions/occupancyData/multiTargets.txt",std::ofstream::out | std::ofstream::trunc);
    
    for (int i = 0; i < predictions; i++) { 
        out_file<<timeSeries[lines].at(i)<<","<<resultVec.at(i)<<"\n";
        out_file2<<timeSeries[lines].at(i)<<",";
        if (timeSeries[lines].at(i) == 1) {
            out_file2<<1<<"\n";
        } else out_file2<<-1<<"\n";
        
        if ( (resultVec.at(i) > line) && (timeSeries[lines].at(i) == 1)) { 
            corr++;
            truePos++;
            occu++;
        } else if ( (resultVec.at(i) <= line) && (timeSeries[lines].at(i) == 0)) {
            corr++;
            trueNeg++;
            notoccu++;
        } else if ( (resultVec.at(i) <= line) && (timeSeries[lines].at(i) == 1)) { 
            incorr++; 
            falseNeg++;
            occu++;
        } else if ( (resultVec.at(i) > line) && (timeSeries[lines].at(i) == 0)) { 
            incorr++; 
            falsePos++;
            notoccu++;
        }
        //std::cout<<resultVec.at(i)<<" ------ "<<timeSeries[lines].at(i)<<"\n";
        
    }
    
    std::cout<<std::endl;
    
    std::cout<<"----------------------"<<std::endl;
    std::cout<<"Data "<<std::endl;
    std::cout<<"----------------------"<<std::endl;
    std::cout<<"Occupied: "<<occu<<std::endl;
    std::cout<<"NotOccupied: "<<notoccu<<std::endl<<std::endl;
    
    std::cout<<"----------------------"<<std::endl;
    std::cout<<"margin: "<<line<<std::endl;
    std::cout<<"----------------------"<<std::endl;
    std::cout<<"Correct predictions: "<<corr<<std::endl;
    std::cout<<"Incorrect predictions: "<<incorr<<std::endl<<std::endl;
    
    std::cout<<"True Positive: "<<truePos<<std::endl;
    std::cout<<"True Negative: "<<trueNeg<<std::endl;
    std::cout<<"False Positive: "<<falsePos<<std::endl;
    std::cout<<"False Negative: "<<falseNeg<<std::endl;
    
    std::cout<<std::endl<<"Accuracy: "<<(corr/(double)predictions)*100<<"%"<<std::endl<<std::endl;
    
    
    line = (min + max)/2;
    occu = 0;
    notoccu = 0;
    corr = 0;
    incorr = 0;
    truePos = 0;
    falsePos = 0;
    trueNeg = 0;
    falseNeg = 0;
    
    for (int i = 0; i < predictions; i++) {    
        if ( (resultVec.at(i) > line) && (timeSeries[lines].at(i) == 1)) { 
            corr++;
            truePos++;
            occu++;
        } else if ( (resultVec.at(i) <= line) && (timeSeries[lines].at(i) == 0)) {
            corr++;
            trueNeg++;
            notoccu++;
        } else if ( (resultVec.at(i) <= line) && (timeSeries[lines].at(i) == 1)) { 
            incorr++; 
            falseNeg++;
            occu++;
        } else if ( (resultVec.at(i) > line) && (timeSeries[lines].at(i) == 0)) { 
            incorr++; 
            falsePos++;
            notoccu++;
        }
        
        
        
        if (line > 0) {
            if ( (resultVec.at(i) <= line) && (resultVec.at(i) > 0)) {
                if (timeSeries[lines].at(i) == 0) {
                    corrNwMgn++;
                } else incorrNwMgn++;
            }
        } else {
            if ( (resultVec.at(i) > line) && (resultVec.at(i) < 0)) {
                if (timeSeries[lines].at(i) == 1) {
                    corrNwMgn++;
                } else incorrNwMgn++;
            }
        }
        
    }
    
    std::cout<<"----------------------"<<std::endl;
    std::cout<<"margin: "<<line<<std::endl;
    std::cout<<"----------------------"<<std::endl;
    std::cout<<"Correct predictions: "<<corr<<std::endl;
    std::cout<<"Incorrect predictions: "<<incorr<<std::endl<<std::endl;
    
    std::cout<<"True Positive: "<<truePos<<std::endl;
    std::cout<<"True Negative: "<<trueNeg<<std::endl;
    std::cout<<"False Positive: "<<falsePos<<std::endl;
    std::cout<<"False Negative: "<<falseNeg<<std::endl;
    
    std::cout<<std::endl<<"Accuracy: "<<(corr/(double)predictions)*100<<"%"<<std::endl<<std::endl;
    
    std::cout<<"----------------------"<<std::endl;
    std::cout<<"Within the new margin and 0"<<std::endl;
    std::cout<<"----------------------"<<std::endl;
    std::cout<<"Correct: "<<corrNwMgn<<std::endl;
    std::cout<<"Incorrect: "<<incorrNwMgn<<std::endl<<std::endl<<std::endl;
    
    return 0;
}

/*
 * 
 */
int main(int argc, char** argv) {
    
    //conv2();
    
    lstm();
    
    return 0;
}
