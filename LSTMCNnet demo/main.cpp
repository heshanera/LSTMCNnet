/* 
 * File:   main.cpp
 * Author: heshan
 *
 * Created on June 15, 2018, 4:38 PM
 */

#include <iostream>
#include <algorithm>
#include <vector>
#include <LSTMCNnet.hpp>

/**
 * Time Series = { t, t+1, t+2, .... t+x} 
 * Input  = { {t, t+1, .. t+m} ...., {t+q+1, t+q+2, .. t+n} }
 */
int conv2() {
    
    // Generating a convolutional network
    int width = 25;
    int height = 2;
    int iterations = 20;
    int inputSize = 40;
    int targetsC = 1;
    double learningRate = 0.8;
    
    std::string infiles[] = {
        "seaLevelPressure.txt",
        "InternetTraff.txt",
        "dailyMinimumTemperatures.txt",
        "monthlySunspotNumbers.txt"
    };
    
    std::string inFile = infiles[2];

    // network structure
    std::tuple<int, int, int> dimensions = std::make_tuple(1,height,width);
    
    struct::ConvLayStruct CL1;
    CL1.filterSize = 2; // filter size: N x N
    CL1.filters = 1; // No of filters
    CL1.stride = 1;
    
    struct::PoolLayStruct PL1;
    PL1.poolH = 1; // pool size: N x N
    PL1.poolW = 2;
    
    struct::FCLayStruct FCL1;
    FCL1.outputs = 40; // neurons in fully connected layer
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
    std::vector<double> timeSeries2;
    timeSeries2 = fp.read("datasets/univariate/input/"+inFile,1);
    timeSeries =  dp.process(timeSeries2,1);
        
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
    out_file.open("datasets/univariate/predictions/CNN/predict_"+inFile,std::ofstream::out | std::ofstream::trunc);
    std::ofstream out_file2;
    out_file2.open("datasets/univariate/predictions/CNN/expect_"+inFile,std::ofstream::out | std::ofstream::trunc);
    
    Eigen::MatrixXd tstMatArr[1];
    double errorSq = 0, MSE;
    double expected;
    double val;
    int predSize = 2000;//timeSeries.size() - matSize; // training size 500 points
    
    std::vector<double> inVec;
    int inputVecSize = height*width;
    
    int numPredPoints = 3;
    double predPoints[numPredPoints];
    
    for (int j = 0; j < numPredPoints; j++) {
        predPoints[j] = 0;
    }

    // max and min training values
    double trainMax = *std::max_element(timeSeries.begin(), timeSeries.begin()+(inputSize+(width*height)));
    double trainMin = *std::min_element(timeSeries.begin(), timeSeries.begin()+(inputSize+(width*height)));
    // max and min predicted values
    double predictMax = std::numeric_limits<double>::min();
    double predictMin = std::numeric_limits<double>::max();
    
    for (int i = 0; i < inputSize; i++) {
        inVec.clear();
        for (int j = 0; j < inputVecSize; j++) {
            inVec.push_back(timeSeries2.at(i+j));
        }
        inVec = dp.process(inVec,0);
        
        tstMatArr[0] = Eigen::MatrixXd::Zero(height,width);
        for (int a = 0; a < height; a++) {
            for (int b = 0; b < width; b++) {
                tstMatArr[0](a,b) = inVec.at(( a * width ) + b);
            }
        }

        for (int j = 0; j < numPredPoints; j++) {      
            prediction = cn.predict(tstMatArr);
            inVec = std::vector<double>(inVec.begin(), inVec.begin()+inputVecSize-1);
            inVec.push_back(prediction(0,0));
            for (int a = 0; a < height; a++) {
                for (int b = 0; b < width; b++) {
                    tstMatArr[0](a,b) = inVec.at(( a * width ) + b);
                }
            }
            predPoints[((i+inputVecSize+j)%numPredPoints)] += prediction(0,0);     
        }
        
        if (i >= numPredPoints-1) {
            prediction(0,0) = predPoints[((i+inputVecSize)%numPredPoints)]/(double)numPredPoints;
            if (prediction(0,0) > predictMax) predictMax = prediction(0,0);
            if (prediction(0,0) < predictMin) predictMin = prediction(0,0);
        }
        predPoints[((i+inputVecSize)%numPredPoints)] = 0;
    }
    
    for (int i = inputSize; i < predSize; i++) {

        inVec.clear();
        for (int j = 0; j < inputVecSize; j++) {
            inVec.push_back(timeSeries2.at(i+j));
        }
        inVec = dp.process(inVec,0);
//        trainMax = *std::max_element(inVec.begin(), inVec.end());
//        trainMin = *std::min_element(inVec.begin(), inVec.end());
        
        tstMatArr[0] = Eigen::MatrixXd::Zero(height,width);
        for (int a = 0; a < height; a++) {
            for (int b = 0; b < width; b++) {
                tstMatArr[0](a,b) = inVec.at(( a * width ) + b);
            }
        }
        
//        prediction = cn.predict(tstMatArr);
        for (int j = 0; j < numPredPoints; j++) {      
            prediction = cn.predict(tstMatArr);
            inVec = std::vector<double>(inVec.begin()+1, inVec.begin()+inputVecSize);
            inVec.push_back(prediction(0,0));
            for (int a = 0; a < height; a++) {
                for (int b = 0; b < width; b++) {
                    tstMatArr[0](a,b) = inVec.at(( a * width ) + b);
                }
            }
            predPoints[((i+inputVecSize+j)%numPredPoints)] += prediction(0,0);     
        }
        
        prediction(0,0) = predPoints[((i+inputVecSize)%numPredPoints)]/(double)numPredPoints;
        predPoints[((i+inputVecSize)%numPredPoints)] = 0;
        
        //std::cout<<prediction(0,0)<<"\n"; 
        expected = timeSeries.at(i + (width*height));
        val = prediction(0,0);
        errorSq += pow(val - expected,2);
        
        // post process
        val = (val - predictMin)*((trainMax - trainMin)/(predictMax - predictMin)) + trainMin;
        
        out_file<<dp.postProcess(val)/4+10<<"\n";
        out_file2<<timeSeries2.at(i+inputVecSize)<<"\n";
    }
    MSE = errorSq/(predSize-inputSize);
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
//        std::cout<<prediction<<"\n"; 
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
    float learningRate = 0.001;
    int predictions = 2000; // prediction points
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
    
    std::string inFile = datasets[8];
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
        predPoints[((i+inputVecSize)%numPredPoints)] = 0;
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
            input[0] = std::vector<double>(inVec.begin()+1, inVec.begin()+inputVecSize);
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

/**
 * prediction model from LSTM network
 */
int lstmPredModel(){

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
    
    std::string fileName = datasets[1];
    
    ModelStruct modelStruct;
    modelStruct.memCells = 4;
    modelStruct.trainDataSize = 300;
    modelStruct.inputVecSize = 60;
    modelStruct.learningRate = 0.001;
    modelStruct.trainingIterations = 10; 
    modelStruct.numPredPoints = 1;
    modelStruct.dataFile = "datasets/univariate/input/"+fileName;
    LSTMPredictionModel pm(&modelStruct);
    pm.train();
    
    std::string expect = "datasets/univariate/predictions/LSTM/expect_"+fileName;
    std::string predict = "datasets/univariate/predictions/LSTM/predict_"+fileName;
    pm.predict(1300, expect, predict);
    
    return 0;
}

/**
 * prediction model from CNN
 */
int cnnPredModel(){

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
    
    std::string fileName = datasets[9];
    
    ModelStruct modelStruct;
    modelStruct.trainDataSize = 300;
    modelStruct.matWidth = 50;
    modelStruct.matHeight = 2;
    modelStruct.trainingIterations = 10; 
    modelStruct.learningRate = 1;
    modelStruct.numPredPoints = 1;
    modelStruct.targetC = 1;
    modelStruct.dataFile = "datasets/univariate/input/"+fileName;
    
    struct::ConvLayStruct CL1;
    CL1.filterSize = 2; // filter size: N x N
    CL1.filters = 1; // No of filters
    CL1.stride = 1;

    struct::PoolLayStruct PL1;
    PL1.poolH = 1; // pool size: N x N
    PL1.poolW = 2;

    struct::FCLayStruct FCL1;
    FCL1.outputs = 60; // neurons in fully connected layer
    struct::FCLayStruct FCL2;
    FCL2.outputs = 20; // neurons in fully connected layer
    struct::FCLayStruct FCL3;
    FCL3.outputs = 1; // neurons in fully connected layer

    char layerOrder[] = {/*'C','P',*/'C','P','F','F','F'};
    struct::ConvLayStruct CLs[] = {CL1/*,CL2*/};
    struct::PoolLayStruct PLs[] = {PL1/*,PL2*/};
    struct::FCLayStruct FCLs[] = {FCL1,FCL2,FCL3};

    modelStruct.netStruct.layers = 5;
    modelStruct.netStruct.layerOrder = layerOrder;
    modelStruct.netStruct.CL = CLs;
    modelStruct.netStruct.PL = PLs;
    modelStruct.netStruct.FCL = FCLs;
    
    CNNPredictionModel pm(&modelStruct);
    pm.train();
    
    std::string expect = "datasets/univariate/predictions/CNN/expect_"+fileName;
    std::string predict = "datasets/univariate/predictions/CNN/predict_"+fileName;
    pm.predict(3000, expect, predict);
        
    return 0;
}

/**
 * prediction model from LSTM and CNN
 * both networks are trained from the same data set
 * when predicting the the original time series is given to the LSTM
 * input for the CNN is generated using the outputs of lSTM
 * output from the CNN is the final prediction
 */
int lstmcnnPredModel(){

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
    
    std::string fileName = datasets[9];
    
    ModelStruct modelStruct;
    modelStruct.trainDataSize = 600;
    modelStruct.learningRate = 0.0001;
    modelStruct.trainingIterations = 8; 
    modelStruct.numPredPoints = 1;
    modelStruct.dataFile = "datasets/univariate/input/"+fileName;
    
    // LSTM parameters
    modelStruct.memCells = 6;
    
    // CNN parameters
    modelStruct.matWidth = 30;
    modelStruct.matHeight = 2;
    modelStruct.targetC = 1;
    
    struct::ConvLayStruct CL1;
    CL1.filterSize = 2; // filter size: N x N
    CL1.filters = 1; // No of filters
    CL1.stride = 1;

    struct::PoolLayStruct PL1;
    PL1.poolH = 1; // pool size: N x N
    PL1.poolW = 2;

    struct::FCLayStruct FCL1;
    FCL1.outputs = 80; // neurons in fully connected layer
    struct::FCLayStruct FCL2;
    FCL2.outputs = 10; // neurons in fully connec ted layer
    struct::FCLayStruct FCL3;
    FCL3.outputs = 1; // neurons in fully connected layer

    char layerOrder[] = {/*'C','P',*/'C','P','F','F','F'};
    struct::ConvLayStruct CLs[] = {CL1/*,CL2*/};
    struct::PoolLayStruct PLs[] = {PL1/*,PL2*/};
    struct::FCLayStruct FCLs[] = {FCL1,FCL2,FCL3};

    modelStruct.netStruct.layers = 5;
    modelStruct.netStruct.layerOrder = layerOrder;
    modelStruct.netStruct.CL = CLs;
    modelStruct.netStruct.PL = PLs;
    modelStruct.netStruct.FCL = FCLs;
    
    LSTMCNNPredictionModel pm(&modelStruct);
    pm.train();
    
    std::string expect = "datasets/univariate/predictions/LSTMCNN/expect_"+fileName;
    std::string predict = "datasets/univariate/predictions/LSTMCNN/predict_"+fileName;
    pm.predict(3000, expect, predict);
    
    return 0;
    
}

/**
 * prediction model from LSTM and CNN
 * both networks are trained from the same data set
 * when predicting the the original time series is given to the CNN
 * input for the LSTM is generated using the outputs of CNN
 * output from the LSTM is the final prediction
 */
int cnnlstmPredModel(){

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
    
    std::string fileName = datasets[9];
    
    ModelStruct modelStruct;
    modelStruct.trainDataSize = 600;
    modelStruct.learningRate = 0.0001;
    modelStruct.trainingIterations = 8; 
    modelStruct.numPredPoints = 1;
    modelStruct.dataFile = "datasets/univariate/input/"+fileName;
    
    // LSTM parameters
    modelStruct.memCells = 6;
    
    // CNN parameters
    modelStruct.matWidth = 30;
    modelStruct.matHeight = 2;
    modelStruct.targetC = 1;
    
    struct::ConvLayStruct CL1;
    CL1.filterSize = 2; // filter size: N x N
    CL1.filters = 1; // No of filters
    CL1.stride = 1;

    struct::PoolLayStruct PL1;
    PL1.poolH = 1; // pool size: N x N
    PL1.poolW = 2;

    struct::FCLayStruct FCL1;
    FCL1.outputs = 80; // neurons in fully connected layer
    struct::FCLayStruct FCL2;
    FCL2.outputs = 10; // neurons in fully connec ted layer
    struct::FCLayStruct FCL3;
    FCL3.outputs = 1; // neurons in fully connected layer

    char layerOrder[] = {/*'C','P',*/'C','P','F','F','F'};
    struct::ConvLayStruct CLs[] = {CL1/*,CL2*/};
    struct::PoolLayStruct PLs[] = {PL1/*,PL2*/};
    struct::FCLayStruct FCLs[] = {FCL1,FCL2,FCL3};

    modelStruct.netStruct.layers = 5;
    modelStruct.netStruct.layerOrder = layerOrder;
    modelStruct.netStruct.CL = CLs;
    modelStruct.netStruct.PL = PLs;
    modelStruct.netStruct.FCL = FCLs;
    
    CNNLSTMPredictionModel pm(&modelStruct);
    pm.train();
    
    std::string expect = "datasets/univariate/predictions/CNNLSTM/expect_"+fileName;
    std::string predict = "datasets/univariate/predictions/CNNLSTM/predict_"+fileName;
    pm.predict(3000, expect, predict);
    
    return 0;
    
}

/**
 * prediction model from LSTM and CNN
 * both LSTM and CNN predict the points for the same input vector
 * prediction is combined to give the final output
 */
int lstmcnnfcPredModel(){

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
    
    std::string fileName = datasets[0];
    
    ModelStruct modelStruct;
    modelStruct.trainDataSize = 300;
    modelStruct.learningRate = 0.001;
    modelStruct.trainingIterations = 8; 
    modelStruct.numPredPoints = 1;
    modelStruct.dataFile = "datasets/univariate/input/"+fileName;
    
    // LSTM parameters
    modelStruct.memCells = 6;
    
    // CNN parameters
    modelStruct.matWidth = 35;
    modelStruct.matHeight = 2;
    modelStruct.targetC = 1;
    
    struct::ConvLayStruct CL1;
    CL1.filterSize = 2; // filter size: N x N
    CL1.filters = 1; // No of filters
    CL1.stride = 1;

    struct::PoolLayStruct PL1;
    PL1.poolH = 1; // pool size: N x N
    PL1.poolW = 2;

    struct::FCLayStruct FCL1;
    FCL1.outputs = 40; // neurons in fully connected layer
    struct::FCLayStruct FCL2;
    FCL2.outputs = 10; // neurons in fully connected layer
    struct::FCLayStruct FCL3;
    FCL3.outputs = 1; // neurons in fully connected layer

    char layerOrder[] = {/*'C','P',*/'C','P','F','F','F'};
    struct::ConvLayStruct CLs[] = {CL1/*,CL2*/};
    struct::PoolLayStruct PLs[] = {PL1/*,PL2*/};
    struct::FCLayStruct FCLs[] = {FCL1,FCL2,FCL3};

    modelStruct.netStruct.layers = 5;
    modelStruct.netStruct.layerOrder = layerOrder;
    modelStruct.netStruct.CL = CLs;
    modelStruct.netStruct.PL = PLs;
    modelStruct.netStruct.FCL = FCLs;
    
    LSTMCNNFCPredictionModel pm(&modelStruct);
    pm.train();
    
    std::string expect = "datasets/univariate/predictions/LSTMCNNFC/expect_"+fileName;
    std::string predict = "datasets/univariate/predictions/LSTMCNNFC/predict_"+fileName;
    pm.predict(1300, expect, predict);
    
    return 0;
    
}




int lstmPredAnom(){

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
    
    std::string fileName = datasets[1];
    
    ModelStruct modelStruct;
    modelStruct.memCells = 6;
    modelStruct.trainDataSize = 300;
    modelStruct.inputVecSize = 60;
    modelStruct.learningRate = 0.001;
    modelStruct.trainingIterations = 10; 
    modelStruct.numPredPoints = 1;
    modelStruct.dataFile = "datasets/univariate/input/"+fileName;
    LSTMPredictionModel pm(&modelStruct);
    pm.train();
    
    pm.initPredData("datasets/univariate/anomalyInputs/"+fileName);
    
    std::string expect = "datasets/univariate/predictions/LSTM/expect_"+fileName;
    std::string predict = "datasets/univariate/predictions/LSTM/predict_"+fileName;
    
//    pm.predict(1300, expect, predict);
    pm.predict(1300, expect, predict, 5, 50000);
    
    return 0;
}

int cnnPredAnom(){

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
    
    std::string fileName = datasets[1];
    
    ModelStruct modelStruct;
    modelStruct.trainDataSize = 60;
    modelStruct.matWidth = 20;
    modelStruct.matHeight = 2;
    modelStruct.trainingIterations = 20; 
    modelStruct.learningRate = 0.001;
    modelStruct.numPredPoints = 1;
    modelStruct.targetC = 1;
    modelStruct.dataFile = "datasets/univariate/input/"+fileName;
    
    struct::ConvLayStruct CL1;
    CL1.filterSize = 2; // filter size: N x N
    CL1.filters = 1; // No of filters
    CL1.stride = 1;

    struct::PoolLayStruct PL1;
    PL1.poolH = 1; // pool size: N x N
    PL1.poolW = 2;

    struct::FCLayStruct FCL1;
    FCL1.outputs = 60; // neurons in fully connected layer
    struct::FCLayStruct FCL2;
    FCL2.outputs = 20; // neurons in fully connected layer
    struct::FCLayStruct FCL3;
    FCL3.outputs = 1; // neurons in fully connected layer

    char layerOrder[] = {/*'C','P',*/'C','P','F','F','F'};
    struct::ConvLayStruct CLs[] = {CL1/*,CL2*/};
    struct::PoolLayStruct PLs[] = {PL1/*,PL2*/};
    struct::FCLayStruct FCLs[] = {FCL1,FCL2,FCL3};

    modelStruct.netStruct.layers = 5;
    modelStruct.netStruct.layerOrder = layerOrder;
    modelStruct.netStruct.CL = CLs;
    modelStruct.netStruct.PL = PLs;
    modelStruct.netStruct.FCL = FCLs;
    
    CNNPredictionModel pm(&modelStruct);
    pm.train();
    
    pm.initPredData("datasets/univariate/anomalyInputs/"+fileName);
    
    std::string expect = "datasets/univariate/predictions/CNN/expect_"+fileName;
    std::string predict = "datasets/univariate/predictions/CNN/predict_"+fileName;
//    pm.predict(1100, expect, predict);
    pm.predict(1100, expect, predict, 5, 50000, 420000);
    
    return 0;
}

int lstmcnnfcPredAnom(){

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
        /*10*/ "hr2.txt",
        /*11*/ "averageSpeed.txt",
        /*12*/ "nycTaxi.txt",
        /*13*/ "datasetX.txt"
    };
    
    std::string fileName = datasets[13];
    
    ModelStruct modelStruct;
    modelStruct.trainDataSize = 600;
    modelStruct.learningRate = 0.003;
    modelStruct.trainingIterations = 10; 
    modelStruct.numPredPoints = 1;
    modelStruct.dataFile = "datasets/univariate/input/"+fileName;
    
    // LSTM parameters
    modelStruct.memCells = 6;
    
    // CNN parameters
    modelStruct.matWidth = 60;
    modelStruct.matHeight = 2;
    modelStruct.targetC = 1;
    
    struct::ConvLayStruct CL1;
    CL1.filterSize = 2; // filter size: N x N
    CL1.filters = 1; // No of filters
    CL1.stride = 1;

    struct::PoolLayStruct PL1;
    PL1.poolH = 1; // pool size: N x N
    PL1.poolW = 2;

    struct::FCLayStruct FCL1;
    FCL1.outputs = 80; // neurons in fully connected layer
    struct::FCLayStruct FCL2;
    FCL2.outputs = 40; // neurons in fully connected layer
    struct::FCLayStruct FCL3;
    FCL3.outputs = 1; // neurons in fully connected layer

    char layerOrder[] = {/*'C','P',*/'C','P','F','F','F'};
    struct::ConvLayStruct CLs[] = {CL1/*,CL2*/};
    struct::PoolLayStruct PLs[] = {PL1/*,PL2*/};
    struct::FCLayStruct FCLs[] = {FCL1,FCL2,FCL3};

    modelStruct.netStruct.layers = 5;
    modelStruct.netStruct.layerOrder = layerOrder;
    modelStruct.netStruct.CL = CLs;
    modelStruct.netStruct.PL = PLs;
    modelStruct.netStruct.FCL = FCLs;
    
    LSTMCNNFCPredictionModel pm(&modelStruct);
    pm.train();
    
//    pm.initPredData("datasets/univariate/anomalyInputs/"+fileName);
    
    std::string expect = "datasets/univariate/predictions/LSTMCNNFC/expect_"+fileName;
    std::string predict = "datasets/univariate/predictions/LSTMCNNFC/predict_"+fileName;
//    pm.predict(3500, expect, predict,0.8,0.2);
//    pm.predict(3500, expect, predict, 20, 25, 35, 0.8, 0.2);
    
//    pm.predictNorm(1000, expect, predict,0.5,0.5);
//    pm.predictNorm(3300, expect, predict, 5, 25,280);
    
    pm.predictAdaptNorm(3500, 
            expect, predict, 
            1000/*time*/, 
            20/*similarity vector size*/, 
            25/*marker value*/, 
            18000/*similarity margin*/, 
            0.8/*LSTM weight*/, 
            0.2/*CNN weight*/
    );
    
    return 0;
    
}

int lstmcnnfcNAB(){

    std::string datasets[] = {
        /*0*/ "art_daily_nojump.txt",
        
        /************ realAWSCloudwatch ************/
        /*1*/ "ec2_cpu_utilization_5f5533.txt", // 4032 data points
        /*2*/ "ec2_cpu_utilization_24ae8d.txt", // 4032 data points
        /*3*/ "ec2_cpu_utilization_53ea38.txt", // 4032 data points  
        /*4*/ "ec2_cpu_utilization_77c1ca.txt", // 4032 data points  
        /*5*/ "ec2_cpu_utilization_825cc2.txt", // 4032 data points
        /*6*/ "ec2_cpu_utilization_ac20cd.txt", // 4032 data points
        /*7*/ "ec2_cpu_utilization_c6585a.txt", // 4032 data points
        /*8*/ "ec2_cpu_utilization_fe7f93.txt", // 4032 data points
        
        /*9*/ "ec2_disk_write_bytes_1ef3de.txt", // 4730 data points
        /*10*/"ec2_disk_write_bytes_c0d644.txt", // 4032 data points 
        /*11*/"ec2_network_in_5abac7.txt",       // 4730 data points
        /*12*/"ec2_network_in_257a54.txt",       // 4032 data points
        /*13*/"elb_request_count_8c0756.txt",    // 4032 data points
        /*14*/"grok_asg_anomaly.txt",            // 4621 data points
        /*15*/"iio_us-east-1_i-a2eb1cd9_NetworkIn.txt", // 1243 data points
        /*16*/"rds_cpu_utilization_cc0c53.txt",  // 4032 data points
        /*17*/"rds_cpu_utilization_e47b3b.txt",  // 4032 data points
                
        /************ realAdExchange ************/        
        /*18*/"exchange-2_cpc_results.txt",      // 1624 data points
        /*19*/"exchange-2_cpm_results.txt",      // 1624 data points
        /*20*/"exchange-3_cpc_results.txt",      // 1538 data points
        /*21*/"exchange-3_cpm_results.txt",      // 1538 data points
        /*22*/"exchange-4_cpc_results.txt",      // 1643 data points
        /*23*/"exchange-4_cpm_results.txt",      // 1643 data points
        
        
    };
    
    std::string fileName = datasets[23];
    
    ModelStruct modelStruct;
    modelStruct.trainDataSize = 200;
    modelStruct.learningRate = 0.01;
    modelStruct.trainingIterations = 10; 
    modelStruct.numPredPoints = 1;
    modelStruct.dataFile = "datasets/univariate/NAB/input/"+fileName;
    
    // LSTM parameters
    modelStruct.memCells = 10;
    
    // CNN parameters
    modelStruct.matWidth = 10 ;
    modelStruct.matHeight = 2;
    modelStruct.targetC = 1;
    
    struct::ConvLayStruct CL1;
    CL1.filterSize = 2; // filter size: N x N
    CL1.filters = 1; // No of filters
    CL1.stride = 1;

    struct::PoolLayStruct PL1;
    PL1.poolH = 1; // pool size: N x N
    PL1.poolW = 2;

    struct::FCLayStruct FCL1;
    FCL1.outputs = 40; // neurons in fully connected layer
    struct::FCLayStruct FCL2;
    FCL2.outputs = 20; // neurons in fully connected layer
    struct::FCLayStruct FCL3;
    FCL3.outputs = 1; // neurons in fully connected layer

    char layerOrder[] = {'C','P','F','F','F'};
    struct::ConvLayStruct CLs[] = {CL1};
    struct::PoolLayStruct PLs[] = {PL1};
    struct::FCLayStruct FCLs[] = {FCL1,FCL2,FCL3};

    modelStruct.netStruct.layers = 5;
    modelStruct.netStruct.layerOrder = layerOrder;
    modelStruct.netStruct.CL = CLs;
    modelStruct.netStruct.PL = PLs;
    modelStruct.netStruct.FCL = FCLs;
    
    LSTMCNNFCPredictionModel pm(&modelStruct);
    pm.train();
    
    std::string expect = "datasets/univariate/NAB/predictions/LSTMCNNFC/expect_"+fileName;
    std::string predict = "datasets/univariate/NAB/predictions/LSTMCNNFC/predict_"+fileName;
//    pm.predict(1620, expect, predict, 0.5, 0.5);
    pm.predict(1620, expect, predict, 3, 5, 10, 0.5, 0.5);
//    pm.dtwSimilarity(1620, expect, predict, 3, 0.5, 0.5);
    
//    pm.predictNorm(3950, expect, predict, 0.2, 0.8);
//    pm.predictNorm(3950, expect, predict, 5, 50, 190, 0.2, 0.8);
    
    return 0;
    
}

/*
 * 
 */
int main(int argc, char** argv) {
    
    //lstmPredModel();
    //cnnPredModel();
    //lstmcnnPredModel();
    //cnnlstmPredModel();
    //lstmcnnfcPredModel();
    
    // multiple prediction with known anomalies ///////////////////////////////
    //lstmPredAnom();
    //cnnPredAnom();
    //lstmcnnfcPredAnom();
    
    // Numenta Anomaly Benchmark //////////////////////////////////////////////
    lstmcnnfcNAB();
    
    return 0;
}
