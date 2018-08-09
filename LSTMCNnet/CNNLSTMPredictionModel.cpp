/* 
 * File:   CNNLSTMPredictionModel.cpp
 * Author: heshan
 * 
 * Created on August 9, 2018, 3:20 PM
 */

#include "CNNLSTMPredictionModel.hpp"

CNNLSTMPredictionModel::CNNLSTMPredictionModel() { }

CNNLSTMPredictionModel::CNNLSTMPredictionModel(ModelStruct * modelStruct) { 
    this->modelStruct = modelStruct;
}

CNNLSTMPredictionModel::CNNLSTMPredictionModel(const CNNLSTMPredictionModel& orig) { }

CNNLSTMPredictionModel::~CNNLSTMPredictionModel() { }

int CNNLSTMPredictionModel::train() {

    int trainDataSize = modelStruct->trainDataSize; // train data size
    double learningRate = modelStruct->learningRate;
    int iterations = modelStruct->trainingIterations; // training iterations with training data
    
    // CNN parameters
    int height = modelStruct->matHeight;
    int width = modelStruct->matWidth;
    int targetsC = modelStruct->targetC;
    
    // LSTM parameters
    int memCells = modelStruct->memCells; // number of memory cells
    int inputVecSize = height*width; // input vector size
    int timeSteps = inputVecSize; // unfolded time steps

    // Adding the time series in to a vector and preprocessing
    dataproc = new DataProcessor();
    fileProc = new FileProcessor();

    timeSeries2 = fileProc->read(modelStruct->dataFile,1);
    timeSeries =  dataproc->process(timeSeries2,1);

    // Creating the input vector Array for LSTM
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


    // Creating the target vector for LSTM
    std::vector<double>::const_iterator first = timeSeries.begin() + inputVecSize;
    std::vector<double>::const_iterator last = timeSeries.begin() + inputVecSize + trainDataSize;
    std::vector<double> targetVector(first, last);

    // Training the LSTM net
    this->lstm = new LSTMNet(memCells,inputVecSize);
    lstm->train(input, targetVector, trainDataSize, timeSteps, learningRate, iterations);
    

    // CNN network structure
    std::tuple<int, int, int> dimensions = std::make_tuple(1,height,width);

    // Generating input matrices for CNN
    Eigen::MatrixXd ** inMatArr;
    Eigen::MatrixXd * inLblArr;
    Eigen::MatrixXd inMat;
    Eigen::MatrixXd inLbl;
    inMatArr = new Eigen::MatrixXd * [trainDataSize];
    inLblArr = new Eigen::MatrixXd[trainDataSize];

    for (int i = 0; i < trainDataSize; i++) {
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
            inLbl(a,0) = timeSeries.at(i + (width*height));
        }
        inLblArr[i] = inLbl;
    }

    // Generating the CNN
    this->cnn = new CNN(dimensions, modelStruct->netStruct);
    // Training the network
    cnn->train(inMatArr, inLblArr, trainDataSize, iterations, learningRate);
    
    return 0;
}

int CNNLSTMPredictionModel::predict(int points, std::string expect, std::string predict) {

    double errorSq = 0, MSE, expected, val;
    int predSize = points;
    
    // LSTM parameters
    double result;
    
    // CNN parameters
    int height = modelStruct->matHeight;
    int width = modelStruct->matWidth;
    Eigen::MatrixXd tstMatArr[1];
    
    int inputVecSize = height*width; // input vector size
    int trainDataSize = modelStruct->trainDataSize; 
    int numPredPoints = modelStruct->numPredPoints;

    // Predictions
    Eigen::MatrixXd prediction;
    // Open the file to write the time series predictions
    std::ofstream out_file;
    out_file.open(predict,std::ofstream::out | std::ofstream::trunc);
    std::ofstream out_file2;
    out_file2.open(expect,std::ofstream::out | std::ofstream::trunc);
    
    // CNN Inputs
    std::vector<double> cnnInVec;
    cnnInVec.clear();
    
    // LSTM Inputs
    std::vector<double> lstmInVec;
    lstmInVec.clear();
    std::vector<double> * input;
    input = new std::vector<double>[1];
    
    double cnnPredPoints[numPredPoints];
    double lstmPredPoints[numPredPoints];

    for (int j = 0; j < numPredPoints; j++) {
        cnnPredPoints[j] = 0;
        lstmPredPoints[j] = 0;
    }
    
    // max and min training values [ CNN ]
    double trainMax = *std::max_element(timeSeries.begin(), timeSeries.begin()+(trainDataSize+(width*height)));
    double trainMin = *std::min_element(timeSeries.begin(), timeSeries.begin()+(trainDataSize+(width*height)));
    // max and min predicted values [ CNN ]
    double predictMax = std::numeric_limits<double>::min();
    double predictMin = std::numeric_limits<double>::max();
    // max and min predicted values [ LSTM ]

    for (int i = 0; i < trainDataSize-inputVecSize; i++) {
        cnnInVec.clear();
        // filling the input vector using time series data
        for (int j = 0; j < inputVecSize; j++) {
            cnnInVec.push_back(timeSeries2.at(i+j));
        }
        cnnInVec = dataproc->process(cnnInVec,0);
    
        // CNN predictions for the trained data set
        tstMatArr[0] = Eigen::MatrixXd::Zero(height,width);
        for (int a = 0; a < height; a++) {
            for (int b = 0; b < width; b++) {
                tstMatArr[0](a,b) = cnnInVec.at(( a * width ) + b);
            }
        }
        
        // max and min values of CNN predictions for the training data set
        prediction = cnn->predict(tstMatArr);
        if (prediction(0,0) > predictMax) predictMax = prediction(0,0);
        if (prediction(0,0) < predictMin) predictMin = prediction(0,0);
                
    }
    
    for (int i = trainDataSize-inputVecSize; i < trainDataSize; i++) {
        cnnInVec.clear();
        // filling the input vector using time series data
        for (int j = 0; j < inputVecSize; j++) {
            cnnInVec.push_back(timeSeries2.at(i+j));
        }
        cnnInVec = dataproc->process(cnnInVec,0);
    
        // CNN predictions for the trained data set
        tstMatArr[0] = Eigen::MatrixXd::Zero(height,width);
        for (int a = 0; a < height; a++) {
            for (int b = 0; b < width; b++) {
                tstMatArr[0](a,b) = cnnInVec.at(( a * width ) + b);
            }
        }
        prediction = cnn->predict(tstMatArr);
        // post process CNN prediction
        val = prediction(0,0);
        val = (val - predictMin)*((trainMax - trainMin)/(predictMax - predictMin)) + trainMin;
        lstmInVec.push_back(val);
        
        
        for (int j = 0; j < numPredPoints; j++) {          
            result = timeSeries2.at(i+inputVecSize);
            lstmPredPoints[((i+inputVecSize+j)%numPredPoints)] += result;     
        }
        result = lstmPredPoints[((i+inputVecSize)%numPredPoints)]/(double)numPredPoints;
        lstmPredPoints[((i+inputVecSize)%numPredPoints)] = 0;
    }

    for (int i = trainDataSize; i < predSize; i++) {

        // LSTM predictions
        input[0] = lstmInVec;
        for (int j = 0; j < numPredPoints; j++) {          
            result = lstm->predict(input); 
            input[0] = std::vector<double>(lstmInVec.begin()+1, lstmInVec.begin()+inputVecSize);
            input[0].push_back(result);
            lstmPredPoints[((i+inputVecSize+j)%numPredPoints)] += result;     
        }
        result = lstmPredPoints[((i+inputVecSize)%numPredPoints)]/(double)numPredPoints;
        lstmPredPoints[((i+inputVecSize)%numPredPoints)] = 0;
        
        // calculating the Mean Squared Error
        expected = timeSeries.at(i+inputVecSize-1);
        errorSq += std::pow(expected-result,2);
        result = dataproc->postProcess(result);
        
        // writing the predictions
        out_file<<result<<"\n";
        out_file2<<timeSeries2.at(i+inputVecSize-1)<<"\n";
        
        cnnInVec.clear();
        for (int j = 0; j < inputVecSize; j++) {
            cnnInVec.push_back(timeSeries2.at(i+j));
        }
        cnnInVec = dataproc->process(cnnInVec,0);
        
        // Filling the matrix for the CNN input
        tstMatArr[0] = Eigen::MatrixXd::Zero(height,width);
        for (int a = 0; a < height; a++) {
            for (int b = 0; b < width; b++) {
                tstMatArr[0](a,b) = cnnInVec.at(( a * width ) + b);
            }
        }
        
        // CNN predictions
        prediction = cnn->predict(tstMatArr);
        // post process CNN prediction
        val = prediction(0,0);
        val = (val - predictMin)*((trainMax - trainMin)/(predictMax - predictMin)) + trainMin;
        
        // adding the CNN predicted value to the LSTM input vector
        lstmInVec = std::vector<double>(lstmInVec.begin()+1, lstmInVec.begin()+inputVecSize);
        lstmInVec.push_back(val);
        
    }
    
    out_file.close();
    out_file2.close();
    
    MSE = errorSq/(predSize-trainDataSize);
    std::cout<<"\nMean Squared Error: "<<MSE<<"\n\n"; 
    
    return 0;
}

