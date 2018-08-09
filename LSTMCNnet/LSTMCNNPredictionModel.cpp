/* 
 * File:   LSTMCNNPredictionModel.cpp
 * Author: heshan
 * 
 * Created on August 8, 2018, 11:43 PM
 */

#include "LSTMCNNPredictionModel.hpp"

LSTMCNNPredictionModel::LSTMCNNPredictionModel() { }

LSTMCNNPredictionModel::LSTMCNNPredictionModel(ModelStruct * modelStruct) {
    this->modelStruct = modelStruct;
}

LSTMCNNPredictionModel::LSTMCNNPredictionModel(const LSTMCNNPredictionModel& orig) { }

LSTMCNNPredictionModel::~LSTMCNNPredictionModel() { }

int LSTMCNNPredictionModel::train() {

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

int LSTMCNNPredictionModel::predict(int points, std::string expect, std::string predict) {

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
    std::vector<double> inVec;
    std::vector<double> * input;
    input = new std::vector<double>[1];
    
    double predPoints[numPredPoints];
    double lstmPredPoints[numPredPoints];

    for (int j = 0; j < numPredPoints; j++) {
        predPoints[j] = 0;
        lstmPredPoints[j] = 0;
    }

    // creating the input for the CNN using lSTM predictions
    for (int i = 0; i < inputVecSize; i++) {
        inVec.clear();
        // filling the input vector using time series data
        for (int j = 0; j < inputVecSize; j++) {
            inVec.push_back(timeSeries2.at(i+j));
        }
        inVec = dataproc->process(inVec,0);
    
        // LSTM network predictions for the trained data set
        input[0] = inVec;
        result = lstm->predict(input);
        cnnInVec.push_back(result);
    }
    
    
    // max and min training values [ CNN ]
    double trainMax = *std::max_element(timeSeries.begin(), timeSeries.begin()+(trainDataSize+(width*height)));
    double trainMin = *std::min_element(timeSeries.begin(), timeSeries.begin()+(trainDataSize+(width*height)));
    // max and min predicted values [ CNN ]
    double predictMax = std::numeric_limits<double>::min();
    double predictMin = std::numeric_limits<double>::max();

    for (int i = inputVecSize; i < trainDataSize; i++) {
        inVec.clear();
        // filling the input vector using time series data
        for (int j = 0; j < inputVecSize; j++) {
            inVec.push_back(timeSeries2.at(i+j));
        }
        inVec = dataproc->process(inVec,0);
    
        // CNN predictions for the trained data set
        tstMatArr[0] = Eigen::MatrixXd::Zero(height,width);
        for (int a = 0; a < height; a++) {
            for (int b = 0; b < width; b++) {
                tstMatArr[0](a,b) = cnnInVec.at(( a * width ) + b);
            }
        }

        for (int j = 0; j < numPredPoints; j++) {      
            prediction = cnn->predict(tstMatArr);
            cnnInVec = std::vector<double>(cnnInVec.begin()+1, cnnInVec.begin()+inputVecSize);
            cnnInVec.push_back(prediction(0,0));
            for (int a = 0; a < height; a++) {
                for (int b = 0; b < width; b++) {
                    tstMatArr[0](a,b) = cnnInVec.at(( a * width ) + b);
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
        
        // LSTM network predictions for the trained data set
        input[0] = inVec;
        result = lstm->predict(input); 
        cnnInVec = std::vector<double>(cnnInVec.begin()+1, cnnInVec.begin()+inputVecSize);
        cnnInVec.push_back(result);
        
    }

    for (int i = trainDataSize; i < predSize; i++) {

        inVec.clear();
        for (int j = 0; j < inputVecSize; j++) {
            inVec.push_back(timeSeries2.at(i+j));
        }
        inVec = dataproc->process(inVec,0);
        
        // Filling the matrix for the CNN input
        tstMatArr[0] = Eigen::MatrixXd::Zero(height,width);
        for (int a = 0; a < height; a++) {
            for (int b = 0; b < width; b++) {
                tstMatArr[0](a,b) = cnnInVec.at(( a * width ) + b);
            }
        }
        
        // CNN predictions
        for (int j = 0; j < numPredPoints; j++) {      
            prediction = cnn->predict(tstMatArr);
            cnnInVec = std::vector<double>(cnnInVec.begin()+1, cnnInVec.begin()+inputVecSize);
            cnnInVec.push_back(prediction(0,0));
            for (int a = 0; a < height; a++) {
                for (int b = 0; b < width; b++) {
                    tstMatArr[0](a,b) = cnnInVec.at(( a * width ) + b);
                }
            }
            predPoints[((i+inputVecSize+j)%numPredPoints)] += prediction(0,0);     
        }
        prediction(0,0) = predPoints[((i+inputVecSize)%numPredPoints)]/(double)numPredPoints;
        predPoints[((i+inputVecSize)%numPredPoints)] = 0;
        
        // post process CNN prediction
        val = prediction(0,0);
        val = (val - predictMin)*((trainMax - trainMin)/(predictMax - predictMin)) + trainMin;
         
        // calculating the Mean Squared Error
        expected = timeSeries.at(i+inputVecSize+1);
        errorSq += std::pow(expected-val,2);
        val = dataproc->postProcess(val);
        
        // writing the 
        out_file<<val<<"\n";
        out_file2<<timeSeries2.at(i+inputVecSize)<<"\n";

        // LSTM predictions
        input[0] = inVec;
        result = lstm->predict(input); 
        cnnInVec = std::vector<double>(cnnInVec.begin()+1, cnnInVec.begin()+inputVecSize);
        cnnInVec.push_back(result);
        
    }
    
    out_file.close();
    out_file2.close();
    
    MSE = errorSq/(predSize-trainDataSize);
    std::cout<<"\nMean Squared Error: "<<MSE<<"\n\n"; 
    
    return 0;
}


