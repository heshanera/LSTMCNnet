/* 
 * File:   LSTMPredictionModel.cpp
 * Author: heshan
 * 
 * Created on August 7, 2018, 5:12 PM
 */

#include "LSTMPredictionModel.hpp"

LSTMPredictionModel::LSTMPredictionModel() { }

LSTMPredictionModel::LSTMPredictionModel(ModelStruct * modelStruct) { 
    this->modelStruct = modelStruct;
}

LSTMPredictionModel::LSTMPredictionModel(const LSTMPredictionModel& orig) { }

LSTMPredictionModel::~LSTMPredictionModel() { }

int LSTMPredictionModel::train() {

    int memCells = modelStruct->memCells; // number of memory cells
    int trainDataSize = modelStruct->trainDataSize; // train data size
    int inputVecSize = modelStruct->inputVecSize; // input vector size
    int timeSteps = modelStruct->inputVecSize; // unfolded time steps
    double learningRate = modelStruct->learningRate;
    int iterations = modelStruct->trainingIterations; // training iterations with training data

    // Adding the time series in to a vector and preprocessing
    dataproc = new DataProcessor();
    fileProc = new FileProcessor();

    timeSeries2 = fileProc->read(modelStruct->dataFile,1);
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
    this->lstm = new LSTMNet(memCells,inputVecSize);
    lstm->train(input, targetVector, trainDataSize, timeSteps, learningRate, iterations);
    
    
    return 0;
}

int LSTMPredictionModel::predict(int points, std::string expect, std::string predict) {

    int inputVecSize = modelStruct->inputVecSize; // input vector size
            
    // Open the file to write the time series predictions
    std::ofstream out_file;
    out_file.open(predict,std::ofstream::out | std::ofstream::trunc);
    std::ofstream out_file2;
    out_file2.open(expect,std::ofstream::out | std::ofstream::trunc);

    std::vector<double> * input;
    std::vector<double> inVec;
    input = new std::vector<double>[1];
    double result;
    double expected;
    double MSE = 0;

    int numPredPoints = modelStruct->numPredPoints;
    double predPoints[numPredPoints];

    for (int j = 0; j < numPredPoints; j++) {
        predPoints[j] = 0;
    }

    for (int i = 0; i < numPredPoints-1; i++) {
        inVec.clear();
        for (int j = 0; j < inputVecSize; j++) {
            inVec.push_back(timeSeries2.at(i+j));
        }

        inVec = dataproc->process(inVec,0);
        input[0] = inVec;
        for (int j = 0; j < numPredPoints; j++) {          
            result = lstm->predict(input); 
            input[0] = std::vector<double>(inVec.begin(), inVec.begin()+inputVecSize-2);
            input[0].push_back(result);
            predPoints[((i+inputVecSize+j)%numPredPoints)] += result;     
        }
        predPoints[((i+inputVecSize)%numPredPoints)] = 0;
    }

    for (int i = numPredPoints-1; i < points; i++) {

        inVec.clear();
        for (int j = 0; j < inputVecSize; j++) {
            inVec.push_back(timeSeries2.at(i+j));
        }

        inVec = dataproc->process(inVec,0);
        input[0] = inVec;
        for (int j = 0; j < numPredPoints; j++) {          
            result = lstm->predict(input); 
            input[0] = std::vector<double>(inVec.begin()+1, inVec.begin()+inputVecSize);
            input[0].push_back(result);
            predPoints[((i+inputVecSize+j)%numPredPoints)] += result;     
        }

        result = predPoints[((i+inputVecSize)%numPredPoints)]/(double)numPredPoints;
        predPoints[((i+inputVecSize)%numPredPoints)] = 0;

        // calculating the Mean Squared Error
        expected = timeSeries.at(i+inputVecSize);
        MSE += std::pow(expected-result,2);
        result = dataproc->postProcess(result);
        out_file<<result<<"\n";
        out_file2<<timeSeries2.at(i+inputVecSize)<<"\n";

    }

    out_file.close();
    out_file2.close();

    MSE /= points;
    std::cout<<"Mean Squared Error: "<<MSE<<"\n";
    
    return 0;
}
