/* 
 * File:   PredictionModel.hpp
 * Author: heshan
 *
 * Created on August 4, 2018, 10:59 AM
 */

#ifndef PREDICTIONMODEL_HPP
#define PREDICTIONMODEL_HPP

#include <iostream>
#include "CNNet/CNN.hpp"
#include "LSTMnet/LSTMNet.h"
#include "LSTMnet/DataProcessor.h"
#include "LSTMnet/FileProcessor.h"
#include "DTW.hpp"

//class ModelStruct;

class PredictionModel {
public:
    PredictionModel();
    PredictionModel(const PredictionModel& orig);
    virtual ~PredictionModel();
private:
};

class ModelStruct {
public:
    virtual ~ModelStruct();
public:
    int trainingIterations; // training iterations with training data
    int trainDataSize; // train data size
    double learningRate; // learning rate
    // LSTM
    int memCells; // number of memory cells
    int inputVecSize; // input vector size
    int predictions; // prediction points
    int numPredPoints; // future points
    std::string dataFile; // path to the data file
    // CNN
    int matWidth;
    int matHeight;
    int targetC;
    struct::NetStruct netStruct;
};


#endif /* PREDICTIONMODEL_HPP */

