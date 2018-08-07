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

class ModelStruct;

class PredictionModel {
public:
    enum Model { LSTM, DNN, LSTMDNN, DNNLSTM, LSTMDNNFC}; 
public:
    PredictionModel(ModelStruct * ModelStruct);
    virtual ~PredictionModel();
    int train();
    int predict(int points, std::string expect, std::string predict);
private:
    FileProcessor * fileProc;
    DataProcessor * dataproc;
    ModelStruct * modelStruct;
    std::vector<double> timeSeries;
    std::vector<double> timeSeries2;
private:
    int trainLSTM();
    int trainCNN();
    LSTMNet * lstm;
    CNN * cnn;

};

class ModelStruct {
public:
    PredictionModel::Model model; // model type
    int trainingIterations; // training iterations with training data
    int trainDataSize; // train data size
    float learningRate; // learning rate
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

