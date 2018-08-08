/* 
 * File:   LSTMCNNPredictionModel.hpp
 * Author: heshan
 *
 * Created on August 8, 2018, 11:43 PM
 */

#ifndef LSTMCNNPREDICTIONMODEL_HPP
#define LSTMCNNPREDICTIONMODEL_HPP

#include <iostream>
#include "LSTMnet/LSTMNet.h"
#include "CNNet/CNN.hpp"
#include "LSTMnet/DataProcessor.h"
#include "LSTMnet/FileProcessor.h"
#include "PredictionModel.hpp"

class LSTMCNNPredictionModel {
public:
    LSTMCNNPredictionModel();
    LSTMCNNPredictionModel(ModelStruct * modelStruct);
    LSTMCNNPredictionModel(const LSTMCNNPredictionModel& orig);
    virtual ~LSTMCNNPredictionModel();
    
    int train();
    int predict(int points, std::string expect, std::string predict);
private:
    LSTMNet * lstm;
    CNN * cnn;
    ModelStruct * modelStruct;
    FileProcessor * fileProc;
    DataProcessor * dataproc;
    std::vector<double> timeSeries;
    std::vector<double> timeSeries2;

};

#endif /* LSTMCNNPREDICTIONMODEL_HPP */




