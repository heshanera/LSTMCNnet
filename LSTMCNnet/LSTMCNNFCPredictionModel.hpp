/* 
 * File:   LSTMCNNFCPredictionModel.hpp
 * Author: heshan
 *
 * Created on August 7, 2018, 10:10 PM
 */

#ifndef LSTMCNNFCPREDICTIONMODEL_HPP
#define LSTMCNNFCPREDICTIONMODEL_HPP

#include <iostream>
#include "LSTMnet/LSTMNet.h"
#include "CNNet/CNN.hpp"
#include "LSTMnet/DataProcessor.h"
#include "LSTMnet/FileProcessor.h"
#include "PredictionModel.hpp"

class LSTMCNNFCPredictionModel {
public:
    LSTMCNNFCPredictionModel();
    LSTMCNNFCPredictionModel(ModelStruct * modelStruct);
    LSTMCNNFCPredictionModel(const LSTMCNNFCPredictionModel& orig);
    virtual ~LSTMCNNFCPredictionModel();
    
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

#endif /* LSTMCNNFCPREDICTIONMODEL_HPP */
