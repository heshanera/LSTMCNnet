/* 
 * File:   CNNLSTMPredictionModel.hpp
 * Author: heshan
 *
 * Created on August 9, 2018, 3:20 PM
 */

#ifndef CNNLSTMPREDICTIONMODEL_HPP
#define CNNLSTMPREDICTIONMODEL_HPP

#include <iostream>
#include "LSTMnet/LSTMNet.h"
#include "CNNet/CNN.hpp"
#include "LSTMnet/DataProcessor.h"
#include "LSTMnet/FileProcessor.h"
#include "PredictionModel.hpp"

class CNNLSTMPredictionModel {
public:
    CNNLSTMPredictionModel();
    CNNLSTMPredictionModel(ModelStruct * modelStruct);
    CNNLSTMPredictionModel(const CNNLSTMPredictionModel& orig);
    virtual ~CNNLSTMPredictionModel();
    
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

#endif /* CNNLSTMPREDICTIONMODEL_HPP */

