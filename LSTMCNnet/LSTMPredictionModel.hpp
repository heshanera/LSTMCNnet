/* 
 * File:   LSTMPredictionModel.hpp
 * Author: heshan
 *
 * Created on August 7, 2018, 5:12 PM
 */

#ifndef LSTMPREDICTIONMODEL_HPP
#define LSTMPREDICTIONMODEL_HPP

#include <iostream>
#include "LSTMnet/LSTMNet.h"
#include "LSTMnet/DataProcessor.h"
#include "LSTMnet/FileProcessor.h"
#include "PredictionModel.hpp"

class LSTMPredictionModel {
public:
    LSTMPredictionModel();
    LSTMPredictionModel(ModelStruct * modelStruct);
    LSTMPredictionModel(const LSTMPredictionModel& orig);
    virtual ~LSTMPredictionModel();
    
    int train();
    int predict(int points, std::string expect, std::string predict);
private:
    LSTMNet * lstm;
    ModelStruct * modelStruct;
    FileProcessor * fileProc;
    DataProcessor * dataproc;
    std::vector<double> timeSeries;
    std::vector<double> timeSeries2;

};

#endif /* LSTMPREDICTIONMODEL_HPP */
