/* 
 * File:   CNNPredictionModel.hpp
 * Author: heshan
 *
 * Created on August 7, 2018, 7:53 PM
 */

#ifndef CNNPREDICTIONMODEL_HPP
#define CNNPREDICTIONMODEL_HPP

#include <iostream>
#include "CNNet/CNN.hpp"
#include "LSTMnet/DataProcessor.h"
#include "LSTMnet/FileProcessor.h"
#include "PredictionModel.hpp"

class CNNPredictionModel {
public:
    CNNPredictionModel();
    CNNPredictionModel(ModelStruct * modelStruct);
    CNNPredictionModel(const CNNPredictionModel& orig);
    virtual ~CNNPredictionModel();
    
    int train();
    int initPredData(std::string file);
    int predict(int points, std::string expect, std::string predict);
private:
    CNN * cnn;
    ModelStruct * modelStruct;
    FileProcessor * fileProc;
    DataProcessor * dataproc;
    std::vector<double> timeSeries;
    std::vector<double> timeSeries2;

};

#endif /* CNNPREDICTIONMODEL_HPP */

