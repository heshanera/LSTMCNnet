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
    
    /**
     * Initialize the LSTMCNN model
     * @param modelStruct: parameters for model
     */
    LSTMCNNPredictionModel(ModelStruct * modelStruct);
    LSTMCNNPredictionModel(const LSTMCNNPredictionModel& orig);
    virtual ~LSTMCNNPredictionModel();
    
    /**
     * Initialize the LSTMCNNFC model
     * @param modelStruct: parameters for model
     */
    int train();
    
    /**
     * Predict the given number of points and write the predicted values to given file
     * 
     * @param points: number of prediction points
     * @param expect: file path to write the expected values
     * @param predict: file path to write the predicted value
     * @return 0
     */
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




