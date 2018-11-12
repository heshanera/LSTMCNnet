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
    
    /**
     * Initialize the CNNLSTM model
     * @param modelStruct: parameters for model
     */
    CNNLSTMPredictionModel(ModelStruct * modelStruct);
    CNNLSTMPredictionModel(const CNNLSTMPredictionModel& orig);
    virtual ~CNNLSTMPredictionModel();
    
    /**
     * Train the prediction model
     * @return 0
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

#endif /* CNNLSTMPREDICTIONMODEL_HPP */

