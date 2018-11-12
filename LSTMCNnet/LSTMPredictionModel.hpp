/* 
 * File:   LSTMPredictionModel.hpp
 * Author: heshan
 *
 * Created on August 7, 2018, 5:12 PM
 */

#ifndef LSTMPREDICTIONMODEL_HPP
#define LSTMPREDICTIONMODEL_HPP

#include "PredictionModel.hpp"

class LSTMPredictionModel {
public:
    LSTMPredictionModel();
    
    /**
     * Initialize the LSTM model
     * @param modelStruct: parameters for model
     */
    LSTMPredictionModel(ModelStruct * modelStruct);
    LSTMPredictionModel(const LSTMPredictionModel& orig);
    virtual ~LSTMPredictionModel();
    
    /**
     * Initialize the LSTMCNNFC model
     * @param modelStruct: parameters for model
     */
    int train();
    
    /**
     * Input Data for the predictions
     * 
     * @param file: input data ( time series )
     * @return 0
     */
    int initPredData(std::string file);
    
    /**
     * Predict the given number of points and write the predicted values to given file
     * 
     * @param points: number of prediction points
     * @param expect: file path to write the expected values
     * @param predict: file path to write the predicted value
     * @return 0
     */
    int predict(int points, std::string expect, std::string predict);
    
    /**
     * Predict the given number of points, Identify the anomalies using DTW and write the anomalous points to given file
     * 
     * @param points: points to be predicted
     * @param expect: file path to write the expected values
     * @param predict: file path to write the predicted anomaly points
     * @param simVecSize: DTW similarity vector size 
     * @param marker: value to write for the anomalous point
     * @param simMargin: DTW similarity margin to detect anomalous points
     * @return 0
     */
    int predict(int points, std::string expect, std::string predict, int simVecSize, double marker, double simMargin = 0);
private:
    LSTMNet * lstm;
    ModelStruct * modelStruct;
    FileProcessor * fileProc;
    DataProcessor * dataproc;
    std::vector<double> timeSeries;
    std::vector<double> timeSeries2;
    
};

#endif /* LSTMPREDICTIONMODEL_HPP */

