/* 
 * File:   CNNPredictionModel.hpp
 * Author: heshan
 *
 * Created on August 7, 2018, 7:53 PM
 */

#ifndef CNNPREDICTIONMODEL_HPP
#define CNNPREDICTIONMODEL_HPP

#include "PredictionModel.hpp"

class CNNPredictionModel {
public:
    CNNPredictionModel();
    
    /**
     * Initialize the CNN model
     * @param modelStruct: parameters for model
     */
    CNNPredictionModel(ModelStruct * modelStruct);
    CNNPredictionModel(const CNNPredictionModel& orig);
    virtual ~CNNPredictionModel();
    
    /**
     * Train the prediction model
     * @return 0
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
    CNN * cnn;
    ModelStruct * modelStruct;
    FileProcessor * fileProc;
    DataProcessor * dataproc;
    std::vector<double> timeSeries;
    std::vector<double> timeSeries2;

};

#endif /* CNNPREDICTIONMODEL_HPP */

