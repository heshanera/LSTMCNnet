/* 
 * File:   LSTMCNNFCPredictionModel.hpp
 * Author: heshan
 *
 * Created on August 7, 2018, 10:10 PM
 */

#ifndef LSTMCNNFCPREDICTIONMODEL_HPP
#define LSTMCNNFCPREDICTIONMODEL_HPP

#include "PredictionModel.hpp"

class LSTMCNNFCPredictionModel {
public:
    LSTMCNNFCPredictionModel();
    LSTMCNNFCPredictionModel(ModelStruct * modelStruct);
    LSTMCNNFCPredictionModel(const LSTMCNNFCPredictionModel& orig);
    virtual ~LSTMCNNFCPredictionModel();
    
    /**
     * Train the prediction model
     * 
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
     * @param points: points to be predicted
     * @param expect: file path to write the expected values
     * @param predict: file path to write the predicted value
     * @param lstmW: prediction weight for the lstm (default value = 0.5, lstmW + cnnW = 1)
     * @param cnnW: prediction weight for the cnn (default value = 0.5, lstmW + cnnW = 1)
     * @return 0
     */
    int predict(int points, std::string expect, std::string predict, double lstmW = 0.5, double cnnW = 0.5);
    
    /**
     * Predict the given number of points, Identify the anomalies using DTW and write the anomalous points to given file
     * 
     * @param points: points to be predicted
     * @param expect: file path to write the expected values
     * @param predict: file path to write the predicted anomaly points
     * @param simVecSize: DTW similarity vector size 
     * @param marker: value to write for the anomalous point
     * @param simMargin: DTW similarity margin to detect anomalous points
     * @param lstmW: prediction weight for the lstm (default value = 0.5, lstmW + cnnW = 1)
     * @param cnnW: prediction weight for the cnn (default value = 0.5, lstmW + cnnW = 1)
     * @return 0
     */
    int predict(
        int points, std::string expect, std::string predict, 
        int simVecSize, double marker, double simMargin = 0, 
        double lstmW = 0.5, double cnnW = 0.5
    );
    
    /**
     * Predict the given number of points and write the predicted values to given file
     * keep track on the training data points to predict the normal behavior
     * 
     * @param points: points to be predicted
     * @param expect: file path to write the expected values
     * @param predict: file path to write the predicted value
     * @param lstmW: prediction weight for the lstm (default value = 0.5, lstmW + cnnW = 1)
     * @param cnnW: prediction weight for the cnn (default value = 0.5, lstmW + cnnW = 1)
     * @return 0
     */
    int predictNorm(int points, std::string expect, std::string predict, double lstmW = 0.5, double cnnW = 0.5);
    
    /**
     * Predict the given number of points, Identify the anomalies using DTW and write the anomalous points to given file
     * keep track on the training data points to predict the normal behavior
     * 
     * @param points: points to be predicted
     * @param expect: file path to write the expected values
     * @param predict: file path to write the predicted anomaly points
     * @param simVecSize: DTW similarity vector size 
     * @param marker: value to write for the anomalous point
     * @param simMargin: DTW similarity margin to detect anomalous points
     * @param lstmW: prediction weight for the lstm (default value = 0.5, lstmW + cnnW = 1)
     * @param cnnW: prediction weight for the cnn (default value = 0.5, lstmW + cnnW = 1)
     * @return 0
     */
    int predictNorm(
        int points, std::string expect, std::string predict, 
        int simVecSize, double marker, double simMargin = 0, 
        double lstmW = 0.5, double cnnW = 0.5
    );
    
    /**
     * Predict the given number of points and write the predicted values to given file
     * keep track on the training data points to predict the normal behavior
     * adapt for the changes in the normal behavior
     * 
     * @param points: points to be predicted
     * @param expect: file path to write the expected values
     * @param predict: file path to write the predicted value
     * @param timeLimit: time limit to update the normal behavior
     * @param lstmW: prediction weight for the lstm (default value = 0.5, lstmW + cnnW = 1)
     * @param cnnW: prediction weight for the cnn (default value = 0.5, lstmW + cnnW = 1)
     * @return 0
     */
    int predictAdaptNorm(
        int points, 
        std::string expect, 
        std::string predict, 
        int timeLimit,
        double lstmW = 0.5, 
        double cnnW = 0.5
    );
    
    /**
     * Predict the given number of points, Identify the anomalies using DTW and write the anomalous points to given file
     * keep track on the training data points to predict the normal behavior
     * adapt for the changes in the normal behavior
     * 
     * @param points: points to be predicted
     * @param expect: file path to write the expected values
     * @param predict: file path to write the predicted anomaly points
     * @param timeLimit: time limit to update the normal behavior
     * @param simVecSize: DTW similarity vector size 
     * @param marker: value to write for the anomalous point
     * @param simMargin: DTW similarity margin to detect anomalous points
     * @param lstmW: prediction weight for the lstm (default value = 0.5, lstmW + cnnW = 1)
     * @param cnnW: prediction weight for the cnn (default value = 0.5, lstmW + cnnW = 1)
     * @return 0
     */
    int predictAdaptNorm(
        int points, std::string expect, std::string predict, 
        int timeLimit, int simVecSize, double marker, 
        double simMargin = 0, double lstmW = 0.5, double cnnW = 0.5
    );
    
    /**
     * 
     * @param points: points to be predicted
     * @param expect: file path to write the expected values
     * @param predict: file path to write the predicted anomaly points
     * @param simVecSize: DTW similarity vector size 
     * @param lstmW: prediction weight for the lstm (default value = 0.5, lstmW + cnnW = 1)
     * @param cnnW: prediction weight for the cnn (default value = 0.5, lstmW + cnnW = 1)
     * @return 0
     */
    int dtwSimilarity(
        int points, std::string expect, std::string predict, 
        int simVecSize, double lstmW = 0.5, double cnnW = 0.5
    );
    
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

