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
     * @return 
     */
    int train();
    /**
     * 
     * @param file: input data ( time series )
     * @return 
     */
    int initPredData(std::string file);
    /**
     * 
     * @param points
     * @param expect
     * @param predict
     * @return 
     */
    int predict(int points, std::string expect, std::string predict);
    /**
     * 
     * @param points
     * @param expect
     * @param predict
     * @param simVecSize
     * @param marker
     * @param simMargin
     * @return 
     */
    int predict(int points, std::string expect, std::string predict, int simVecSize, double marker, double simMargin = 0);
    /**
     * 
     * @param points
     * @param expect
     * @param predict
     * @return 
     */
    int predictNorm(int points, std::string expect, std::string predict);
    /**
     * 
     * @param points
     * @param expect
     * @param predict
     * @param simVecSize
     * @param marker
     * @param simMargin
     * @return 
     */
    int predictNorm(int points, std::string expect, std::string predict, int simVecSize, double marker, double simMargin = 0);
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

