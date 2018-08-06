/* 
 * File:   PredictionModel.cpp
 * Author: heshan
 * 
 * Created on August 4, 2018, 10:59 AM
 */

#include "PredictionModel.hpp"

PredictionModel::PredictionModel(ModelStruct * modelStruct) { 
    this->modelStruct = modelStruct;
}

int PredictionModel::train() {

    switch(modelStruct->model) {
        case PredictionModel::LSTM: {
            
            int memCells = modelStruct->memCells; // number of memory cells
            int trainDataSize = modelStruct->trainDataSize; // train data size
            int inputVecSize = modelStruct->inputVecSize; // input vector size
            int timeSteps = modelStruct->inputVecSize; // unfolded time steps
            float learningRate = modelStruct->learningRate;
            int iterations = modelStruct->trainingIterations; // training iterations with training data

            // Adding the time series in to a vector and preprocessing
            dataproc = new DataProcessor();
            fileProc = new FileProcessor();

            timeSeries2 = fileProc->read(modelStruct->dataFile,1);
            timeSeries =  dataproc->process(timeSeries2,1);

            // Creating the input vector Array
            std::vector<double> * input;
            input = new std::vector<double>[trainDataSize];
            std::vector<double> inputVec;

            for (int i = 0; i < trainDataSize; i++) {
                inputVec.clear();
                for (int j = 0; j < inputVecSize; j++) {
                    inputVec.push_back(timeSeries.at(i+j));
                }
                inputVec =  dataproc->process(inputVec,0);
                input[i] = inputVec; 
            }


            // Creating the target vector using the time series 
            std::vector<double>::const_iterator first = timeSeries.begin() + inputVecSize;
            std::vector<double>::const_iterator last = timeSeries.begin() + inputVecSize + trainDataSize;
            std::vector<double> targetVector(first, last);

            // Training the LSTM net
            this->lstm = new LSTMNet(memCells,inputVecSize);
            lstm->train(input, targetVector, trainDataSize, timeSteps, learningRate, iterations);
            
            break;
        } case PredictionModel::CNN: {
        
            // Generating a convolutional network
            int width = modelStruct->matWidth;
            int height = modelStruct->matHeight;
            int iterations = modelStruct->trainingIterations;
            int inputSize = modelStruct->trainDataSize;
            int targetsC = modelStruct->targetC;
            double learningRate = modelStruct->learningRate;

            std::string inFile = modelStruct->dataFile;

            // network structure
            std::tuple<int, int, int> dimensions = std::make_tuple(1,height,width);
            
            // Generating input matrices
            Eigen::MatrixXd ** inMatArr;
            Eigen::MatrixXd * inLblArr;
            Eigen::MatrixXd inMat;
            Eigen::MatrixXd inLbl;
            inMatArr = new Eigen::MatrixXd * [inputSize];
            inLblArr = new Eigen::MatrixXd[inputSize];

            // Reading the file
            std::vector<double> timeSeries;
            std::vector<double> timeSeries2;
            timeSeries2 = fileProc->read("datasets/univariate/input/"+inFile,1);
            timeSeries =  dataproc->process(timeSeries2,1);

            for (int i = 0; i < inputSize; i++) {
                // inputs
                inMatArr[i] = new Eigen::MatrixXd[1]; // image depth
                inMat = Eigen::MatrixXd(height,width);
                for (int a = 0; a < height; a++) {
                    for (int b = 0; b < width; b++) {
                        inMat(a,b) = timeSeries.at(i + ( a * width ) + b);
                    }
                }
                inMatArr[i][0] = inMat;
                // labels
                inLbl = Eigen::MatrixXd::Zero(targetsC,1);
                for (int a = 0; a < targetsC; a++) {
        //            inLbl(a,0) = timeSeries.at(((i + 1) * width) + a);
                    inLbl(a,0) = timeSeries.at(i + (width*height));
                }
                inLblArr[i] = inLbl;
            }

            // Generating the network
            struct::NetStruct netStruct;
            CNN cn(dimensions, netStruct);
            // Training the network
//            cn.train(inMatArr, inLblArr, inputSize, iterations, learningRate);
        
            break;
        }
    }

    return 0;
}

int PredictionModel::predict(int points, std::string expect, std::string predict) {

    switch(modelStruct->model) {
        case PredictionModel::LSTM: {
            
            // Open the file to write the time series predictions
            std::ofstream out_file;
            out_file.open(predict,std::ofstream::out | std::ofstream::trunc);
            std::ofstream out_file2;
            out_file2.open(expect,std::ofstream::out | std::ofstream::trunc);

            std::vector<double> * input;
            std::vector<double> inVec;
            input = new std::vector<double>[1];
            double result;
            double expected;
            double MSE = 0;
            
            inVec.clear();
            for (int i =0; i <60; i+=5){
                inVec.push_back(0.2);
                inVec.push_back(0.4);
                inVec.push_back(0.6);
                inVec.push_back(0.8);
                inVec.push_back(0.3);
            }
            input = new std::vector<double>[1];
            input[0] = inVec;
            
            int numPredPoints = modelStruct->numPredPoints;
            double predPoints[numPredPoints];

            for (int j = 0; j < numPredPoints; j++) {
                predPoints[j] = 0;
            }

            std::cout << std::fixed;

            for (int i = 0; i < numPredPoints-1; i++) {
                inVec.clear();
                for (int j = 0; j < modelStruct->inputVecSize; j++) {
                    inVec.push_back(timeSeries2.at(i+j));
                }

                inVec = dataproc->process(inVec,0);
                input[0] = inVec;
                for (int j = 0; j < numPredPoints; j++) {          
                    result = lstm->predict(input); 
                    input[0] = std::vector<double>(inVec.begin(), inVec.begin()+modelStruct->inputVecSize-2);
                    input[0].push_back(result);
                    predPoints[((i+modelStruct->inputVecSize+j)%numPredPoints)] += result;     
                }
                predPoints[((i+modelStruct->inputVecSize)%numPredPoints)] = 0;
            }

            for (int i = numPredPoints-1; i < points; i++) {

                inVec.clear();
                for (int j = 0; j < modelStruct->inputVecSize; j++) {
                    inVec.push_back(timeSeries2.at(i+j));
                }

                inVec = dataproc->process(inVec,0);
                input[0] = inVec;
                for (int j = 0; j < numPredPoints; j++) {          
                    result = lstm->predict(input); 
                    input[0] = std::vector<double>(inVec.begin()+1, inVec.begin()+modelStruct->inputVecSize);
                    input[0].push_back(result);
                    predPoints[((i+modelStruct->inputVecSize+j)%numPredPoints)] += result;     
                }

                result = predPoints[((i+modelStruct->inputVecSize)%numPredPoints)]/(double)numPredPoints;
                predPoints[((i+modelStruct->inputVecSize)%numPredPoints)] = 0;

                // calculating the Mean Squared Error
                expected = timeSeries.at(i+modelStruct->inputVecSize+1);
                MSE += std::pow(expected-result,2);
                result = dataproc->postProcess(result);
                out_file<<result<<"\n";
                out_file2<<timeSeries2.at(i+modelStruct->inputVecSize)<<"\n";

            }
            
            out_file.close();
            out_file2.close();

            MSE /= points;
            std::cout<<"Mean Squared Error: "<<MSE<<"\n";
            std::cout << std::scientific;
        
            break;
        }
    }
    
    
    return 0;
}

PredictionModel::~PredictionModel() { }

