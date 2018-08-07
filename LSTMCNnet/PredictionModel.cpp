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
        } case PredictionModel::DNN: {
        
            // Generating a convolutional network
            int width = modelStruct->matWidth;
            int height = modelStruct->matHeight;
            int iterations = modelStruct->trainingIterations;
            int inputSize = modelStruct->trainDataSize;
            int targetsC = modelStruct->targetC;
            double learningRate = modelStruct->learningRate;

            // network structure
            std::tuple<int, int, int> dimensions = std::make_tuple(1,height,width);
            
            // Generating input matrices
            Eigen::MatrixXd ** inMatArr;
            Eigen::MatrixXd * inLblArr;
            Eigen::MatrixXd inMat;
            Eigen::MatrixXd inLbl;
            inMatArr = new Eigen::MatrixXd * [inputSize];
            inLblArr = new Eigen::MatrixXd[inputSize];

            dataproc = new DataProcessor();
            fileProc = new FileProcessor();
            
            // Reading the file
            timeSeries2 = fileProc->read(modelStruct->dataFile,1);
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
            this->cnn = new CNN(dimensions, modelStruct->netStruct);
            // Training the network
            cnn->train(inMatArr, inLblArr, inputSize, iterations, learningRate);
        
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
            
        } case PredictionModel::DNN: {
            
            int width = modelStruct->matWidth;
            int height = modelStruct->matHeight;
            int inputSize = modelStruct->trainDataSize;
            
            // Predictions
            Eigen::MatrixXd prediction;
            // Open the file to write the time series predictions
            std::ofstream out_file;
            out_file.open(predict,std::ofstream::out | std::ofstream::trunc);
            std::ofstream out_file2;
            out_file2.open(expect,std::ofstream::out | std::ofstream::trunc);

            Eigen::MatrixXd tstMatArr[1];
            double errorSq = 0, MSE;
            double expected;
            double val;
            int predSize = points;

            std::vector<double> inVec;
            int inputVecSize = height*width;

            int numPredPoints = 3;
            double predPoints[numPredPoints];

            for (int j = 0; j < numPredPoints; j++) {
                predPoints[j] = 0;
            }
            
            std::cout << std::fixed;

            // max and min training values
            double trainMax = *std::max_element(timeSeries.begin(), timeSeries.begin()+(inputSize+(width*height)));
            double trainMin = *std::min_element(timeSeries.begin(), timeSeries.begin()+(inputSize+(width*height)));
            // max and min predicted values
            double predictMax = std::numeric_limits<double>::min();
            double predictMin = std::numeric_limits<double>::max();

            for (int i = 0; i < inputSize; i++) {
                inVec.clear();
                for (int j = 0; j < inputVecSize; j++) {
                    inVec.push_back(timeSeries2.at(i+j));
                }
                inVec = dataproc->process(inVec,0);

                tstMatArr[0] = Eigen::MatrixXd::Zero(height,width);
                for (int a = 0; a < height; a++) {
                    for (int b = 0; b < width; b++) {
                        tstMatArr[0](a,b) = inVec.at(( a * width ) + b);
                    }
                }

                for (int j = 0; j < numPredPoints; j++) {      
                    prediction = cnn->predict(tstMatArr);
                    inVec = std::vector<double>(inVec.begin(), inVec.begin()+inputVecSize-1);
                    inVec.push_back(prediction(0,0));
                    for (int a = 0; a < height; a++) {
                        for (int b = 0; b < width; b++) {
                            tstMatArr[0](a,b) = inVec.at(( a * width ) + b);
                        }
                    }
                    predPoints[((i+inputVecSize+j)%numPredPoints)] += prediction(0,0);     
                }

                if (i >= numPredPoints-1) {
                    prediction(0,0) = predPoints[((i+inputVecSize)%numPredPoints)]/(double)numPredPoints;
                    if (prediction(0,0) > predictMax) predictMax = prediction(0,0);
                    if (prediction(0,0) < predictMin) predictMin = prediction(0,0);
                }
                predPoints[((i+inputVecSize)%numPredPoints)] = 0;
            }

            for (int i = inputSize; i < predSize; i++) {

                inVec.clear();
                for (int j = 0; j < inputVecSize; j++) {
                    inVec.push_back(timeSeries2.at(i+j));
                }
                inVec = dataproc->process(inVec,0);
            //        trainMax = *std::max_element(inVec.begin(), inVec.end());
            //        trainMin = *std::min_element(inVec.begin(), inVec.end());

                tstMatArr[0] = Eigen::MatrixXd::Zero(height,width);
                for (int a = 0; a < height; a++) {
                    for (int b = 0; b < width; b++) {
                        tstMatArr[0](a,b) = inVec.at(( a * width ) + b);
                    }
                }

            //        prediction = cn.predict(tstMatArr);
                for (int j = 0; j < numPredPoints; j++) {      
                    prediction = cnn->predict(tstMatArr);
                    inVec = std::vector<double>(inVec.begin()+1, inVec.begin()+inputVecSize);
                    inVec.push_back(prediction(0,0));
                    for (int a = 0; a < height; a++) {
                        for (int b = 0; b < width; b++) {
                            tstMatArr[0](a,b) = inVec.at(( a * width ) + b);
                        }
                    }
                    predPoints[((i+inputVecSize+j)%numPredPoints)] += prediction(0,0);     
                }

                prediction(0,0) = predPoints[((i+inputVecSize)%numPredPoints)]/(double)numPredPoints;
                predPoints[((i+inputVecSize)%numPredPoints)] = 0;

                //std::cout<<prediction(0,0)<<"\n"; 
                expected = timeSeries.at(i + (width*height));
                val = prediction(0,0);
                errorSq += pow(val - expected,2);

                // post process
                val = (val - predictMin)*((trainMax - trainMin)/(predictMax - predictMin)) + trainMin;

                out_file<<dataproc->postProcess(val)<<"\n";
                out_file2<<timeSeries2.at(i+inputVecSize)<<"\n";
            }
            MSE = errorSq/(predSize-inputSize);
            std::cout<<"\nMean Squared Error: "<<MSE<<"\n\n"; 
            std::cout << std::scientific;
            break;
        }
    }
    
    
    return 0;
}

PredictionModel::~PredictionModel() { }

