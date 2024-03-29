/* 
 * File:   LSTMCNNFCPredictionModel.cpp
 * Author: heshan
 * 
 * Created on August 7, 2018, 10:10 PM
 */

#include "LSTMCNNFCPredictionModel.hpp"

LSTMCNNFCPredictionModel::LSTMCNNFCPredictionModel() { }

LSTMCNNFCPredictionModel::LSTMCNNFCPredictionModel(ModelStruct * modelStruct) {
    this->modelStruct = modelStruct;
}

LSTMCNNFCPredictionModel::LSTMCNNFCPredictionModel(const LSTMCNNFCPredictionModel& orig) { }

LSTMCNNFCPredictionModel::~LSTMCNNFCPredictionModel() { }

int LSTMCNNFCPredictionModel::train() {

    int trainDataSize = modelStruct->trainDataSize; // train data size
    double learningRate = modelStruct->learningRate;
    int iterations = modelStruct->trainingIterations; // training iterations with training data
    
    // CNN parameters
    int height = modelStruct->matHeight;
    int width = modelStruct->matWidth;
    int targetsC = modelStruct->targetC;
    
    // LSTM parameters
    int memCells = modelStruct->memCells; // number of memory cells
    int inputVecSize = height*width; // input vector size
    int timeSteps = inputVecSize; // unfolded time steps

    // Adding the time series in to a vector and preprocessing
    dataproc = new DataProcessor();
    fileProc = new FileProcessor();

    timeSeries2 = fileProc->read(modelStruct->dataFile,1);
    timeSeries =  dataproc->process(timeSeries2,1);

    // Creating the input vector Array for LSTM
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


    // Creating the target vector for LSTM
    std::vector<double>::const_iterator first = timeSeries.begin() + inputVecSize;
    std::vector<double>::const_iterator last = timeSeries.begin() + inputVecSize + trainDataSize;
    std::vector<double> targetVector(first, last);

    // Training the LSTM net
    this->lstm = new LSTMNet(memCells,inputVecSize);
    lstm->train(input, targetVector, trainDataSize, timeSteps, learningRate, iterations);
    

    // CNN network structure
    std::tuple<int, int, int> dimensions = std::make_tuple(1,height,width);

    // Generating input matrices for CNN
    Eigen::MatrixXd ** inMatArr;
    Eigen::MatrixXd * inLblArr;
    Eigen::MatrixXd inMat;
    Eigen::MatrixXd inLbl;
    inMatArr = new Eigen::MatrixXd * [trainDataSize];
    inLblArr = new Eigen::MatrixXd[trainDataSize];

    for (int i = 0; i < trainDataSize; i++) {
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
            inLbl(a,0) = timeSeries.at(i + (width*height));
        }
        inLblArr[i] = inLbl;
    }

    // Generating the CNN
    this->cnn = new CNN(dimensions, modelStruct->netStruct);
    // Training the network
    cnn->train(inMatArr, inLblArr, trainDataSize, iterations, learningRate);

    return 0;
}

int LSTMCNNFCPredictionModel::trainFromFile() {

    int trainDataSize = modelStruct->trainDataSize; // train data size
    double learningRate = modelStruct->learningRate;
    int iterations = modelStruct->trainingIterations; // training iterations with training data
    
    // CNN parameters
    int height = modelStruct->matHeight;
    int width = modelStruct->matWidth;
    int targetsC = modelStruct->targetC;
    
    // LSTM parameters
    int memCells = modelStruct->memCells; // number of memory cells
    int inputVecSize = height*width; // input vector size
    int timeSteps = inputVecSize; // unfolded time steps

    // Adding the time series in to a vector and preprocessing
    dataproc = new DataProcessor();
    fileProc = new FileProcessor();

    timeSeries2 = fileProc->read(modelStruct->dataFile,1,(trainDataSize+inputVecSize));
    timeSeries =  dataproc->process(timeSeries2,1);

    // Creating the input vector Array for LSTM
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


    // Creating the target vector for LSTM
    std::vector<double>::const_iterator first = timeSeries.begin() + inputVecSize;
    std::vector<double>::const_iterator last = timeSeries.begin() + inputVecSize + trainDataSize;
    std::vector<double> targetVector(first, last);

    // Training the LSTM net
    this->lstm = new LSTMNet(memCells,inputVecSize);
    lstm->train(input, targetVector, trainDataSize, timeSteps, learningRate, iterations);
    

    // CNN network structure
    std::tuple<int, int, int> dimensions = std::make_tuple(1,height,width);

    // Generating input matrices for CNN
    Eigen::MatrixXd ** inMatArr;
    Eigen::MatrixXd * inLblArr;
    Eigen::MatrixXd inMat;
    Eigen::MatrixXd inLbl;
    inMatArr = new Eigen::MatrixXd * [trainDataSize];
    inLblArr = new Eigen::MatrixXd[trainDataSize];

    for (int i = 0; i < trainDataSize; i++) {
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
            inLbl(a,0) = timeSeries.at(i + (width*height));
        }
        inLblArr[i] = inLbl;
    }

    // Generating the CNN
    this->cnn = new CNN(dimensions, modelStruct->netStruct);
    // Training the network
    cnn->train(inMatArr, inLblArr, trainDataSize, iterations, learningRate);

    return 0;
    
}

int LSTMCNNFCPredictionModel::initPredData(std::string file) {

    std::vector<double> predDatatimeSeries;
    predDatatimeSeries = fileProc->read(file,1);
    int inputVecSize = modelStruct->matHeight*modelStruct->matWidth;
    timeSeries2 = std::vector<double>(
            timeSeries2.begin(), 
            timeSeries2.begin() + modelStruct->trainDataSize + inputVecSize
    );
    timeSeries2.insert( timeSeries2.end(), predDatatimeSeries.begin(), predDatatimeSeries.end() );
    return 0;
}

int LSTMCNNFCPredictionModel::predict(int points, std::string expect, std::string predict, double lstmW, double cnnW, int abs) {
    
    double errorSq = 0, MSE, expected, val;
    int predSize = points;
    
    // LSTM parameters
    double result;
    
    // CNN parameters
    int height = modelStruct->matHeight;
    int width = modelStruct->matWidth;
    Eigen::MatrixXd tstMatArr[1];
    
    int inputVecSize = height*width; // input vector size
    int inputSize = modelStruct->trainDataSize; 
    int numPredPoints = modelStruct->numPredPoints;

    // Predictions
    Eigen::MatrixXd prediction;
    // Open the file to write the time series predictions
    std::ofstream out_file;
    out_file.open(predict,std::ofstream::out | std::ofstream::trunc);
    std::ofstream out_file2;
    out_file2.open(expect,std::ofstream::out | std::ofstream::trunc);
    
    // CNN Inputs
    std::vector<double> inVec;
    
    // LSTM Inputs
    std::vector<double> * input;
    input = new std::vector<double>[1];
    
    double predPoints[numPredPoints];
    double lstmPredPoints[numPredPoints];

    for (int j = 0; j < numPredPoints; j++) {
        predPoints[j] = 0;
        lstmPredPoints[j] = 0;
    }
    
    
    // max and min training values [ CNN ]
    double trainMax = *std::max_element(timeSeries.begin(), timeSeries.begin()+(inputSize+(width*height)));
    double trainMin = *std::min_element(timeSeries.begin(), timeSeries.begin()+(inputSize+(width*height)));
    // max and min predicted values [ CNN ]
    double predictMax = std::numeric_limits<double>::min();
    double predictMin = std::numeric_limits<double>::max();

    for (int i = 0; i < inputSize; i++) {
        inVec.clear();
        // filling the input vector using time series data
        for (int j = 0; j < inputVecSize; j++) {
            inVec.push_back(timeSeries2.at(i+j));
        }
        inVec = dataproc->process(inVec,0);
    
        
        // LSTM network predictions for the trained data set
        input[0] = inVec;
        for (int j = 0; j < numPredPoints; j++) {          
            result = lstm->predict(input); 
            input[0] = std::vector<double>(inVec.begin()+1, inVec.begin()+inputVecSize);
            input[0].push_back(result);
            lstmPredPoints[((i+inputVecSize+j)%numPredPoints)] += result;     
        }
        lstmPredPoints[((i+inputVecSize)%numPredPoints)] = 0;

        
        // CNN predictions for the trained data set
        tstMatArr[0] = Eigen::MatrixXd::Zero(height,width);
        for (int a = 0; a < height; a++) {
            for (int b = 0; b < width; b++) {
                tstMatArr[0](a,b) = inVec.at(( a * width ) + b);
            }
        }

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
        
        // Filling the matrix for the CNN input
        tstMatArr[0] = Eigen::MatrixXd::Zero(height,width);
        for (int a = 0; a < height; a++) {
            for (int b = 0; b < width; b++) {
                tstMatArr[0](a,b) = inVec.at(( a * width ) + b);
            }
        }
        
        // LSTM predictions
        input[0] = inVec;
        for (int j = 0; j < numPredPoints; j++) {          
            result = lstm->predict(input); 
            input[0] = std::vector<double>(inVec.begin()+1, inVec.begin()+inputVecSize);
            input[0].push_back(result);
            lstmPredPoints[((i+inputVecSize+j)%numPredPoints)] += result;     
        }
        result = lstmPredPoints[((i+inputVecSize)%numPredPoints)]/(double)numPredPoints;
        lstmPredPoints[((i+inputVecSize)%numPredPoints)] = 0;

        // CNN predictions
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
        
        // post process CNN prediction
        val = prediction(0,0);
        val = (val - predictMin)*((trainMax - trainMin)/(predictMax - predictMin)) + trainMin;
        
        // combining the results LSTM and CNN
//        val = (result + val)/2;
        val = (result*lstmW + val*cnnW);
         
        // calculating the Mean Squared Error
        expected = timeSeries.at(i+inputVecSize);
        // get the absolute value
        if (abs) val = std::abs(val);
        errorSq += std::pow(expected-val,2);
        result = dataproc->postProcess(val);
        
        // writing the predictions
        out_file<<result<<"\n";
        out_file2<<timeSeries2.at(i+inputVecSize)<<"\n";

    }
    
    out_file.close();
    out_file2.close();
    
    MSE = errorSq/(predSize-inputSize);
    std::cout<<"\nMean Squared Error: "<<MSE<<"\n\n"; 
    
    return 0;
}

int LSTMCNNFCPredictionModel::predict(
    int points, std::string expect, std::string predict, 
    int simVecSize, double marker, double simMargin, double lstmW, double cnnW, int abs
) {
    
    Eigen::VectorXd expectedVec = Eigen::VectorXd::Zero(simVecSize);
    Eigen::VectorXd predictedVec = Eigen::VectorXd::Zero(simVecSize);
    int subVSize = simVecSize-1;
    double similarity;
    
    double errorSq = 0, MSE, expected, val;
    int predSize = points;
    
    // LSTM parameters
    double result;
    
    // CNN parameters
    int height = modelStruct->matHeight;
    int width = modelStruct->matWidth;
    Eigen::MatrixXd tstMatArr[1];
    
    int inputVecSize = height*width; // input vector size
    int inputSize = modelStruct->trainDataSize; 
    int numPredPoints = modelStruct->numPredPoints;

    // Predictions
    Eigen::MatrixXd prediction;
    // Open the file to write the time series predictions
    std::ofstream out_file;
    out_file.open(predict,std::ofstream::out | std::ofstream::trunc);
    std::ofstream out_file2;
    out_file2.open(expect,std::ofstream::out | std::ofstream::trunc);
    
    // CNN Inputs
    std::vector<double> inVec;
    
    // LSTM Inputs
    std::vector<double> * input;
    input = new std::vector<double>[1];
    
    double predPoints[numPredPoints];
    double lstmPredPoints[numPredPoints];

    for (int j = 0; j < numPredPoints; j++) {
        predPoints[j] = 0;
        lstmPredPoints[j] = 0;
    }
    
    
    // max and min training values [ CNN ]
    double trainMax = *std::max_element(timeSeries.begin(), timeSeries.begin()+(inputSize+(width*height)));
    double trainMin = *std::min_element(timeSeries.begin(), timeSeries.begin()+(inputSize+(width*height)));
    // max and min predicted values [ CNN ]
    double predictMax = std::numeric_limits<double>::min();
    double predictMin = std::numeric_limits<double>::max();

    for (int i = 0; i < inputSize; i++) {
        inVec.clear();
        // filling the input vector using time series data
        for (int j = 0; j < inputVecSize; j++) {
            inVec.push_back(timeSeries2.at(i+j));
        }
        inVec = dataproc->process(inVec,0);
    
        
        // LSTM network predictions for the trained data set
        input[0] = inVec;
        for (int j = 0; j < numPredPoints; j++) {          
            result = lstm->predict(input); 
            input[0] = std::vector<double>(inVec.begin()+1, inVec.begin()+inputVecSize);
            input[0].push_back(result);
            lstmPredPoints[((i+inputVecSize+j)%numPredPoints)] += result;     
        }
        lstmPredPoints[((i+inputVecSize)%numPredPoints)] = 0;
        
        // CNN predictions for the trained data set
        tstMatArr[0] = Eigen::MatrixXd::Zero(height,width);
        for (int a = 0; a < height; a++) {
            for (int b = 0; b < width; b++) {
                tstMatArr[0](a,b) = inVec.at(( a * width ) + b);
            }
        }

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
        
        // Filling the matrix for the CNN input
        tstMatArr[0] = Eigen::MatrixXd::Zero(height,width);
        for (int a = 0; a < height; a++) {
            for (int b = 0; b < width; b++) {
                tstMatArr[0](a,b) = inVec.at(( a * width ) + b);
            }
        }
        
        // LSTM predictions
        input[0] = inVec;
        for (int j = 0; j < numPredPoints; j++) {          
            result = lstm->predict(input); 
            input[0] = std::vector<double>(inVec.begin()+1, inVec.begin()+inputVecSize);
            input[0].push_back(result);
            lstmPredPoints[((i+inputVecSize+j)%numPredPoints)] += result;     
        }
        result = lstmPredPoints[((i+inputVecSize)%numPredPoints)]/(double)numPredPoints;
        lstmPredPoints[((i+inputVecSize)%numPredPoints)] = 0;

        // CNN predictions
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
        
        // post process CNN prediction
        val = prediction(0,0);
        val = (val - predictMin)*((trainMax - trainMin)/(predictMax - predictMin)) + trainMin;
        
        // combining the results LSTM and CNN
        val = (result*lstmW + val*cnnW);
        
        // get the absolute value
        if (abs) val = std::abs(val);
         
        // calculating the Mean Squared Error
        expected = timeSeries.at(i+inputVecSize);
        errorSq += std::pow(expected-val,2);
        result = dataproc->postProcess(val);
        
        // filling the values to compare the similarity
        for (int j = 0; j < subVSize; j++) {
            expectedVec(j) = expectedVec(j+1);
            predictedVec(j) = predictedVec(j+1);
        }
        predictedVec(subVSize) = result;
        expectedVec(subVSize) = timeSeries2.at(i+inputVecSize);
        
        // Extracting the similarity
        similarity = DTW::getSimilarity(expectedVec,predictedVec);
        
        if (similarity > simMargin) { 
            out_file<<marker<<"\n";
        } else {
            out_file<<"\n";
        }
        out_file2<<timeSeries2.at(i+inputVecSize)<<"\n";

    }
    
    out_file.close();
    out_file2.close();
    
    MSE = errorSq/(predSize-inputSize);
    std::cout<<"\nMean Squared Error: "<<MSE<<"\n\n"; 
    
    return 0;
}

int LSTMCNNFCPredictionModel::predictNorm(int points, std::string expect, std::string predict, double lstmW, double cnnW) {
    
    double errorSq = 0, MSE, expected, val;
    int predSize = points;
    
    // LSTM parameters
    double result;
    
    // CNN parameters
    int height = modelStruct->matHeight;
    int width = modelStruct->matWidth;
    Eigen::MatrixXd tstMatArr[1];
    
    int inputVecSize = height*width; // input vector size
    int inputSize = modelStruct->trainDataSize; 
    int numPredPoints = modelStruct->numPredPoints;

    // Predictions
    Eigen::MatrixXd prediction;
    // Open the file to write the time series predictions
    std::ofstream out_file;
    out_file.open(predict,std::ofstream::out | std::ofstream::trunc);
    std::ofstream out_file2;
    out_file2.open(expect,std::ofstream::out | std::ofstream::trunc);
    
    // CNN Inputs
    std::vector<double> inVec;
    
    // LSTM Inputs
    std::vector<double> * input;
    input = new std::vector<double>[1];
    
    double predPoints[numPredPoints];
    double lstmPredPoints[numPredPoints];

    for (int j = 0; j < numPredPoints; j++) {
        predPoints[j] = 0;
        lstmPredPoints[j] = 0;
    }
    
    
    // max and min training values [ CNN ]
    double trainMax = *std::max_element(timeSeries.begin(), timeSeries.begin()+(inputSize+(width*height)));
    double trainMin = *std::min_element(timeSeries.begin(), timeSeries.begin()+(inputSize+(width*height)));
    // max and min predicted values [ CNN ]
    double predictMax = std::numeric_limits<double>::min();
    double predictMin = std::numeric_limits<double>::max();

    for (int i = 0; i < inputSize; i++) {
        inVec.clear();
        // filling the input vector using time series data
        for (int j = 0; j < inputVecSize; j++) {
            inVec.push_back(timeSeries2.at(i+j));
        }
        inVec = dataproc->process(inVec,0);
    
        
        // LSTM network predictions for the trained data set
        input[0] = inVec;
        for (int j = 0; j < numPredPoints; j++) {          
            result = lstm->predict(input); 
            input[0] = std::vector<double>(inVec.begin()+1, inVec.begin()+inputVecSize);
            input[0].push_back(result);
            lstmPredPoints[((i+inputVecSize+j)%numPredPoints)] += result;     
        }
        lstmPredPoints[((i+inputVecSize)%numPredPoints)] = 0;

        
        // CNN predictions for the trained data set
        tstMatArr[0] = Eigen::MatrixXd::Zero(height,width);
        for (int a = 0; a < height; a++) {
            for (int b = 0; b < width; b++) {
                tstMatArr[0](a,b) = inVec.at(( a * width ) + b);
            }
        }

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

        if (i >= numPredPoints-1) {
            prediction(0,0) = predPoints[((i+inputVecSize)%numPredPoints)]/(double)numPredPoints;
            if (prediction(0,0) > predictMax) predictMax = prediction(0,0);
            if (prediction(0,0) < predictMin) predictMin = prediction(0,0);
        }
        predPoints[((i+inputVecSize)%numPredPoints)] = 0;
    }

    std::vector<double> inVec2;
    for (int j = 0; j < inputVecSize; j++) {
        inVec2.push_back(timeSeries2.at(inputSize+j-1));
    }
    for (int i = inputSize; i < predSize; i++) {

        for (int j = 0; j < inputVecSize-1; j++) {
            inVec2.at(j) = inVec2.at(j+1);
        }
        
        if ( std::abs(timeSeries2.at(i+inputVecSize-1) - result) > (result/1.5) ) {
            inVec2.at(inputVecSize-1) = timeSeries2.at((i+inputVecSize-1)%inputSize);
        }
        else inVec2.at(inputVecSize-1) = timeSeries2.at(i+inputVecSize-1);
        
        inVec = dataproc->process(inVec2,0);
        
        // Filling the matrix for the CNN input
        tstMatArr[0] = Eigen::MatrixXd::Zero(height,width);
        for (int a = 0; a < height; a++) {
            for (int b = 0; b < width; b++) {
                tstMatArr[0](a,b) = inVec.at(( a * width ) + b);
            }
        }
        
        // LSTM predictions
        input[0] = inVec;
        for (int j = 0; j < numPredPoints; j++) {          
            result = lstm->predict(input); 
            input[0] = std::vector<double>(inVec.begin()+1, inVec.begin()+inputVecSize);
            input[0].push_back(result);
            lstmPredPoints[((i+inputVecSize+j)%numPredPoints)] += result;     
        }
        result = lstmPredPoints[((i+inputVecSize)%numPredPoints)]/(double)numPredPoints;
        lstmPredPoints[((i+inputVecSize)%numPredPoints)] = 0;

        // CNN predictions
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
        
        // post process CNN prediction
        val = prediction(0,0);
        val = (val - predictMin)*((trainMax - trainMin)/(predictMax - predictMin)) + trainMin;
        
        // combining the results LSTM and CNN
        val = (result*lstmW + val*cnnW);
         
        // calculating the Mean Squared Error
        expected = timeSeries.at(i+inputVecSize);
        errorSq += std::pow(expected-val,2);
        result = dataproc->postProcess(val);
        
        // writing the predictions
        out_file<<result<<"\n";
        out_file2<<timeSeries2.at(i+inputVecSize)<<"\n";

    }
    
    out_file.close();
    out_file2.close();
    
    MSE = errorSq/(predSize-inputSize);
    std::cout<<"\nMean Squared Error: "<<MSE<<"\n\n"; 
    
    return 0;
}

int LSTMCNNFCPredictionModel::predictNorm(
    int points, std::string expect, std::string predict, 
    int simVecSize, double marker, double simMargin, double lstmW, double cnnW
) {
    
    Eigen::VectorXd expectedVec = Eigen::VectorXd::Zero(simVecSize);
    Eigen::VectorXd predictedVec = Eigen::VectorXd::Zero(simVecSize);
    int subVSize = simVecSize-1;
    double similarity;
    
    double errorSq = 0, MSE, expected, val;
    int predSize = points;
    
    // LSTM parameters
    double result;
    
    // CNN parameters
    int height = modelStruct->matHeight;
    int width = modelStruct->matWidth;
    Eigen::MatrixXd tstMatArr[1];
    
    int inputVecSize = height*width; // input vector size
    int inputSize = modelStruct->trainDataSize; 
    int numPredPoints = modelStruct->numPredPoints;

    // Predictions
    Eigen::MatrixXd prediction;
    // Open the file to write the time series predictions
    std::ofstream out_file;
    out_file.open(predict,std::ofstream::out | std::ofstream::trunc);
    std::ofstream out_file2;
    out_file2.open(expect,std::ofstream::out | std::ofstream::trunc);
    
    // CNN Inputs
    std::vector<double> inVec;
    
    // LSTM Inputs
    std::vector<double> * input;
    input = new std::vector<double>[1];
    
    double predPoints[numPredPoints];
    double lstmPredPoints[numPredPoints];

    for (int j = 0; j < numPredPoints; j++) {
        predPoints[j] = 0;
        lstmPredPoints[j] = 0;
    }
    
    
    // max and min training values [ CNN ]
    double trainMax = *std::max_element(timeSeries.begin(), timeSeries.begin()+(inputSize+(width*height)));
    double trainMin = *std::min_element(timeSeries.begin(), timeSeries.begin()+(inputSize+(width*height)));
    // max and min predicted values [ CNN ]
    double predictMax = std::numeric_limits<double>::min();
    double predictMin = std::numeric_limits<double>::max();

    for (int i = 0; i < inputSize; i++) {
        inVec.clear();
        // filling the input vector using time series data
        for (int j = 0; j < inputVecSize; j++) {
            inVec.push_back(timeSeries2.at(i+j));
        }
        inVec = dataproc->process(inVec,0);
    
        
        // LSTM network predictions for the trained data set
        input[0] = inVec;
        for (int j = 0; j < numPredPoints; j++) {          
            result = lstm->predict(input); 
            input[0] = std::vector<double>(inVec.begin()+1, inVec.begin()+inputVecSize);
            input[0].push_back(result);
            lstmPredPoints[((i+inputVecSize+j)%numPredPoints)] += result;     
        }
        lstmPredPoints[((i+inputVecSize)%numPredPoints)] = 0;
        
        // CNN predictions for the trained data set
        tstMatArr[0] = Eigen::MatrixXd::Zero(height,width);
        for (int a = 0; a < height; a++) {
            for (int b = 0; b < width; b++) {
                tstMatArr[0](a,b) = inVec.at(( a * width ) + b);
            }
        }

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

        if (i >= numPredPoints-1) {
            prediction(0,0) = predPoints[((i+inputVecSize)%numPredPoints)]/(double)numPredPoints;
            if (prediction(0,0) > predictMax) predictMax = prediction(0,0);
            if (prediction(0,0) < predictMin) predictMin = prediction(0,0);
        }
        predPoints[((i+inputVecSize)%numPredPoints)] = 0;
    }

    std::vector<double> inVec2;
    for (int j = 0; j < inputVecSize; j++) {
        inVec2.push_back(timeSeries2.at(inputSize+j-1));
    }
    
    for (int i = inputSize; i < predSize; i++) {

        for (int j = 0; j < inputVecSize-1; j++) {
            inVec2.at(j) = inVec2.at(j+1);
        }
        
        if ( std::abs(timeSeries2.at(i+inputVecSize-1) - result) > (result/1.5) ) {
            inVec2.at(inputVecSize-1) = timeSeries2.at((i+inputVecSize-1)%inputSize);
        }
        else inVec2.at(inputVecSize-1) = timeSeries2.at(i+inputVecSize-1);
        
        inVec = dataproc->process(inVec,0);
        
        // Filling the matrix for the CNN input
        tstMatArr[0] = Eigen::MatrixXd::Zero(height,width);
        for (int a = 0; a < height; a++) {
            for (int b = 0; b < width; b++) {
                tstMatArr[0](a,b) = inVec.at(( a * width ) + b);
            }
        }
        
        // LSTM predictions
        input[0] = inVec;
        for (int j = 0; j < numPredPoints; j++) {          
            result = lstm->predict(input); 
            input[0] = std::vector<double>(inVec.begin()+1, inVec.begin()+inputVecSize);
            input[0].push_back(result);
            lstmPredPoints[((i+inputVecSize+j)%numPredPoints)] += result;     
        }
        result = lstmPredPoints[((i+inputVecSize)%numPredPoints)]/(double)numPredPoints;
        lstmPredPoints[((i+inputVecSize)%numPredPoints)] = 0;

        // CNN predictions
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
        
        // post process CNN prediction
        val = prediction(0,0);
        val = (val - predictMin)*((trainMax - trainMin)/(predictMax - predictMin)) + trainMin;
        
        // combining the results LSTM and CNN
//        val = (result + val)/2;
        val = (result*lstmW + val*cnnW);
         
        // calculating the Mean Squared Error
        expected = timeSeries.at(i+inputVecSize);
        errorSq += std::pow(expected-val,2);
        result = dataproc->postProcess(val);
        
        // filling the values to compare the similarity
        for (int j = 0; j < subVSize; j++) {
            expectedVec(j) = expectedVec(j+1);
            predictedVec(j) = predictedVec(j+1);
        }
        predictedVec(subVSize) = result;
        expectedVec(subVSize) = timeSeries2.at(i+inputVecSize);
        
        // Extracting the similarity
        similarity = DTW::getSimilarity(expectedVec,predictedVec);
//        out_file<<similarity<<"\n";
        if (similarity > simMargin) { 
            out_file<<marker<<"\n";
        } else {
            out_file<<"\n";
        }
        out_file2<<timeSeries2.at(i+inputVecSize)<<"\n";

    }
    
    out_file.close();
    out_file2.close();
    
    MSE = errorSq/(predSize-inputSize);
    std::cout<<"\nMean Squared Error: "<<MSE<<"\n\n"; 
    
    return 0;
}

int LSTMCNNFCPredictionModel::predictAdaptNorm(int points, std::string expect, std::string predict, int timeLimit, double lstmW, double cnnW) {
    
    return 0;
}

int LSTMCNNFCPredictionModel::predictAdaptNorm(
    int points, std::string expect, std::string predict, 
    int timeLimit, int simVecSize, double marker, 
    double simMargin, double lstmW, double cnnW
) {

    Eigen::VectorXd expectedVec = Eigen::VectorXd::Zero(simVecSize);
    Eigen::VectorXd predictedVec = Eigen::VectorXd::Zero(simVecSize);
    int subVSize = simVecSize-1;
    double similarity;
    double maxSim = 0;
    
    double errorSq = 0, MSE, expected, val;
    int predSize = points;
    
    // LSTM parameters
    double result;
    
    // CNN parameters
    int height = modelStruct->matHeight;
    int width = modelStruct->matWidth;
    Eigen::MatrixXd tstMatArr[1];
    
    int inputVecSize = height*width; // input vector size
    int inputSize = modelStruct->trainDataSize; 
    int numPredPoints = modelStruct->numPredPoints;

    // Predictions
    Eigen::MatrixXd prediction;
    // Open the file to write the time series predictions
    std::ofstream out_file;
    out_file.open(predict,std::ofstream::out | std::ofstream::trunc);
    std::ofstream out_file2;
    out_file2.open(expect,std::ofstream::out | std::ofstream::trunc);
    
    // CNN Inputs
    std::vector<double> inVec;
    
    // LSTM Inputs
    std::vector<double> * input;
    input = new std::vector<double>[1];
    
    double predPoints[numPredPoints];
    double lstmPredPoints[numPredPoints];

    for (int j = 0; j < numPredPoints; j++) {
        predPoints[j] = 0;
        lstmPredPoints[j] = 0;
    }
    
    
    // max and min training values [ CNN ]
    double trainMax = *std::max_element(timeSeries.begin(), timeSeries.begin()+(inputSize+(width*height)));
    double trainMin = *std::min_element(timeSeries.begin(), timeSeries.begin()+(inputSize+(width*height)));
    // max and min predicted values [ CNN ]
    double predictMax = std::numeric_limits<double>::min();
    double predictMin = std::numeric_limits<double>::max();

    for (int i = 0; i < inputSize; i++) {
        inVec.clear();
        // filling the input vector using time series data
        for (int j = 0; j < inputVecSize; j++) {
            inVec.push_back(timeSeries2.at(i+j));
        }
        inVec = dataproc->process(inVec,0);
    
        
        // LSTM network predictions for the trained data set
        input[0] = inVec;
        for (int j = 0; j < numPredPoints; j++) {          
            result = lstm->predict(input); 
            input[0] = std::vector<double>(inVec.begin()+1, inVec.begin()+inputVecSize);
            input[0].push_back(result);
            lstmPredPoints[((i+inputVecSize+j)%numPredPoints)] += result;     
        }
        lstmPredPoints[((i+inputVecSize)%numPredPoints)] = 0;
        
        // filling the values to compare the similarity
        for (int j = 0; j < subVSize; j++) {
            expectedVec(j) = expectedVec(j+1);
            predictedVec(j) = predictedVec(j+1);
        }
        predictedVec(subVSize) = dataproc->postProcess(result);
        expectedVec(subVSize) = timeSeries2.at(i+inputVecSize);        
        similarity = DTW::getSimilarity(expectedVec,predictedVec); 
        if ( maxSim < similarity) maxSim = similarity;

        
        // CNN predictions for the trained data set
        tstMatArr[0] = Eigen::MatrixXd::Zero(height,width);
        for (int a = 0; a < height; a++) {
            for (int b = 0; b < width; b++) {
                tstMatArr[0](a,b) = inVec.at(( a * width ) + b);
            }
        }

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

        if (i >= numPredPoints-1) {
            prediction(0,0) = predPoints[((i+inputVecSize)%numPredPoints)]/(double)numPredPoints;
            if (prediction(0,0) > predictMax) predictMax = prediction(0,0);
            if (prediction(0,0) < predictMin) predictMin = prediction(0,0);
        }
        predPoints[((i+inputVecSize)%numPredPoints)] = 0;
    }
    
    if ( simMargin != 0) maxSim = simMargin;

    std::vector<double> inVec2;
    for (int j = 0; j < inputVecSize; j++) {
        inVec2.push_back(timeSeries2.at(inputSize+j-1));
    }
    
    // start index of the initial normal pattern
    int normStart = 0;
    // time points without an anomaly [warp distance greater than given margin]
    int continuance = 0;
    // index of the last time point that has warp distance greater than given margin
    int lastAnom = 0;
    
    for (int i = inputSize; i < predSize; i++) {

        for (int j = 0; j < inputVecSize-1; j++) {
            inVec2.at(j) = inVec2.at(j+1);
        }
        
        if ( result*2 < timeSeries2.at(i+inputVecSize-1) ) {
            inVec2.at(inputVecSize-1) = timeSeries2.at(((i+inputVecSize-1)%inputSize) + normStart);
        }
        else inVec2.at(inputVecSize-1) = timeSeries2.at(i+inputVecSize-1);
        inVec2.at(inputVecSize-1) = timeSeries2.at(i+inputVecSize-1);
        
        
        inVec = dataproc->process(inVec2,0);
        
        // Filling the matrix for the CNN input
        tstMatArr[0] = Eigen::MatrixXd::Zero(height,width);
        for (int a = 0; a < height; a++) {
            for (int b = 0; b < width; b++) {
                tstMatArr[0](a,b) = inVec.at(( a * width ) + b);
            }
        }
        
        // LSTM predictions
        input[0] = inVec;
        for (int j = 0; j < numPredPoints; j++) {          
            result = lstm->predict(input); 
            input[0] = std::vector<double>(inVec.begin()+1, inVec.begin()+inputVecSize);
            input[0].push_back(result);
            lstmPredPoints[((i+inputVecSize+j)%numPredPoints)] += result;     
        }
        result = lstmPredPoints[((i+inputVecSize)%numPredPoints)]/(double)numPredPoints;
        lstmPredPoints[((i+inputVecSize)%numPredPoints)] = 0;

        // CNN predictions
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
        
        // post process CNN prediction
        val = prediction(0,0);
        val = (val - predictMin)*((trainMax - trainMin)/(predictMax - predictMin)) + trainMin;
        
        // combining the results LSTM and CNN
        val = (result*lstmW + val*cnnW);
         
        // calculating the Mean Squared Error
        expected = timeSeries.at(i+inputVecSize);
        errorSq += std::pow(expected-val,2);
        result = dataproc->postProcess(val);
        
        // check for the continuance of the time series to update the normal pattern
        if (continuance = timeLimit) {
            if ((i+inputVecSize-timeLimit) == lastAnom) {
//                if (inputSize < timeLimit) normStart = (i+inputVecSize-inputSize);
//                else normStart = (i+inputVecSize-timeLimit);
                normStart = lastAnom;    
                std::cout<<"normal pattern changed: "<<normStart<<"\n";
            }
        }
        
        // filling the values to compare the similarity
        for (int j = 0; j < subVSize; j++) {
            expectedVec(j) = expectedVec(j+1);
            predictedVec(j) = predictedVec(j+1);
        }
        predictedVec(subVSize) = result;
        expectedVec(subVSize) = timeSeries2.at(i+inputVecSize);
        
        // Extracting the similarity
        similarity = DTW::getSimilarity(expectedVec,predictedVec);
        
        if (similarity > maxSim) { 
            
            // index of the point after the identified anomalous point
            lastAnom = i+inputVecSize+1;
            // reset the time period
            continuance = 0;
            out_file<<marker<<"\n";
        } else {
            out_file<<"\n";
        }
        
//        out_file<<similarity<<"\n";
        out_file2<<timeSeries2.at(i+inputVecSize)<<"\n";

    }
    
    out_file.close();
    out_file2.close();
    
    MSE = errorSq/(predSize-inputSize);
    std::cout<<"\nMean Squared Error: "<<MSE<<"\n\n"; 
    
    return 0;
}

int LSTMCNNFCPredictionModel::predictFromFile(
    std::string infile, int points, std::string predict, 
    double lstmW, double cnnW, int abs
) {
    
    double errorSq = 0, MSE, expected, val;
    int predSize = points;
    
    // LSTM parameters
    double result;
    
    // CNN parameters
    int height = modelStruct->matHeight;
    int width = modelStruct->matWidth;
    Eigen::MatrixXd tstMatArr[1];
    
    int inputVecSize = height*width; // input vector size
    int inputSize = modelStruct->trainDataSize; 
    int numPredPoints = modelStruct->numPredPoints;

    // Predictions
    Eigen::MatrixXd prediction;
    // Open the file to write the time series predictions
    std::ofstream out_file;
    out_file.open(predict,std::ofstream::out | std::ofstream::trunc);
    
    // CNN Inputs
    std::vector<double> inVec;
    
    // LSTM Inputs
    std::vector<double> * input;
    input = new std::vector<double>[1];
    
    double predPoints[numPredPoints];
    double lstmPredPoints[numPredPoints];

    for (int j = 0; j < numPredPoints; j++) {
        predPoints[j] = 0;
        lstmPredPoints[j] = 0;
    }
    
    
    // max and min training values [ CNN ]
    double trainMax = *std::max_element(timeSeries.begin(), timeSeries.begin()+(inputSize+(width*height)));
    double trainMin = *std::min_element(timeSeries.begin(), timeSeries.begin()+(inputSize+(width*height)));
    // max and min predicted values [ CNN ]
    double predictMax = std::numeric_limits<double>::min();
    double predictMin = std::numeric_limits<double>::max();

    for (int i = 0; i < inputSize; i++) {
        inVec.clear();
        // filling the input vector using time series data
        for (int j = 0; j < inputVecSize; j++) {
            inVec.push_back(timeSeries2.at(i+j));
        }
        inVec = dataproc->process(inVec,0);
    
        
        // LSTM network predictions for the trained data set
        input[0] = inVec;
        for (int j = 0; j < numPredPoints; j++) {          
            result = lstm->predict(input); 
            input[0] = std::vector<double>(inVec.begin()+1, inVec.begin()+inputVecSize);
            input[0].push_back(result);
            lstmPredPoints[((i+inputVecSize+j)%numPredPoints)] += result;     
        }
        lstmPredPoints[((i+inputVecSize)%numPredPoints)] = 0;

        
        // CNN predictions for the trained data set
        tstMatArr[0] = Eigen::MatrixXd::Zero(height,width);
        for (int a = 0; a < height; a++) {
            for (int b = 0; b < width; b++) {
                tstMatArr[0](a,b) = inVec.at(( a * width ) + b);
            }
        }

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

        if (i >= numPredPoints-1) {
            prediction(0,0) = predPoints[((i+inputVecSize)%numPredPoints)]/(double)numPredPoints;
            if (prediction(0,0) > predictMax) predictMax = prediction(0,0);
            if (prediction(0,0) < predictMin) predictMin = prediction(0,0);
        }
        predPoints[((i+inputVecSize)%numPredPoints)] = 0;
    }

    // open file
    std::ifstream inFile (infile);
    int readLines = 0;
    std::string line;
    std::vector<double> tmpInVec;
    int i = 0;
    

//  timeSeries2.clear();
    if (inFile.is_open()) {

        // read the data for initial input vector
        tmpInVec.clear();
        while ( getline (inFile,line) ) {
            readLines++;
            try{
                tmpInVec.push_back(std::stod(line));
            } catch (std::exception& e) {
                std::cout<<std::endl
                        <<"Error in line "
                        <<(readLines)
                        <<": "<<e.what()<<std::endl;
            }
            if (readLines == inputVecSize) break;
        }

        // writing the header
        out_file<<"Value,Prediction\n";
        
        // Predictions using the model
        int readLines = 0;
        while ( getline (inFile,line) ) {
            readLines++;
            try{
                tmpInVec.push_back(std::stod(line));
            } catch (std::exception& e) {
                std::cout<<std::endl
                        <<"Error in line "
                        <<(readLines)
                        <<": "<<e.what()<<std::endl;
            }

            inVec.clear();
            for (int j = 0; j < inputVecSize; j++) {
                inVec.push_back(tmpInVec.at(j));
            }
            inVec = dataproc->process(inVec,0);

            // Filling the matrix for the CNN input
            tstMatArr[0] = Eigen::MatrixXd::Zero(height,width);
            for (int a = 0; a < height; a++) {
                for (int b = 0; b < width; b++) {
                    tstMatArr[0](a,b) = inVec.at(( a * width ) + b);
                }
            }

            // LSTM predictions
            input[0] = inVec;
            for (int j = 0; j < numPredPoints; j++) {          
                result = lstm->predict(input); 
                input[0] = std::vector<double>(inVec.begin()+1, inVec.begin()+inputVecSize);
                input[0].push_back(result);
                lstmPredPoints[((i+inputVecSize+j)%numPredPoints)] += result;     
            }
            result = lstmPredPoints[((i+inputVecSize)%numPredPoints)]/(double)numPredPoints;
            lstmPredPoints[((i+inputVecSize)%numPredPoints)] = 0;

            // CNN predictions
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

            // post process CNN prediction
            val = prediction(0,0);
            val = (val - predictMin)*((trainMax - trainMin)/(predictMax - predictMin)) + trainMin;

            // combining the results LSTM and CNN
            val = (result*lstmW + val*cnnW);

            // calculating the Mean Squared Error
//          expected = timeSeries.at(inputVecSize);
            // get the absolute value
            if (abs) val = std::abs(val);
//          errorSq += std::pow(expected-val,2);
            result = dataproc->postProcess(val);

            // writing the predictions
//          out_file<<result<<"\n";
//          out_file2<<tmpInVec.at(inputVecSize)<<"\n";
            out_file<<tmpInVec.at(inputVecSize)<<","<<result<<"\n";

            i++;
            tmpInVec.erase (tmpInVec.begin());
            
            std::cout<<*tmpInVec.begin()<<"\n";
            
            if (readLines == points) {
                break;
            }// else std::cout<<"Processing data: "<<"line "<<i<<'\r';
        }
        std::cout<<"Processing data: Completed ["<<i<<" points]";
        inFile.close();
    }
    else std::cout << "Unable to open file '"<<infile<<"'"; 
        
    
    out_file.close();
    
//    MSE = errorSq/(predSize-inputSize);
//    std::cout<<"\nMean Squared Error: "<<MSE<<"\n\n"; 
    
    return 0;
}

int LSTMCNNFCPredictionModel::dtwSimilarity(
    int points, std::string expect, std::string predict, 
    int simVecSize, double lstmW, double cnnW, int abs
) {
    
    Eigen::VectorXd expectedVec = Eigen::VectorXd::Zero(simVecSize);
    Eigen::VectorXd predictedVec = Eigen::VectorXd::Zero(simVecSize);
    int subVSize = simVecSize-1;
    double similarity;
    
    double errorSq = 0, MSE, expected, val;
    int predSize = points;
    
    // LSTM parameters
    double result;
    
    // CNN parameters
    int height = modelStruct->matHeight;
    int width = modelStruct->matWidth;
    Eigen::MatrixXd tstMatArr[1];
    
    int inputVecSize = height*width; // input vector size
    int inputSize = modelStruct->trainDataSize; 
    int numPredPoints = modelStruct->numPredPoints;

    // Predictions
    Eigen::MatrixXd prediction;
    // Open the file to write the time series predictions
    std::ofstream out_file;
    out_file.open(predict,std::ofstream::out | std::ofstream::trunc);
    std::ofstream out_file2;
    out_file2.open(expect,std::ofstream::out | std::ofstream::trunc);
    
    // CNN Inputs
    std::vector<double> inVec;
    
    // LSTM Inputs
    std::vector<double> * input;
    input = new std::vector<double>[1];
    
    double predPoints[numPredPoints];
    double lstmPredPoints[numPredPoints];

    for (int j = 0; j < numPredPoints; j++) {
        predPoints[j] = 0;
        lstmPredPoints[j] = 0;
    }
    
    
    // max and min training values [ CNN ]
    double trainMax = *std::max_element(timeSeries.begin(), timeSeries.begin()+(inputSize+(width*height)));
    double trainMin = *std::min_element(timeSeries.begin(), timeSeries.begin()+(inputSize+(width*height)));
    // max and min predicted values [ CNN ]
    double predictMax = std::numeric_limits<double>::min();
    double predictMin = std::numeric_limits<double>::max();

    for (int i = 0; i < inputSize; i++) {
        inVec.clear();
        // filling the input vector using time series data
        for (int j = 0; j < inputVecSize; j++) {
            inVec.push_back(timeSeries2.at(i+j));
        }
        inVec = dataproc->process(inVec,0);
    
        
        // LSTM network predictions for the trained data set
        input[0] = inVec;
        for (int j = 0; j < numPredPoints; j++) {          
            result = lstm->predict(input); 
            input[0] = std::vector<double>(inVec.begin()+1, inVec.begin()+inputVecSize);
            input[0].push_back(result);
            lstmPredPoints[((i+inputVecSize+j)%numPredPoints)] += result;     
        }
        lstmPredPoints[((i+inputVecSize)%numPredPoints)] = 0;
        
        // CNN predictions for the trained data set
        tstMatArr[0] = Eigen::MatrixXd::Zero(height,width);
        for (int a = 0; a < height; a++) {
            for (int b = 0; b < width; b++) {
                tstMatArr[0](a,b) = inVec.at(( a * width ) + b);
            }
        }

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
        
        // Filling the matrix for the CNN input
        tstMatArr[0] = Eigen::MatrixXd::Zero(height,width);
        for (int a = 0; a < height; a++) {
            for (int b = 0; b < width; b++) {
                tstMatArr[0](a,b) = inVec.at(( a * width ) + b);
            }
        }
        
        // LSTM predictions
        input[0] = inVec;
        for (int j = 0; j < numPredPoints; j++) {          
            result = lstm->predict(input); 
            input[0] = std::vector<double>(inVec.begin()+1, inVec.begin()+inputVecSize);
            input[0].push_back(result);
            lstmPredPoints[((i+inputVecSize+j)%numPredPoints)] += result;     
        }
        result = lstmPredPoints[((i+inputVecSize)%numPredPoints)]/(double)numPredPoints;
        lstmPredPoints[((i+inputVecSize)%numPredPoints)] = 0;

        // CNN predictions
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
        
        // post process CNN prediction
        val = prediction(0,0);
        val = (val - predictMin)*((trainMax - trainMin)/(predictMax - predictMin)) + trainMin;
        
        // combining the results LSTM and CNN
        val = (result*lstmW + val*cnnW);
        
        // get the absolute value
        if (abs) val = std::abs(val);
         
        // calculating the Mean Squared Error
        expected = timeSeries.at(i+inputVecSize);
        errorSq += std::pow(expected-val,2);
        result = dataproc->postProcess(val);
        
        // filling the values to compare the similarity
        for (int j = 0; j < subVSize; j++) {
            expectedVec(j) = expectedVec(j+1);
            predictedVec(j) = predictedVec(j+1);
        }
        predictedVec(subVSize) = result;
        expectedVec(subVSize) = timeSeries2.at(i+inputVecSize);
        
        // Extracting the similarity
        similarity = DTW::getSimilarity(expectedVec,predictedVec); 
        out_file<<similarity<<"\n";
        out_file2<<timeSeries2.at(i+inputVecSize)<<"\n";
    }
    
    out_file.close();
    out_file2.close();
    
    MSE = errorSq/(predSize-inputSize);
    std::cout<<"\nMean Squared Error: "<<MSE<<"\n\n"; 
    
    return 0;
}

