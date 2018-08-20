/* 
 * File:   CNNPredictionModel.cpp
 * Author: heshan
 * 
 * Created on August 7, 2018, 7:53 PM
 */

#include "CNNPredictionModel.hpp"

CNNPredictionModel::CNNPredictionModel() { }

CNNPredictionModel::CNNPredictionModel(ModelStruct * modelStruct) {
    this->modelStruct = modelStruct;
}

CNNPredictionModel::CNNPredictionModel(const CNNPredictionModel& orig) { }

CNNPredictionModel::~CNNPredictionModel() { }

int CNNPredictionModel::train() {

    // Generating a convolutional network
    int width = modelStruct->matWidth;
    int height = modelStruct->matHeight;
    int iterations = modelStruct->trainingIterations;
    int trainDataSize = modelStruct->trainDataSize;
    int targetsC = modelStruct->targetC;
    double learningRate = modelStruct->learningRate;

    // network structure
    std::tuple<int, int, int> dimensions = std::make_tuple(1,height,width);

    // Generating input matrices
    Eigen::MatrixXd ** inMatArr;
    Eigen::MatrixXd * inLblArr;
    Eigen::MatrixXd inMat;
    Eigen::MatrixXd inLbl;
    inMatArr = new Eigen::MatrixXd * [trainDataSize];
    inLblArr = new Eigen::MatrixXd[trainDataSize];

    dataproc = new DataProcessor();
    fileProc = new FileProcessor();
    
    // Reading the file
    timeSeries2 = fileProc->read(modelStruct->dataFile,1);
    timeSeries =  dataproc->process(timeSeries2,1);

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

    // Generating the network
    this->cnn = new CNN(dimensions, modelStruct->netStruct);
    // Training the network
    cnn->train(inMatArr, inLblArr, trainDataSize, iterations, learningRate);
    
    return 0;
}

int CNNPredictionModel::initPredData(std::string file) {

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

int CNNPredictionModel::predict(int points, std::string expect, std::string predict) {

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

    int numPredPoints = modelStruct->numPredPoints;
    double predPoints[numPredPoints];

    for (int j = 0; j < numPredPoints; j++) {
        predPoints[j] = 0;
    }


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

        prediction(0,0) = predPoints[((i+inputVecSize)%numPredPoints)]/(double)numPredPoints;
        predPoints[((i+inputVecSize)%numPredPoints)] = 0; 
        
        // post process
        val = prediction(0,0);
        val = (val - predictMin)*((trainMax - trainMin)/(predictMax - predictMin)) + trainMin;
        
        // calculating the error
        expected = timeSeries.at(i + (width*height));
        errorSq += pow(val - expected,2);

        out_file<<dataproc->postProcess(val)<<"\n";
        out_file2<<timeSeries2.at(i+inputVecSize)<<"\n";
    }
    MSE = errorSq/(predSize-inputSize);
    std::cout<<"\nMean Squared Error: "<<MSE<<"\n\n"; 
    
    return 0;
}

int CNNPredictionModel::predict(int points, std::string expect, std::string predict, int simVecSize, double marker, double simMargin) {

    Eigen::VectorXd expectedVec = Eigen::VectorXd::Zero(simVecSize);
    Eigen::VectorXd predictedVec = Eigen::VectorXd::Zero(simVecSize);
    int subVSize = simVecSize-1;
    double similarity;
    double maxSim = 0;
    
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

    int numPredPoints = modelStruct->numPredPoints;
    double predPoints[numPredPoints];

    for (int j = 0; j < numPredPoints; j++) {
        predPoints[j] = 0;
    }


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
        
        // post process
        val = prediction(0,0);
        val = (val - predictMin)*((trainMax - trainMin)/(predictMax - predictMin)) + trainMin;
        
        // filling the values to compare the similarity
        for (int j = 0; j < subVSize; j++) {
            expectedVec(j) = expectedVec(j+1);
            predictedVec(j) = predictedVec(j+1);
        }
        
        val = dataproc->postProcess(val);
        if ( std::isnan(val) ) val = 0;
        predictedVec(subVSize) = val;
        expectedVec(subVSize) = timeSeries2.at(i+inputVecSize);        
        similarity = DTW::getSimilarity(expectedVec,predictedVec); 
        if ( maxSim < similarity) maxSim = similarity;
        
    }
    
    if ( simMargin != 0) maxSim = simMargin;

    for (int i = inputSize; i < predSize; i++) {

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
        
        // post process
        val = prediction(0,0);
        val = (val - predictMin)*((trainMax - trainMin)/(predictMax - predictMin)) + trainMin;
        
        // calculating the error
        expected = timeSeries.at(i + (width*height));
        errorSq += pow(val - expected,2);

        // filling the values to compare the similarity
        for (int j = 0; j < subVSize; j++) {
            expectedVec(j) = expectedVec(j+1);
            predictedVec(j) = predictedVec(j+1);
        }
        
        val = dataproc->postProcess(val);
        if ( std::isnan(val) ) val = 0;
        predictedVec(subVSize) = val;
        expectedVec(subVSize) = timeSeries2.at(i+inputVecSize);
        
        // Extracting the similarity
        similarity = DTW::getSimilarity(expectedVec,predictedVec);
        if (similarity > maxSim) { 
            out_file<<marker<<"\n";
        } else {
            out_file<<"\n";
        }
        
        out_file2<<timeSeries2.at(i+inputVecSize)<<"\n";
    }
    MSE = errorSq/(predSize-inputSize);
    std::cout<<"\nMean Squared Error: "<<MSE<<"\n\n"; 
    
    return 0;
}