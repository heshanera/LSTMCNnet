/* 
 * File:   DTW.hpp
 * Author: heshan
 *
 * Created on August 10, 2018, 10:59 AM
 */

#ifndef DTW_HPP
#define DTW_HPP

#include <iostream>
#include <limits>
#include "CNNet/Eigen/Dense"

class DTW {
public:
    DTW();
    DTW(const DTW& orig);
    virtual ~DTW();
    
    /**
     * 
     * @param Vector 1
     * @param Vector 1
     * @return similarity measure between the 2 vectors 
     */
    static double getSimilarity(Eigen::VectorXd, Eigen::VectorXd);
    /**
     * TO DO 
     */
    static double fastDTW(Eigen::VectorXd, Eigen::VectorXd, int);
    /**
     * 
     * @param value 1
     * @param value 2
     * @param value 3
     * @return minimum of the 3 values
     */
    static double min(double, double, double);
private:

};

#endif /* DTW_HPP */

