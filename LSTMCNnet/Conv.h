/* 
 * File:   Conv.h
 * Author: heshan
 *
 * Created on June 15, 2018, 4:35 PM
 */

#ifndef CONV_H
#define CONV_H

#include "CNNet/CNN.hpp"

class Conv {
public:
    Conv();
    Conv(const Conv& orig);
    virtual ~Conv();
    
    int run();
    
private:

};

#endif /* CONV_H */

