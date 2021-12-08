//
//  Stopwatch.hpp
//
//  Created by Charles callahan on 1/9/21.
//  Copyright © 2021 Charles callahan. All rights reserved.
//

#ifndef Stopwatch_hpp
#define Stopwatch_hpp

#include <stdio.h>
#include <chrono>

class Stopwatch{
public:
    Stopwatch(){};
    void start();
    double stop_time();
    void print_stop_time();
private:
    std::chrono::high_resolution_clock::time_point t1;
    std::chrono::high_resolution_clock::time_point t2;
};
#endif /* Stopwatch_hpp */
