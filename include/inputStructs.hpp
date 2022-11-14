#ifndef INPUTSTRUCTS_HPP
#define INPUTSTRUCTS_HPP
/**
 * @brief This header contains struct definitions that are used globally for passing data through from python to the GPU. 
 *  
 */
struct featureDataHTP
{ //absorption feature database for HTP profile
    double linecenter; //line center (wavenumber), 32b float only has ~7 decimals of precision, too low to accurately position line
    float Gam0; //Gamma0
    float Gam2; //Gamma2
    float Delta0; //shift0
    float Delta2; //shift2
    float anuVC; //nuVC
    float eta; //eta
    float lineIntensity; //line intensity
};

#endif /* INPUTSTRUCTS_HPP */
