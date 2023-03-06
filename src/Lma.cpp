#include "Lma.hpp"
#include <iostream>

LMATask::LMATask(float lambda){
    this->lambda = lambda;
}

LMVec* LMATask::fitBeta(LMVec& data, LMVec& betaGuess, LMAFitSettings& settings){
    LMVec* currBeta = betaGuess.copy();
    for(int i = 0; i < settings.maxIterations; i++){
        LMMat* JT = getTransposedJacobian(*currBeta);
        LMMat* JJTLI = JT->calcMMTpLambdaI(lambda); // = JTJ + lambda*I

        LMVec* f = getModel(currBeta);
        LMVec* err = (data) - (f);
        LMVec* rhs = (*JT)*(err);
        delete f;
        delete err;
        delete JT;

        LMVec* delta = JJTLI->solve(rhs);
        
        delete JJTLI;
        delete rhs;
        currBeta->addInPlace(delta);
        std::cout << "Iteration: "<<i<<" Error: "<<delta->sumSq()<<"\n";
        delete delta;
        }
}
