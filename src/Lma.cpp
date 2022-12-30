#include "Lma.hpp"
#include <iostream>

LMATask::LMATask(float lambda){
    this->lambda = lambda;
}

LMVec* LMATask::fitBeta(LMVec& data, LMVec& betaGuess, LMAFitSettings& settings){
    LMVec* currBeta = betaGuess.copy();
    for(int i = 0; i < settings.maxIterations; i++){
        LMMat* J = getJacobian(*currBeta);
        LMMat* JT = J->transpose();
        LMMat* JTJ = (*JT)*(J);

        LMMat* lambdaI = J->idMatLike();
        lambdaI->scale(lambda);


        LMMat* lhs = (*JTJ)+(lambdaI);
        
        //free up some mem
        delete JTJ;
        delete lambdaI;
        delete J;

        LMVec* f = getModel(currBeta);
        LMVec* err = (data) - (f);
        LMVec* rhs = (*JT)*(err);
        delete f;
        delete err;
        delete JT;

        LMVec* delta = lhs->solve(rhs);
        
        delete lhs;
        delete rhs;
        currBeta->addInPlace(delta);
        std::cout << "Iteration: "<<i<<" Error: "<<delta->sumSq()<<"\n";
        delete delta;
        }
}
