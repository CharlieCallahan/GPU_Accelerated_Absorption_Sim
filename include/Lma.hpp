/* Copyright (c) 2022 Charlie Callahan
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 * LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 * OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 * WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef LMA_HPP
#define LMA_HPP

/**
 * @brief Virtual class that defines required operations for a vector used by the LMA routine
 * 
 */
class LMVec{
    public:
    /**
     * @brief Add vectors
     * 
     * @param vec 
     * @return LMVec 
     */
    virtual LMVec* operator+( LMVec* vec)=0;

    virtual void addInPlace( LMVec* vec)=0;

    /**
     * @brief Subtract vectors
     * 
     * @param vec 
     * @return LMVec 
     */
    virtual LMVec* operator-( LMVec* vec)=0;

    /**
     * @brief Return the sum of each value squared
     * 
     * @return float 
     */
    virtual float sumSq()=0; 

    /**
     * @brief Make a copy of this vector
     * 
     * @return LMVec* 
     */
    virtual LMVec* copy()=0;

    virtual ~LMVec(){}
};

/**
 * @brief Virtual class that defines required matrix operations for lma.
 * Intended to allow different types of matrices, sparse, GPU vs CPU etc.
 * 
 */
class LMMat{
    public:
    
    /**
     * @brief Calculate mat*transpose(mat) + lambda*IdentityMatrix
     * 
     * @param lambda 
     * @return LMMat* 
     */
    virtual LMMat* calcMMTpLambdaI(float lambda)=0;

    /**
     * @brief solves the linear equation A*x=b for x - (A is this matrix)
     * 
     * @param b 
     * @return LMVec* x
     */
    virtual LMVec* solve(LMVec* b)=0;

    /**
     * @brief Matrix vector multiplication
     * 
     */
    virtual LMVec* operator*( LMVec* vec)=0;

    virtual ~LMMat(){};
};


struct LMAFitSettings{
    int maxIterations = 100;
};

/**
 * @brief Code to handle levenberg marquardt curve fitting. The idea here is to define the actual LM algorithm
 * in this base class and make a derived class which can define its own application specific implementation of 
 * getJacobian and getModel.
 * 
 */

class LMATask {
public:
    LMATask(float lambda);

    /**
     * @brief This runs the fitting routine and returns a best fit for beta.
     * 
     * @return LMVec 
     */
    LMVec* fitBeta( LMVec& data, LMVec& betaGuess, LMAFitSettings& settings);

    /**
     * @brief Get the transposed jacobian 
     * 
     *               len(data)
     *          _________________
     *         | df_x0/dbeta_0, df_x1/dbeta_0 ... 
     *         | df_x0/dbeta_1,
     *         |    .
     len(beta) |    .
     *         |    .
     *         |
     *         |
     * 
     * 
     * @param beta 
     * @return LMMat 
     */
    virtual LMMat* getTransposedJacobian(LMVec& beta) = 0;

    /**
     * @brief Evaluate the model at beta
     * 
     * @param beta 
     * @return LMVec 
     */
    virtual LMVec* getModel(LMVec* beta) = 0;

    float lambda;
};

#endif /* LMA_HPP */
