#ifndef CL_COMPLEX_H
#define CL_COMPLEX_H

struct clComplex{
    // clComplex(double real, double cmplx){re=real;cmp=cmplx;}
    double re;
    double cmp;
};

struct clComplex make_clComplex(double real, double cmplx){
    struct clComplex out;
    out.re = real;
    out.cmp = cmplx;
    return out;
}

// struct clComplex clExp(double z){
//     return make_clComplex(cos(z),sin(z));
// }

double clReal(struct clComplex z){
    return z.re;
}

double clImag(struct clComplex z){
    return z.cmp;
}

struct clComplex clPolar(double r, double t){
    return make_clComplex(r * cos(t), r * sin(t));
}

struct clComplex clMul(struct clComplex x, struct clComplex y){
    struct clComplex prod;
    prod = make_clComplex((clReal(x) * clReal(y)) - 
                    (clImag(x) * clImag(y)),
                    (clReal(x) * clImag(y)) + 
                    (clImag(x) * clReal(y)));
    return prod;
}

struct clComplex clDiv(struct clComplex x,struct clComplex y)
{
    struct clComplex quot;
    double s = (fabs(clReal(y))) + (fabs(clImag(y)));
    double oos = 1.0 / s;
    double ars = clReal(x) * oos;
    double ais = clImag(x) * oos;
    double brs = clReal(y) * oos;
    double bis = clImag(y) * oos;
    s = (brs * brs) + (bis * bis);
    oos = 1.0 / s;
    quot = make_clComplex (((ars * brs) + (ais * bis)) * oos,
                                 ((ais * brs) - (ars * bis)) * oos);
    return quot;
}

struct clComplex clSub(struct clComplex x, struct clComplex y) {
    return make_clComplex(clReal(x) - clReal(y), clImag(x) - clImag(y));
}

struct clComplex clAdd(struct clComplex x, struct clComplex y){
    return make_clComplex(clReal(x) + clReal(y), clImag(x) + clImag(y));
}

#endif