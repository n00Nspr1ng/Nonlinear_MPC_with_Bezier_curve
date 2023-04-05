/* This file was automatically generated by CasADi.
   The CasADi copyright holders make no ownership claim of its contents. */
#ifdef __cplusplus
extern "C" {
#endif

/* How to prefix internal symbols */
#ifdef CASADI_CODEGEN_PREFIX
  #define CASADI_NAMESPACE_CONCAT(NS, ID) _CASADI_NAMESPACE_CONCAT(NS, ID)
  #define _CASADI_NAMESPACE_CONCAT(NS, ID) NS ## ID
  #define CASADI_PREFIX(ID) CASADI_NAMESPACE_CONCAT(CODEGEN_PREFIX, ID)
#else
  #define CASADI_PREFIX(ID) Path_parametric_MPC_expl_ode_fun_ ## ID
#endif

#include <math.h>

#ifndef casadi_real
#define casadi_real double
#endif

#ifndef casadi_int
#define casadi_int int
#endif

/* Add prefix to internal symbols */
#define casadi_f0 CASADI_PREFIX(f0)
#define casadi_fmax CASADI_PREFIX(fmax)
#define casadi_s0 CASADI_PREFIX(s0)
#define casadi_s1 CASADI_PREFIX(s1)
#define casadi_s2 CASADI_PREFIX(s2)
#define casadi_sign CASADI_PREFIX(sign)
#define casadi_sq CASADI_PREFIX(sq)

/* Symbol visibility in DLLs */
#ifndef CASADI_SYMBOL_EXPORT
  #if defined(_WIN32) || defined(__WIN32__) || defined(__CYGWIN__)
    #if defined(STATIC_LINKED)
      #define CASADI_SYMBOL_EXPORT
    #else
      #define CASADI_SYMBOL_EXPORT __declspec(dllexport)
    #endif
  #elif defined(__GNUC__) && defined(GCC_HASCLASSVISIBILITY)
    #define CASADI_SYMBOL_EXPORT __attribute__ ((visibility ("default")))
  #else
    #define CASADI_SYMBOL_EXPORT
  #endif
#endif

casadi_real casadi_sq(casadi_real x) { return x*x;}

casadi_real casadi_sign(casadi_real x) { return x<0 ? -1 : x>0 ? 1 : x;}

casadi_real casadi_fmax(casadi_real x, casadi_real y) {
/* Pre-c99 compatibility */
#if __STDC_VERSION__ < 199901L
  return x>y ? x : y;
#else
  return fmax(x, y);
#endif
}

static const casadi_int casadi_s0[12] = {8, 1, 0, 8, 0, 1, 2, 3, 4, 5, 6, 7};
static const casadi_int casadi_s1[6] = {2, 1, 0, 2, 0, 1};
static const casadi_int casadi_s2[3] = {0, 0, 0};

/* Path_parametric_MPC_expl_ode_fun:(i0[8],i1[2],i2[])->(o0[8]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real w0, w1, w2, w3, w4, w5, w6, w7, w8, w9;
  /* #0: @0 = input[0][3] */
  w0 = arg[0] ? arg[0][3] : 0;
  /* #1: @1 = input[0][2] */
  w1 = arg[0] ? arg[0][2] : 0;
  /* #2: @2 = cos(@1) */
  w2 = cos( w1 );
  /* #3: @3 = (@0*@2) */
  w3  = (w0*w2);
  /* #4: @4 = input[0][4] */
  w4 = arg[0] ? arg[0][4] : 0;
  /* #5: @1 = sin(@1) */
  w1 = sin( w1 );
  /* #6: @5 = (@4*@1) */
  w5  = (w4*w1);
  /* #7: @3 = (@3-@5) */
  w3 -= w5;
  /* #8: output[0][0] = @3 */
  if (res[0]) res[0][0] = w3;
  /* #9: @1 = (@0*@1) */
  w1  = (w0*w1);
  /* #10: @2 = (@4*@2) */
  w2  = (w4*w2);
  /* #11: @1 = (@1+@2) */
  w1 += w2;
  /* #12: output[0][1] = @1 */
  if (res[0]) res[0][1] = w1;
  /* #13: @1 = input[0][5] */
  w1 = arg[0] ? arg[0][5] : 0;
  /* #14: output[0][2] = @1 */
  if (res[0]) res[0][2] = w1;
  /* #15: @2 = 0.287 */
  w2 = 2.8699999999999998e-01;
  /* #16: @3 = 0.0545 */
  w3 = 5.4500000000000000e-02;
  /* #17: @5 = fabs(@0) */
  w5 = fabs( w0 );
  /* #18: @3 = (@3*@5) */
  w3 *= w5;
  /* #19: @2 = (@2-@3) */
  w2 -= w3;
  /* #20: @3 = input[0][7] */
  w3 = arg[0] ? arg[0][7] : 0;
  /* #21: @2 = (@2*@3) */
  w2 *= w3;
  /* #22: @3 = 0.0518 */
  w3 = 5.1799999999999999e-02;
  /* #23: @5 = 5 */
  w5 = 5.;
  /* #24: @5 = (@5*@0) */
  w5 *= w0;
  /* #25: @5 = tanh(@5) */
  w5 = tanh( w5 );
  /* #26: @3 = (@3*@5) */
  w3 *= w5;
  /* #27: @2 = (@2-@3) */
  w2 -= w3;
  /* #28: @3 = 0.00035 */
  w3 = 3.5000000000000000e-04;
  /* #29: @5 = sq(@0) */
  w5 = casadi_sq( w0 );
  /* #30: @3 = (@3*@5) */
  w3 *= w5;
  /* #31: @5 = sign(@0) */
  w5 = casadi_sign( w0 );
  /* #32: @3 = (@3*@5) */
  w3 *= w5;
  /* #33: @2 = (@2-@3) */
  w2 -= w3;
  /* #34: @3 = 0.192 */
  w3 = 1.9200000000000000e-01;
  /* #35: @5 = 1.2 */
  w5 = 1.2000000000000000e+00;
  /* #36: @6 = 2.579 */
  w6 = 2.5790000000000002e+00;
  /* #37: @7 = input[0][6] */
  w7 = arg[0] ? arg[0][6] : 0;
  /* #38: @8 = 0.029 */
  w8 = 2.9000000000000001e-02;
  /* #39: @8 = (@8*@1) */
  w8 *= w1;
  /* #40: @8 = (@8+@4) */
  w8 += w4;
  /* #41: @9 = 0.1 */
  w9 = 1.0000000000000001e-01;
  /* #42: @9 = fmax(@0,@9) */
  w9  = casadi_fmax(w0,w9);
  /* #43: @8 = atan2(@8,@9) */
  w8  = atan2(w8,w9);
  /* #44: @8 = (@7-@8) */
  w8  = (w7-w8);
  /* #45: @6 = (@6*@8) */
  w6 *= w8;
  /* #46: @6 = atan(@6) */
  w6 = atan( w6 );
  /* #47: @5 = (@5*@6) */
  w5 *= w6;
  /* #48: @5 = sin(@5) */
  w5 = sin( w5 );
  /* #49: @3 = (@3*@5) */
  w3 *= w5;
  /* #50: @5 = sin(@7) */
  w5 = sin( w7 );
  /* #51: @5 = (@3*@5) */
  w5  = (w3*w5);
  /* #52: @2 = (@2-@5) */
  w2 -= w5;
  /* #53: @5 = 0.041 */
  w5 = 4.1000000000000002e-02;
  /* #54: @2 = (@2/@5) */
  w2 /= w5;
  /* #55: @5 = (@4*@1) */
  w5  = (w4*w1);
  /* #56: @2 = (@2+@5) */
  w2 += w5;
  /* #57: output[0][3] = @2 */
  if (res[0]) res[0][3] = w2;
  /* #58: @2 = 0.1737 */
  w2 = 1.7369999999999999e-01;
  /* #59: @5 = 1.2691 */
  w5 = 1.2690999999999999e+00;
  /* #60: @6 = 3.3852 */
  w6 = 3.3852000000000002e+00;
  /* #61: @8 = 0.033 */
  w8 = 3.3000000000000002e-02;
  /* #62: @8 = (@8*@1) */
  w8 *= w1;
  /* #63: @8 = (@8-@4) */
  w8 -= w4;
  /* #64: @8 = atan2(@8,@9) */
  w8  = atan2(w8,w9);
  /* #65: @6 = (@6*@8) */
  w6 *= w8;
  /* #66: @6 = atan(@6) */
  w6 = atan( w6 );
  /* #67: @5 = (@5*@6) */
  w5 *= w6;
  /* #68: @5 = sin(@5) */
  w5 = sin( w5 );
  /* #69: @2 = (@2*@5) */
  w2 *= w5;
  /* #70: @7 = cos(@7) */
  w7 = cos( w7 );
  /* #71: @5 = (@3*@7) */
  w5  = (w3*w7);
  /* #72: @5 = (@2+@5) */
  w5  = (w2+w5);
  /* #73: @6 = 0.041 */
  w6 = 4.1000000000000002e-02;
  /* #74: @5 = (@5/@6) */
  w5 /= w6;
  /* #75: @0 = (@0*@1) */
  w0 *= w1;
  /* #76: @5 = (@5-@0) */
  w5 -= w0;
  /* #77: output[0][4] = @5 */
  if (res[0]) res[0][4] = w5;
  /* #78: @5 = 0.029 */
  w5 = 2.9000000000000001e-02;
  /* #79: @5 = (@5*@3) */
  w5 *= w3;
  /* #80: @5 = (@5*@7) */
  w5 *= w7;
  /* #81: @7 = 0.033 */
  w7 = 3.3000000000000002e-02;
  /* #82: @7 = (@7*@2) */
  w7 *= w2;
  /* #83: @5 = (@5-@7) */
  w5 -= w7;
  /* #84: @7 = 2.78e-05 */
  w7 = 2.7800000000000001e-05;
  /* #85: @5 = (@5/@7) */
  w5 /= w7;
  /* #86: output[0][5] = @5 */
  if (res[0]) res[0][5] = w5;
  /* #87: @5 = input[1][0] */
  w5 = arg[1] ? arg[1][0] : 0;
  /* #88: output[0][6] = @5 */
  if (res[0]) res[0][6] = w5;
  /* #89: @5 = input[1][1] */
  w5 = arg[1] ? arg[1][1] : 0;
  /* #90: output[0][7] = @5 */
  if (res[0]) res[0][7] = w5;
  return 0;
}

CASADI_SYMBOL_EXPORT int Path_parametric_MPC_expl_ode_fun(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int Path_parametric_MPC_expl_ode_fun_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int Path_parametric_MPC_expl_ode_fun_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void Path_parametric_MPC_expl_ode_fun_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int Path_parametric_MPC_expl_ode_fun_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void Path_parametric_MPC_expl_ode_fun_release(int mem) {
}

CASADI_SYMBOL_EXPORT void Path_parametric_MPC_expl_ode_fun_incref(void) {
}

CASADI_SYMBOL_EXPORT void Path_parametric_MPC_expl_ode_fun_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int Path_parametric_MPC_expl_ode_fun_n_in(void) { return 3;}

CASADI_SYMBOL_EXPORT casadi_int Path_parametric_MPC_expl_ode_fun_n_out(void) { return 1;}

CASADI_SYMBOL_EXPORT casadi_real Path_parametric_MPC_expl_ode_fun_default_in(casadi_int i){
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* Path_parametric_MPC_expl_ode_fun_name_in(casadi_int i){
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* Path_parametric_MPC_expl_ode_fun_name_out(casadi_int i){
  switch (i) {
    case 0: return "o0";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* Path_parametric_MPC_expl_ode_fun_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s1;
    case 2: return casadi_s2;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* Path_parametric_MPC_expl_ode_fun_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int Path_parametric_MPC_expl_ode_fun_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 5;
  if (sz_res) *sz_res = 2;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 10;
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif
