/*
 * Copyright 2019 Gianluca Frison, Dimitris Kouzoupis, Robin Verschueren,
 * Andrea Zanelli, Niels van Duijkeren, Jonathan Frey, Tommaso Sartor,
 * Branimir Novoselnik, Rien Quirynen, Rezart Qelibari, Dang Doan,
 * Jonas Koenemann, Yutao Chen, Tobias Schöls, Jonas Schlagenhauf, Moritz Diehl
 *
 * This file is part of acados.
 *
 * The 2-Clause BSD License
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.;
 */

#ifndef ACADOS_SOLVER_Path_parametric_MPC_H_
#define ACADOS_SOLVER_Path_parametric_MPC_H_

#include "acados/utils/types.h"

#include "acados_c/ocp_nlp_interface.h"
#include "acados_c/external_function_interface.h"

#define PATH_PARAMETRIC_MPC_NX     8
#define PATH_PARAMETRIC_MPC_NZ     0
#define PATH_PARAMETRIC_MPC_NU     2
#define PATH_PARAMETRIC_MPC_NP     0
#define PATH_PARAMETRIC_MPC_NBX    2
#define PATH_PARAMETRIC_MPC_NBX0   8
#define PATH_PARAMETRIC_MPC_NBU    2
#define PATH_PARAMETRIC_MPC_NSBX   2
#define PATH_PARAMETRIC_MPC_NSBU   0
#define PATH_PARAMETRIC_MPC_NSH    0
#define PATH_PARAMETRIC_MPC_NSG    0
#define PATH_PARAMETRIC_MPC_NSPHI  0
#define PATH_PARAMETRIC_MPC_NSHN   0
#define PATH_PARAMETRIC_MPC_NSGN   0
#define PATH_PARAMETRIC_MPC_NSPHIN 0
#define PATH_PARAMETRIC_MPC_NSBXN  0
#define PATH_PARAMETRIC_MPC_NS     2
#define PATH_PARAMETRIC_MPC_NSN    0
#define PATH_PARAMETRIC_MPC_NG     0
#define PATH_PARAMETRIC_MPC_NBXN   0
#define PATH_PARAMETRIC_MPC_NGN    0
#define PATH_PARAMETRIC_MPC_NY0    10
#define PATH_PARAMETRIC_MPC_NY     10
#define PATH_PARAMETRIC_MPC_NYN    8
#define PATH_PARAMETRIC_MPC_N      30
#define PATH_PARAMETRIC_MPC_NH     0
#define PATH_PARAMETRIC_MPC_NPHI   0
#define PATH_PARAMETRIC_MPC_NHN    0
#define PATH_PARAMETRIC_MPC_NPHIN  0
#define PATH_PARAMETRIC_MPC_NR     0

#ifdef __cplusplus
extern "C" {
#endif


// ** capsule for solver data **
typedef struct Path_parametric_MPC_solver_capsule
{
    // acados objects
    ocp_nlp_in *nlp_in;
    ocp_nlp_out *nlp_out;
    ocp_nlp_out *sens_out;
    ocp_nlp_solver *nlp_solver;
    void *nlp_opts;
    ocp_nlp_plan_t *nlp_solver_plan;
    ocp_nlp_config *nlp_config;
    ocp_nlp_dims *nlp_dims;

    // number of expected runtime parameters
    unsigned int nlp_np;

    /* external functions */
    // dynamics

    external_function_param_casadi *forw_vde_casadi;
    external_function_param_casadi *expl_ode_fun;




    // cost






    // constraints




} Path_parametric_MPC_solver_capsule;

ACADOS_SYMBOL_EXPORT Path_parametric_MPC_solver_capsule * Path_parametric_MPC_acados_create_capsule(void);
ACADOS_SYMBOL_EXPORT int Path_parametric_MPC_acados_free_capsule(Path_parametric_MPC_solver_capsule *capsule);

ACADOS_SYMBOL_EXPORT int Path_parametric_MPC_acados_create(Path_parametric_MPC_solver_capsule * capsule);

ACADOS_SYMBOL_EXPORT int Path_parametric_MPC_acados_reset(Path_parametric_MPC_solver_capsule* capsule, int reset_qp_solver_mem);

/**
 * Generic version of Path_parametric_MPC_acados_create which allows to use a different number of shooting intervals than
 * the number used for code generation. If new_time_steps=NULL and n_time_steps matches the number used for code
 * generation, the time-steps from code generation is used.
 */
ACADOS_SYMBOL_EXPORT int Path_parametric_MPC_acados_create_with_discretization(Path_parametric_MPC_solver_capsule * capsule, int n_time_steps, double* new_time_steps);
/**
 * Update the time step vector. Number N must be identical to the currently set number of shooting nodes in the
 * nlp_solver_plan. Returns 0 if no error occurred and a otherwise a value other than 0.
 */
ACADOS_SYMBOL_EXPORT int Path_parametric_MPC_acados_update_time_steps(Path_parametric_MPC_solver_capsule * capsule, int N, double* new_time_steps);
/**
 * This function is used for updating an already initialized solver with a different number of qp_cond_N.
 */
ACADOS_SYMBOL_EXPORT int Path_parametric_MPC_acados_update_qp_solver_cond_N(Path_parametric_MPC_solver_capsule * capsule, int qp_solver_cond_N);
ACADOS_SYMBOL_EXPORT int Path_parametric_MPC_acados_update_params(Path_parametric_MPC_solver_capsule * capsule, int stage, double *value, int np);
ACADOS_SYMBOL_EXPORT int Path_parametric_MPC_acados_update_params_sparse(Path_parametric_MPC_solver_capsule * capsule, int stage, int *idx, double *p, int n_update);

ACADOS_SYMBOL_EXPORT int Path_parametric_MPC_acados_solve(Path_parametric_MPC_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT int Path_parametric_MPC_acados_free(Path_parametric_MPC_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT void Path_parametric_MPC_acados_print_stats(Path_parametric_MPC_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT int Path_parametric_MPC_acados_custom_update(Path_parametric_MPC_solver_capsule* capsule, double* data, int data_len);


ACADOS_SYMBOL_EXPORT ocp_nlp_in *Path_parametric_MPC_acados_get_nlp_in(Path_parametric_MPC_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT ocp_nlp_out *Path_parametric_MPC_acados_get_nlp_out(Path_parametric_MPC_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT ocp_nlp_out *Path_parametric_MPC_acados_get_sens_out(Path_parametric_MPC_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT ocp_nlp_solver *Path_parametric_MPC_acados_get_nlp_solver(Path_parametric_MPC_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT ocp_nlp_config *Path_parametric_MPC_acados_get_nlp_config(Path_parametric_MPC_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT void *Path_parametric_MPC_acados_get_nlp_opts(Path_parametric_MPC_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT ocp_nlp_dims *Path_parametric_MPC_acados_get_nlp_dims(Path_parametric_MPC_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT ocp_nlp_plan_t *Path_parametric_MPC_acados_get_nlp_plan(Path_parametric_MPC_solver_capsule * capsule);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif  // ACADOS_SOLVER_Path_parametric_MPC_H_
