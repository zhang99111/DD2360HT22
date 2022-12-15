#include "Particles.h"
#include "Alloc.h"
#include <cuda.h>
#include <cuda_runtime.h>

/** allocate particle arrays */
void particle_allocate(struct parameters* param, struct particles* part, int is)
{
    
    // set species ID
    part->species_ID = is;
    // number of particles
    part->nop = param->np[is];
    // maximum number of particles
    part->npmax = param->npMax[is];
    
    // choose a different number of mover iterations for ions and electrons
    if (param->qom[is] < 0){  //electrons
        part->NiterMover = param->NiterMover;
        part->n_sub_cycles = param->n_sub_cycles;
    } else {                  // ions: only one iteration
        part->NiterMover = 1;
        part->n_sub_cycles = 1;
    }
    
    // particles per cell
    part->npcelx = param->npcelx[is];
    part->npcely = param->npcely[is];
    part->npcelz = param->npcelz[is];
    part->npcel = part->npcelx*part->npcely*part->npcelz;
    
    // cast it to required precision
    part->qom = (FPpart) param->qom[is];
    
    long npmax = part->npmax;
    
    // initialize drift and thermal velocities
    // drift
    part->u0 = (FPpart) param->u0[is];
    part->v0 = (FPpart) param->v0[is];
    part->w0 = (FPpart) param->w0[is];
    // thermal
    part->uth = (FPpart) param->uth[is];
    part->vth = (FPpart) param->vth[is];
    part->wth = (FPpart) param->wth[is];
    
    
    //////////////////////////////
    /// ALLOCATION PARTICLE ARRAYS
    //////////////////////////////
    part->x = new FPpart[npmax];
    part->y = new FPpart[npmax];
    part->z = new FPpart[npmax];
    // allocate velocity
    part->u = new FPpart[npmax];
    part->v = new FPpart[npmax];
    part->w = new FPpart[npmax];
    // allocate charge = q * statistical weight
    part->q = new FPinterp[npmax];
    
}
/** deallocate */
void particle_deallocate(struct particles* part)
{
    // deallocate particle variables
    delete[] part->x;
    delete[] part->y;
    delete[] part->z;
    delete[] part->u;
    delete[] part->v;
    delete[] part->w;
    delete[] part->q;
}

/** particle mover */
int mover_PC(struct particles* part, struct EMfield* field, struct grid* grd, struct parameters* param)
{
    // print species and subcycling
    std::cout << "***  MOVER with SUBCYCLYING "<< param->n_sub_cycles << " - species " << part->species_ID << " ***" << std::endl;
 
    // auxiliary variables
    FPpart dt_sub_cycling = (FPpart) param->dt/((double) part->n_sub_cycles);
    FPpart dto2 = .5*dt_sub_cycling, qomdt2 = part->qom*dto2/param->c;
    FPpart omdtsq, denom, ut, vt, wt, udotb;
    
    // local (to the particle) electric and magnetic field
    FPfield Exl=0.0, Eyl=0.0, Ezl=0.0, Bxl=0.0, Byl=0.0, Bzl=0.0;
    
    // interpolation densities
    int ix,iy,iz;
    FPfield weight[2][2][2];
    FPfield xi[2], eta[2], zeta[2];
    
    // intermediate particle position and velocity
    FPpart xptilde, yptilde, zptilde, uptilde, vptilde, wptilde;
    
    // start subcycling
    for (int i_sub=0; i_sub <  part->n_sub_cycles; i_sub++){
        // move each particle with new fields
        for (int i=0; i <  part->nop; i++){
            xptilde = part->x[i];
            yptilde = part->y[i];
            zptilde = part->z[i];
            // calculate the average velocity iteratively
            for(int innter=0; innter < part->NiterMover; innter++){
                // interpolation G-->P
                ix = 2 +  int((part->x[i] - grd->xStart)*grd->invdx);
                iy = 2 +  int((part->y[i] - grd->yStart)*grd->invdy);
                iz = 2 +  int((part->z[i] - grd->zStart)*grd->invdz);
                
                // calculate weights
                xi[0]   = part->x[i] - grd->XN[ix - 1][iy][iz];
                eta[0]  = part->y[i] - grd->YN[ix][iy - 1][iz];
                zeta[0] = part->z[i] - grd->ZN[ix][iy][iz - 1];
                xi[1]   = grd->XN[ix][iy][iz] - part->x[i];
                eta[1]  = grd->YN[ix][iy][iz] - part->y[i];
                zeta[1] = grd->ZN[ix][iy][iz] - part->z[i];
                for (int ii = 0; ii < 2; ii++)
                    for (int jj = 0; jj < 2; jj++)
                        for (int kk = 0; kk < 2; kk++)
                            weight[ii][jj][kk] = xi[ii] * eta[jj] * zeta[kk] * grd->invVOL;
                
                // set to zero local electric and magnetic field
                Exl=0.0, Eyl = 0.0, Ezl = 0.0, Bxl = 0.0, Byl = 0.0, Bzl = 0.0;
                
                for (int ii=0; ii < 2; ii++)
                    for (int jj=0; jj < 2; jj++)
                        for(int kk=0; kk < 2; kk++){
                            Exl += weight[ii][jj][kk]*field->Ex[ix- ii][iy -jj][iz- kk ];
                            Eyl += weight[ii][jj][kk]*field->Ey[ix- ii][iy -jj][iz- kk ];
                            Ezl += weight[ii][jj][kk]*field->Ez[ix- ii][iy -jj][iz -kk ];
                            Bxl += weight[ii][jj][kk]*field->Bxn[ix- ii][iy -jj][iz -kk ];
                            Byl += weight[ii][jj][kk]*field->Byn[ix- ii][iy -jj][iz -kk ];
                            Bzl += weight[ii][jj][kk]*field->Bzn[ix- ii][iy -jj][iz -kk ];
                        }
                
                // end interpolation
                omdtsq = qomdt2*qomdt2*(Bxl*Bxl+Byl*Byl+Bzl*Bzl);
                denom = 1.0/(1.0 + omdtsq);
                // solve the position equation
                ut= part->u[i] + qomdt2*Exl;
                vt= part->v[i] + qomdt2*Eyl;
                wt= part->w[i] + qomdt2*Ezl;
                udotb = ut*Bxl + vt*Byl + wt*Bzl;
                // solve the velocity equation
                uptilde = (ut+qomdt2*(vt*Bzl -wt*Byl + qomdt2*udotb*Bxl))*denom;
                vptilde = (vt+qomdt2*(wt*Bxl -ut*Bzl + qomdt2*udotb*Byl))*denom;
                wptilde = (wt+qomdt2*(ut*Byl -vt*Bxl + qomdt2*udotb*Bzl))*denom;
                // update position
                part->x[i] = xptilde + uptilde*dto2;
                part->y[i] = yptilde + vptilde*dto2;
                part->z[i] = zptilde + wptilde*dto2;
                
                
            } // end of iteration
            // update the final position and velocity
            part->u[i]= 2.0*uptilde - part->u[i];
            part->v[i]= 2.0*vptilde - part->v[i];
            part->w[i]= 2.0*wptilde - part->w[i];
            part->x[i] = xptilde + uptilde*dt_sub_cycling;
            part->y[i] = yptilde + vptilde*dt_sub_cycling;
            part->z[i] = zptilde + wptilde*dt_sub_cycling;
            
            
            //////////
            //////////
            ////////// BC
                                        
            // X-DIRECTION: BC particles
            if (part->x[i] > grd->Lx){
                if (param->PERIODICX==true){ // PERIODIC
                    part->x[i] = part->x[i] - grd->Lx;
                } else { // REFLECTING BC
                    part->u[i] = -part->u[i];
                    part->x[i] = 2*grd->Lx - part->x[i];
                }
            }
                                                                        
            if (part->x[i] < 0){
                if (param->PERIODICX==true){ // PERIODIC
                   part->x[i] = part->x[i] + grd->Lx;
                } else { // REFLECTING BC
                    part->u[i] = -part->u[i];
                    part->x[i] = -part->x[i];
                }
            }
                
            
            // Y-DIRECTION: BC particles
            if (part->y[i] > grd->Ly){
                if (param->PERIODICY==true){ // PERIODIC
                    part->y[i] = part->y[i] - grd->Ly;
                } else { // REFLECTING BC
                    part->v[i] = -part->v[i];
                    part->y[i] = 2*grd->Ly - part->y[i];
                }
            }
                                                                        
            if (part->y[i] < 0){
                if (param->PERIODICY==true){ // PERIODIC
                    part->y[i] = part->y[i] + grd->Ly;
                } else { // REFLECTING BC
                    part->v[i] = -part->v[i];
                    part->y[i] = -part->y[i];
                }
            }
                                                                        
            // Z-DIRECTION: BC particles
            if (part->z[i] > grd->Lz){
                if (param->PERIODICZ==true){ // PERIODIC
                    part->z[i] = part->z[i] - grd->Lz;
                } else { // REFLECTING BC
                    part->w[i] = -part->w[i];
                    part->z[i] = 2*grd->Lz - part->z[i];
                }
            }
                                                                        
            if (part->z[i] < 0){
                if (param->PERIODICZ==true){ // PERIODIC
                    part->z[i] = part->z[i] + grd->Lz;
                } else { // REFLECTING BC
                    part->w[i] = -part->w[i];
                    part->z[i] = -part->z[i];
                }
            }
                                                                        
            
            
        }  // end of subcycling
    } // end of one particle
                                                                        
    return(0); // exit succcesfully
} // end of the mover
__global__ void move_particles_gpu(particles* dPart, EMfield* dField, grid* dGrd, parameters* dParam)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dPart->nop) return;
    FPpart dt_sub_cycling = (FPpart) dParam->dt/((double) dPart->n_sub_cycles);
    FPpart dto2 = .5*dt_sub_cycling, qomdt2 = dPart->qom*dto2/dParam->c;
    FPpart omdtsq, denom, ut, vt, wt, udotb;

    FPfield Exl=0.0, Eyl=0.0, Ezl=0.0, Bxl=0.0, Byl=0.0, Bzl=0.0;
    
    // interpolation densities
    int ix,iy,iz;
    FPfield weight[2][2][2];
    FPfield xi[2], eta[2], zeta[2];
    
    // intermediate particle position and velocity
    FPpart xptilde, yptilde, zptilde, uptilde, vptilde, wptilde;
      for (int i_sub=0; i_sub <  dPart->n_sub_cycles; i_sub++){
        // move each particle with new fields
        //for (int i=0; i <  part->nop; i++){
            xptilde = dPart->x[idx];
            yptilde = dPart->y[idx];
            zptilde = dPart->z[idx];
            // calculate the average velocity iteratively
            for(int innter=0; innter < dPart->NiterMover; innter++){
                // interpolation G-->P
                ix = 2 +  int((dPart->x[idx] - dGrd->xStart)*dGrd->invdx);
                iy = 2 +  int((dPart->y[idx] - dGrd->yStart)*dGrd->invdy);
                iz = 2 +  int((dPart->z[idx] - dGrd->zStart)*dGrd->invdz);
                
                // calculate weights
                xi[0]   = dPart->x[idx] - dGrd->XN[ix - 1][iy][iz];
                eta[0]  = dPart->y[idx] - dGrd->YN[ix][iy - 1][iz];
                zeta[0] = dPart->z[idx] - dGrd->ZN[ix][iy][iz - 1];
                xi[1]   = dGrd->XN[ix][iy][iz] - dPart->x[idx];
                eta[1]  = dGrd->YN[ix][iy][iz] - dPart->y[idx];
                zeta[1] = dGrd->ZN[ix][iy][iz] - dPart->z[idx];
                for (int ii = 0; ii < 2; ii++)
                    for (int jj = 0; jj < 2; jj++)
                        for (int kk = 0; kk < 2; kk++)
                            weight[ii][jj][kk] = xi[ii] * eta[jj] * zeta[kk] * dGrd->invVOL;
                
                // set to zero local electric and magnetic field
                Exl=0.0, Eyl = 0.0, Ezl = 0.0, Bxl = 0.0, Byl = 0.0, Bzl = 0.0;
                
                for (int ii=0; ii < 2; ii++)
                    for (int jj=0; jj < 2; jj++)
                        for(int kk=0; kk < 2; kk++){
                            Exl += weight[ii][jj][kk]*dField->Ex[ix- ii][iy -jj][iz- kk ];
                            Eyl += weight[ii][jj][kk]*dField->Ey[ix- ii][iy -jj][iz- kk ];
                            Ezl += weight[ii][jj][kk]*dField->Ez[ix- ii][iy -jj][iz -kk ];
                            Bxl += weight[ii][jj][kk]*dField->Bxn[ix- ii][iy -jj][iz -kk ];
                            Byl += weight[ii][jj][kk]*dField->Byn[ix- ii][iy -jj][iz -kk ];
                            Bzl += weight[ii][jj][kk]*dField->Bzn[ix- ii][iy -jj][iz -kk ];
                        }
                
                // end interpolation
                omdtsq = qomdt2*qomdt2*(Bxl*Bxl+Byl*Byl+Bzl*Bzl);
                denom = 1.0/(1.0 + omdtsq);
                // solve the position equation
                ut= dPart->u[idx] + qomdt2*Exl;
                vt= dPart->v[idx] + qomdt2*Eyl;
                wt= dPart->w[idx] + qomdt2*Ezl;
                udotb = ut*Bxl + vt*Byl + wt*Bzl;
                // solve the velocity equation
                uptilde = (ut+qomdt2*(vt*Bzl -wt*Byl + qomdt2*udotb*Bxl))*denom;
                vptilde = (vt+qomdt2*(wt*Bxl -ut*Bzl + qomdt2*udotb*Byl))*denom;
                wptilde = (wt+qomdt2*(ut*Byl -vt*Bxl + qomdt2*udotb*Bzl))*denom;
                // update position
                dPart->x[idx] = xptilde + uptilde*dto2;
                dPart->y[idx] = yptilde + vptilde*dto2;
                dPart->z[idx] = zptilde + wptilde*dto2;
                
                
            } // end of iteration
            // update the final position and velocity
            dPart->u[idx]= 2.0*uptilde - dPart->u[idx];
            dPart->v[idx]= 2.0*vptilde - dPart->v[idx];
            dPart->w[idx]= 2.0*wptilde - dPart->w[idx];
            dPart->x[idx] = xptilde + uptilde*dt_sub_cycling;
            dPart->y[idx] = yptilde + vptilde*dt_sub_cycling;
            dPart->z[idx] = zptilde + wptilde*dt_sub_cycling;
            
            
            //////////
            //////////
            ////////// BC
                                        
            // X-DIRECTION: BC particles
            if (dPart->x[idx] > dGrd->Lx){
                if (dParam->PERIODICX==true){ // PERIODIC
                    dPart->x[idx] = dPart->x[idx] - dGrd->Lx;
                } else { // REFLECTING BC
                    dPart->u[idx] = -dPart->u[idx];
                    dPart->x[idx] = 2*dGrd->Lx - dPart->x[idx];
                }
            }
                                                                        
            if (dPart->x[idx] < 0){
                if (dParam->PERIODICX==true){ // PERIODIC
                   dPart->x[idx] = dPart->x[idx] + dGrd->Lx;
                } else { // REFLECTING BC
                    dPart->u[idx] = -dPart->u[idx];
                    dPart->x[idx] = -dPart->x[idx];
                }
            }
                
            
            // Y-DIRECTION: BC particles
            if (dPart->y[idx] > dGrd->Ly){
                if (dParam->PERIODICY==true){ // PERIODIC
                   dPart->y[idx] = dPart->y[idx] - dGrd->Ly;
                } else { // REFLECTING BC
                    dPart->v[idx] = -dPart->v[idx];
                    dPart->y[idx] = 2*dGrd->Ly - dPart->y[idx];
                }
            }
                                                                        
            if (dPart->y[idx] < 0){
                if (dParam->PERIODICY==true){ // PERIODIC
                    dPart->y[idx] = dPart->y[idx] + dGrd->Ly;
                } else { // REFLECTING BC
                    dPart->v[idx] = -dPart->v[idx];
                    dPart->y[idx] = -dPart->y[idx];
                }
            }
                                                                        
            // Z-DIRECTION: BC particles
            if (dPart->z[idx] > dGrd->Lz){
                if (dParam->PERIODICZ==true){ // PERIODIC
                    dPart->z[idx] = dPart->z[idx] - dGrd->Lz;
                } else { // REFLECTING BC
                    dPart->w[idx] = -dPart->w[idx];
                    dPart->z[idx] = 2*dGrd->Lz - dPart->z[idx];
                }
            }
                                                                        
            if (dPart->z[idx] < 0){
                if (dParam->PERIODICZ==true){ // PERIODIC
                    dPart->z[idx] = dPart->z[idx] + dGrd->Lz;
                } else { // REFLECTING BC
                    dPart->w[idx] = -dPart->w[idx];
                    dPart->z[idx] = -dPart->z[idx];
                }
            }
                                                                        
            
            
        }  // end of subcycling
    } // end of one particle
                                                                        
   

int mover_PC_gpu(struct particles* part, struct EMfield* field, struct grid* grd, struct parameters* param)
{
    particles* dPart;
    EMfield* dField;
    grid* dGrd;
    parameters* dParam;
    cudaMalloc(&dPart, sizeof(particles));
    cudaMalloc(&dField, sizeof(EMfield));
    cudaMalloc(&dGrd, sizeof(grid));
    cudaMalloc(&dParam, sizeof(parameters));
    
    cudaMemcpy(dPart, part, sizeof(particles), cudaMemcpyHostToDevice);
    cudaMemcpy(dField, field, sizeof(EMfield), cudaMemcpyHostToDevice);
    cudaMemcpy(dGrd, grd, sizeof(grid), cudaMemcpyHostToDevice);
    cudaMemcpy(dParam, param, sizeof(parameters), cudaMemcpyHostToDevice);
    
    FPpart *dPart_x, *dPart_y, *dPart_z, *dPart_u, *dPart_v, *dPart_w;
    cudaMalloc(&dPart_x, part->npmax * sizeof(FPpart));
    cudaMalloc(&dPart_y, part->npmax * sizeof(FPpart));
    cudaMalloc(&dPart_z, part->npmax * sizeof(FPpart));
    cudaMalloc(&dPart_u, part->npmax * sizeof(FPpart));
    cudaMalloc(&dPart_v, part->npmax * sizeof(FPpart));
    cudaMalloc(&dPart_w, part->npmax * sizeof(FPpart));

    cudaMemcpy(dPart_x, part->x, part->npmax * sizeof(FPpart), cudaMemcpyHostToDevice);
    cudaMemcpy(dPart_y, part->y, part->npmax * sizeof(FPpart), cudaMemcpyHostToDevice);
    cudaMemcpy(dPart_z, part->z, part->npmax * sizeof(FPpart), cudaMemcpyHostToDevice);
    cudaMemcpy(dPart_u, part->u, part->npmax * sizeof(FPpart), cudaMemcpyHostToDevice);
    cudaMemcpy(dPart_v, part->v, part->npmax * sizeof(FPpart), cudaMemcpyHostToDevice);
    cudaMemcpy(dPart_w, part->w, part->npmax * sizeof(FPpart), cudaMemcpyHostToDevice);
    
    cudaMemcpy(&(dPart->x), &dPart_x, sizeof(FPpart*), cudaMemcpyHostToDevice);
    cudaMemcpy(&(dPart->y), &dPart_y, sizeof(FPpart*), cudaMemcpyHostToDevice);
    cudaMemcpy(&(dPart->z), &dPart_z, sizeof(FPpart*), cudaMemcpyHostToDevice);
    cudaMemcpy(&(dPart->u), &dPart_u, sizeof(FPpart*), cudaMemcpyHostToDevice);
    cudaMemcpy(&(dPart->v), &dPart_v, sizeof(FPpart*), cudaMemcpyHostToDevice);
    cudaMemcpy(&(dPart->w), &dPart_w, sizeof(FPpart*), cudaMemcpyHostToDevice);
    
     FPfield *dField_Ex_flat, *dField_Ey_flat, *dField_Ez_flat, *dField_Bxn_flat, *dField_Byn_flat, *dField_Bzn_flat;
    cudaMalloc(&dField_Ex_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield));
    cudaMalloc(&dField_Ey_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield));
    cudaMalloc(&dField_Ez_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield));
    cudaMalloc(&dField_Bxn_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield));
    cudaMalloc(&dField_Byn_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield));
    cudaMalloc(&dField_Bzn_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield));

    cudaMemcpy(dField_Ex_flat, field->Ex_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy(dField_Ey_flat, field->Ey_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy(dField_Ez_flat, field->Ez_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy(dField_Bxn_flat, field->Bxn_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy(dField_Byn_flat, field->Byn_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy(dField_Bzn_flat, field->Bzn_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);
    
    cudaMemcpy(&(dField->Ex_flat), &dField_Ex_flat, sizeof(FPfield*), cudaMemcpyHostToDevice);
    cudaMemcpy(&(dField->Ey_flat), &dField_Ey_flat, sizeof(FPfield*), cudaMemcpyHostToDevice);
    cudaMemcpy(&(dField->Ez_flat), &dField_Ez_flat, sizeof(FPfield*), cudaMemcpyHostToDevice);
    cudaMemcpy(&(dField->Bxn_flat), &dField_Bxn_flat, sizeof(FPfield*), cudaMemcpyHostToDevice);
    cudaMemcpy(&(dField->Byn_flat), &dField_Byn_flat, sizeof(FPfield*), cudaMemcpyHostToDevice);
    cudaMemcpy(&(dField->Bzn_flat), &dField_Bzn_flat, sizeof(FPfield*), cudaMemcpyHostToDevice);
    
    
     FPfield *dGrd_XN_flat, *dGrd_YN_flat, *dGrd_ZN_flat;
    cudaMalloc(&dGrd_XN_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield));
    cudaMalloc(&dGrd_YN_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield));
    cudaMalloc(&dGrd_ZN_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield));

    cudaMemcpy(dGrd_XN_flat, grd->XN_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy(dGrd_YN_flat, grd->YN_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy(dGrd_ZN_flat, grd->ZN_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);
    
    
    cudaMemcpy(&(dGrd->XN_flat), &dGrd_XN_flat, sizeof(FPfield*), cudaMemcpyHostToDevice);
    cudaMemcpy(&(dGrd->YN_flat), &dGrd_YN_flat, sizeof(FPfield*), cudaMemcpyHostToDevice);
    cudaMemcpy(&(dGrd->ZN_flat), &dGrd_ZN_flat, sizeof(FPfield*), cudaMemcpyHostToDevice);
    
    int Db_1=64;
    int Dg_1=(part->nop + Db_1 - 1) / Db_1;
    std::cout << "***  MOVER with SUBCYCLYING(gpu  version) "<< param->n_sub_cycles << " - species " << part->species_ID << " ***" << std::endl;
     move_particles_gpu<<<Dg_1, Db_1>>>(dPart, dField, dGrd, dParam);
     
     cudaMemcpy(part, dPart, sizeof(particles), cudaMemcpyDeviceToHost);
     
     
    FPpart *hostPart_x, *hostPart_y, *hostPart_z, *hostPart_u, *hostPart_v, *hostPart_w;
    hostPart_x = (FPpart*) malloc(part->npmax * sizeof(FPpart));
    hostPart_y= (FPpart*) malloc(part->npmax * sizeof(FPpart));
    hostPart_z = (FPpart*) malloc(part->npmax * sizeof(FPpart));
    hostPart_u = (FPpart*) malloc(part->npmax * sizeof(FPpart));
    hostPart_v = (FPpart*) malloc(part->npmax * sizeof(FPpart));
    hostPart_w = (FPpart*) malloc(part->npmax * sizeof(FPpart));

    cudaMemcpy(hostPart_x, dPart_x, part->npmax * sizeof(FPpart), cudaMemcpyDeviceToHost);
    cudaMemcpy(hostPart_y, dPart_y, part->npmax * sizeof(FPpart), cudaMemcpyDeviceToHost);
    cudaMemcpy(hostPart_z, dPart_z, part->npmax * sizeof(FPpart), cudaMemcpyDeviceToHost);
    cudaMemcpy(hostPart_u, dPart_u, part->npmax * sizeof(FPpart), cudaMemcpyDeviceToHost);
    cudaMemcpy(hostPart_v, dPart_v, part->npmax * sizeof(FPpart), cudaMemcpyDeviceToHost);
    cudaMemcpy(hostPart_w, dPart_w, part->npmax * sizeof(FPpart), cudaMemcpyDeviceToHost);

    // Binding pointers with part struct
    part->x = hostPart_x;
    part->y = hostPart_y;
    part->z = hostPart_z;
    part->u = hostPart_u;
    part->v = hostPart_v;
    part->w = hostPart_w;

    //@@ Free the GPU memory here
    cudaFree(dPart);
    cudaFree(dField);
    cudaFree(dGrd);
    cudaFree(dParam);                                                               

    return(0); // exit succcesfully
} // end of the mover


/** Interpolation Particle --> Grid: This is for species */
void interpP2G(struct particles* part, struct interpDensSpecies* ids, struct grid* grd)
{
    
    // arrays needed for interpolation
    FPpart weight[2][2][2];
    FPpart temp[2][2][2];
    FPpart xi[2], eta[2], zeta[2];
    
    // index of the cell
    int ix, iy, iz;
    
    
    for (register long long i = 0; i < part->nop; i++) {
        
        // determine cell: can we change to int()? is it faster?
        ix = 2 + int (floor((part->x[i] - grd->xStart) * grd->invdx));
        iy = 2 + int (floor((part->y[i] - grd->yStart) * grd->invdy));
        iz = 2 + int (floor((part->z[i] - grd->zStart) * grd->invdz));
        
        // distances from node
        xi[0]   = part->x[i] - grd->XN[ix - 1][iy][iz];
        eta[0]  = part->y[i] - grd->YN[ix][iy - 1][iz];
        zeta[0] = part->z[i] - grd->ZN[ix][iy][iz - 1];
        xi[1]   = grd->XN[ix][iy][iz] - part->x[i];
        eta[1]  = grd->YN[ix][iy][iz] - part->y[i];
        zeta[1] = grd->ZN[ix][iy][iz] - part->z[i];
        
        // calculate the weights for different nodes
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    weight[ii][jj][kk] = part->q[i] * xi[ii] * eta[jj] * zeta[kk] * grd->invVOL;
        
        //////////////////////////
        // add charge density
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->rhon[ix - ii][iy - jj][iz - kk] += weight[ii][jj][kk] * grd->invVOL;
        
        
        ////////////////////////////
        // add current density - Jx
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[i] * weight[ii][jj][kk];
        
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->Jx[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        ////////////////////////////
        // add current density - Jy
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->v[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->Jy[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        
        ////////////////////////////
        // add current density - Jz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->w[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->Jz[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        ////////////////////////////
        // add pressure pxx
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[i] * part->u[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pxx[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        ////////////////////////////
        // add pressure pxy
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[i] * part->v[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pxy[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        
        /////////////////////////////
        // add pressure pxz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[i] * part->w[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pxz[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        /////////////////////////////
        // add pressure pyy
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->v[i] * part->v[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pyy[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        /////////////////////////////
        // add pressure pyz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->v[i] * part->w[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pyz[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        /////////////////////////////
        // add pressure pzz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->w[i] * part->w[i] * weight[ii][jj][kk];
        for (int ii=0; ii < 2; ii++)
            for (int jj=0; jj < 2; jj++)
                for(int kk=0; kk < 2; kk++)
                    ids->pzz[ix -ii][iy -jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
    
    }
   
}
