#include<stdio.h>
#include<math.h>
#include<stdlib.h>
#include<string.h>
#include<fftw3.h>

float  LL, rc, delta_nu, Lbyr;
int    N1, N2, N3;
float  pi=M_PI;
float  tpibyL, Lcube;
double vol, area;

float ***ro, **ro_2d, **ro_2da, *rpar; // for density
fftwf_plan p_ro,  q_ro; // for FFT
fftwf_plan p_2d,  q_2d;
fftwf_plan p_2da, q_2da; 
fftwf_plan p_1d,  q_1d; 

float  ***allocate_fftwf_3d(long N1, long N2, long N3);
double ***allocate_double_3d(int N1, int N2, int N3);
float  **allocate_fftwf_2d(long N1, long N2);
double **allocate_double_2d(int N1, int N2);
long   **allocate_long_2d(int N1, int N2); 
long   ***allocate_long_3d(int N1, int N2, int N3);

void   calpow_2d(int Nbin, double* kpar, double* kperp, long **no, double **power); // P(k_perp, k_par)
void   pk_2d_1d(int Nbin, double* kpar, double* kperp, long **no_2d, double **power_2d, long **no, double **power); // cl(\Delta \nu) --> P(k)
void   cl_nu12_EP(int Nbin, double *kperp, long **no_2d, double **power_2d);
void   cl_nu12(int Nbin, double *kperp, long ***no12, double ***power0, double ***power1);

void main()
{
  int ii, jj, kk;  
  int Nbin;
    
  double *kpar, *kperp, **power, ***power0, ***power1, **power_2d;
  long **no, ***no12, **no_2d;
  
  FILE  *inp, *outpp;
  char file[300], num[80];
  
  /****************************************************/

  inp=fopen("LC8.txt","r");

  fscanf(inp,"%d",&N1);
  fscanf(inp,"%d",&N2);
  fscanf(inp,"%d",&N3);
  
  printf("%d %d %d\n",N1,N2,N3);
  
  ro = allocate_fftwf_3d(N1,N2,N3+2);
  
  double roion=0.0;
  
  for(ii=0;ii<N1;ii++)
    for(jj=0;jj<N2;jj++)
      for(kk=0;kk<N3;kk++)
	{
          fscanf(inp,"%f", &ro[ii][jj][kk]); // check
	  roion+=(double)ro[ii][jj][kk]; // check
	}
  fclose(inp);
  
  roion/=(1.*N1*N2*N3); // mean HI density
  printf("Mean Tb = %2.4f\n",roion);
  
  /************ Input Parameters ***********************/

  Nbin = 10;
  
  LL = 500./256.; // grid spacing
  Lcube=LL*LL*LL;
  vol =Lcube*(1.0*N1*N2*N3);

  Lbyr=LL/9130.4; // delta-theta // z(7.206)=8881.6, z(8.909)=9379.2, rc=9130.4;
  delta_nu=0.1162; // delta-nu

  /****************************************************/

  tpibyL=2.0*pi/Lbyr;  // d(\Omega)   
  area= Lbyr*Lbyr*(1.0*N1*N2); // field of view
  
  /****************************************************/
  
  kpar = calloc((size_t)Nbin,sizeof(double));
  kperp = calloc((size_t)Nbin,sizeof(double));

  power = allocate_double_2d(Nbin,Nbin);
  power_2d = allocate_double_2d(Nbin,N3+2);
  power0 = allocate_double_3d(Nbin,N3,N3);
  power1 = allocate_double_3d(Nbin,N3,N3);
  
  no = allocate_long_2d(Nbin,Nbin);
  no_2d = allocate_long_2d(Nbin,N3+2);
  no12 = allocate_long_3d(Nbin,N3,N3);
  
  ro_2d = allocate_fftwf_2d(N1,N2+2);
  ro_2da = allocate_fftwf_2d(N1,N2+2);
  rpar = (float *) fftwf_malloc((N3+2)*sizeof(float));
  
  /****************************************************/
  /* Creating the plans for forward and reverse FFT's */
  
  p_ro = fftwf_plan_dft_r2c_3d(N1, N2, N3, &(ro[0][0][0]), (fftwf_complex*)&(ro[0][0][0]), FFTW_ESTIMATE);  
  q_ro = fftwf_plan_dft_c2r_3d(N1, N2, N3, (fftwf_complex*)&(ro[0][0][0]), &(ro[0][0][0]), FFTW_ESTIMATE);
  
  p_2d = fftwf_plan_dft_r2c_2d(N1, N2, &(ro_2d[0][0]), (fftwf_complex*)&(ro_2d[0][0]), FFTW_ESTIMATE);  
  q_2d = fftwf_plan_dft_c2r_2d(N1, N2, (fftwf_complex*)&(ro_2d[0][0]), &(ro_2d[0][0]), FFTW_ESTIMATE);
  
  p_2da = fftwf_plan_dft_r2c_2d(N1, N2, &(ro_2da[0][0]), (fftwf_complex*)&(ro_2da[0][0]), FFTW_ESTIMATE);  
  q_2da = fftwf_plan_dft_c2r_2d(N1, N2, (fftwf_complex*)&(ro_2da[0][0]), &(ro_2da[0][0]), FFTW_ESTIMATE);
  
  p_1d = fftwf_plan_dft_r2c_1d(N3, &(rpar[0]), (fftwf_complex*)&(rpar[0]), FFTW_ESTIMATE);  
  q_1d = fftwf_plan_dft_c2r_1d(N3, (fftwf_complex*)&(rpar[0]), &(rpar[0]), FFTW_ESTIMATE);
  
  /****************************************************/
  
  cl_nu12(Nbin, kperp, no12, power0, power1);
    
  /****************************************************/
  
  sprintf(file,"%s","cl_nu12");
  outpp=fopen(file,"w");
  
  for(kk=0;kk<Nbin;kk++)
    {
      for(ii=0;ii<N3;ii++)
	for(jj=0;jj<N3;jj++)
	  fprintf(outpp,"%e\n",power0[kk][ii][jj]*kperp[kk]*(kperp[kk]+1)/(2.0*pi));
      fprintf(outpp,"\n");
    }
  fclose(outpp);

  /****************************************************/
  
}


/****************************************************/

void cl_nu12(int Nbin, double *kperp, long ***no12, double ***power0, double ***power1)
{
  long i, j, k, l, a, b, c, *nperp12;
  int d, e;
  fftwf_complex *comp_ro, *comp_roa;
  float fac1, fac2, m, mu, scale;
  long index, index1;

  FILE  *inp, *outpp;
  char file[300], num[80];
  
  nperp12=calloc((size_t)Nbin,sizeof(long));
  
  fac1=1./(1.*N1*N1);
  fac2=1./(1.*N2*N2);
  
  scale=log10(0.5*N1)/(1.*Nbin);
  
  //------ Initialize Power spectrum modes --------------//

  for(d=0;d<Nbin;d++)
    {
      kperp[d]=0.0;
      nperp12[d]=0;

      for(i=0;i<N3;i++)
        for(j=0;j<N3;j++)
          {
	    power0[d][i][j]=0.0;
	    power1[d][i][j]=0.0;

            no12[d][i][j]=0;
          }
    }
  
  //------ Initialization done --------------// 

  for(k=0;k<N3;k++)
    {
      for(i=0;i<N1;i++)
	for(j=0;j<N2;j++)
	  ro_2d[i][j]=ro[i][j][k];
      
      fftwf_execute(p_2d);
      comp_ro = (fftwf_complex *)&(ro_2d[0][0]);
      
      for(l=0;l<N3;l++)
	{
	  for(i=0;i<N1;i++)
	    for(j=0;j<N2;j++)
	      ro_2da[i][j]=ro[i][j][l];
	  
          fftwf_execute(p_2da);
	  comp_roa = (fftwf_complex *)&(ro_2da[0][0]);
	  
	  for(i=0;i<N1;i++)
	    for(j=0;j<=N2/2;j++)
	      {
		index = i*(N2/2+1) + j;
		
                a=(i>N1/2)? N1-i: i;
                b=j;
		
		m =(float)sqrt(fac1*a*a + fac2*b*b);
		
		d=(int)floorf(log10(m*N1)/scale); // logarithmic k_perp bins, (0, 0) mean
		
		if(d>=0 && d<Nbin)
		  {
		    kperp[d]+= m;
		    nperp12[d]++;
		    
		    power0[d][k][l]+= 1.*((comp_ro[index][0]*comp_roa[index][0]) + (comp_ro[index][1]*comp_roa[index][1]));
		    power1[d][k][l]+= 1.*((comp_ro[index][1]*comp_roa[index][0]) - (comp_ro[index][0]*comp_roa[index][1]));
		    
		    no12[d][k][l]+= 1;
		  }
	      }
	}
    }
  
  //------------ the Power Spectrum -----------------------//
  
  for(d=0;d<Nbin;d++)
    if(nperp12[d]>0)
      {
        kperp[d] =(tpibyL*kperp[d])/(1.*nperp12[d]);
        printf("l=%5.2lf\n",kperp[d]);
      }
  
  for(d=0;d<Nbin;d++)
    for(i=0;i<N3;i++)
      for(j=0;j<N3;j++)
        if(no12[d][i][j]>0)
	  {
  	    power0[d][i][j]=power0[d][i][j]*Lbyr*Lbyr/(1.0*no12[d][i][j]*N1*N2);  // check
  	    power1[d][i][j]=power1[d][i][j]*Lbyr*Lbyr/(1.0*no12[d][i][j]*N1*N2);  // check
	  }
  
}

/****************************************************/

void cl_nu12_EP(int Nbin, double *kperp, long **no_2d, double **power_2d)
{
  long i, j, k, l, a, b;
  int d, e;
  fftwf_complex *comp_ro, *comp_roa;
  float fac1,fac2, m, scale;
  long index;
  
  FILE  *inp1;
  
  fac1=1./(1.*N1*N1);
  fac2=1./(1.*N2*N2);
  
  scale=log10(0.5*N1)/(1.*Nbin);
  
  //------ Initialize Power spectrum modes --------------// 
  
  for(i=0;i<Nbin;i++)
    for(j=0;j<N3+2;j++)
      {
	power_2d[i][j]=0.0;
	no_2d[i][j]=0;
      }
  
  /* //------------- Allocate memory for no of k-modes ----------// */
  
  double *nperp;
  nperp=calloc((size_t)Nbin,sizeof(double));
  
  for(i=0;i<Nbin;i++)
    {
      kperp[i]=0.0;
      nperp[i]=0.0;
    }
  
  //**************** Calculate the power spectrum *************************//
  
  for(k=0;k<N3;k++)
    {
      for(i=0;i<N1;i++)
  	for(j=0;j<N2;j++)
  	  ro_2d[i][j]=ro[i][j][k]*LL*LL;
      
      fftwf_execute(p_2d);
      comp_ro = (fftwf_complex *)&(ro_2d[0][0]);
      
      for(l=0;l<=k;l++)
  	{
  	  for(i=0;i<N1;i++)
  	    for(j=0;j<N2;j++)
  	      ro_2da[i][j]=ro[i][j][l]*LL*LL;
	  
          fftwf_execute(p_2da);
  	  comp_roa = (fftwf_complex *)&(ro_2da[0][0]);
	  
  	  for(i=0;i<N1;i++)
  	    for(j=0;j<=N2/2;j++)
  	      {
  		index = i*(N2/2+1) + j;
		
                a=(i>N1/2)? N1-i: i;
                b=j;
		
  		m =(float)sqrt(fac1*a*a + fac2*b*b);
		
  		d=(int)floorf(log10(m*N1)/scale); // logarithmic k_perp bins, (0, 0) mean
		
  		e=((k-l)>N3/2)? N3-(k-l) : (k-l);  // delta_nu
		
  		if(d==9)
  		  {
  		    kperp[d]+= m;
  		    nperp[d]+= 1.0;
		    
  		    power_2d[d][e]+= 1.*((comp_ro[index][0]*comp_roa[index][0]) + (comp_ro[index][1]*comp_roa[index][1]));
		    
  		    if((k-l)<=N3/2)
  		      power_2d[d][N3/2+1+e]+= 1.*((comp_ro[index][1]*comp_roa[index][0]) - (comp_ro[index][0]*comp_roa[index][1]));
  		    else
  		      power_2d[d][N3/2+1+e]+= 1.*((-comp_ro[index][1]*comp_roa[index][0]) + (comp_ro[index][0]*comp_roa[index][1]));
		    
  		    no_2d[d][e]+= 1;
  		    no_2d[d][N3/2+1+e]+= 1;
  		  }
  	      }
  	}
    }
  
  //------------ the Power Spectrum -----------------------//
  
  d=9;

  if(nperp[d]>0.0)
    kperp[d] =(tpibyL*kperp[d])/nperp[d];
  
  printf("l=%e\n",kperp[d]);
  
  for(j=0;j<=N3/2;j++)
    if(no_2d[d][j]>0)
      {
	power_2d[d][j]=power_2d[d][j]/(1.0*no_2d[d][j]*area);  // check
	power_2d[d][N3/2+1+j]=power_2d[d][N3/2+1+j]/(1.0*no_2d[d][N3/2+1+j]*area);  // check
      }
  
  /****************************************************/
  
  /* inp1=fopen("mean_cl_delta-nu","r"); */
  /* for(j=0;j<=N3/2;j++) */
  /*   fscanf(inp1,"%lf%lf",&power_2d[6][j],&power_2d[6][N3/2+1+j]); // check */
  /* fclose(inp1); */
  
  /* for(j=0;j<=N3/2;j++) */
  /*   printf("%e %e %e\n",1.0*j,power_2d[6][j],power_2d[6][N3/2+1+j]); */
  
  /****************************************************/

  for(i=0;i<N3;i++)
    for(j=0;j<N3;j++)
      {
        e=(abs(i-j)>N3/2)? N3-abs(i-j) : abs(i-j);  // delta_nu
        if((i-j)>=0)
          printf("%e %e\n",power_2d[d][e],power_2d[d][N3/2+1+e]);
        else
          printf("%e %e\n",power_2d[d][e],-1.0*power_2d[d][N3/2+1+e]);
      }

}

/****************************************************/

float  ***allocate_fftwf_3d(long N1,long N2,long N3)
{
  long ii,jj;
  long asize,index;
  float ***phia, *phi;
  
  phia=(float ***) fftwf_malloc (N1 *  sizeof(float **));
  
  
  for(ii=0;ii<N1;++ii)
    phia[ii]=(float **) fftwf_malloc (N2 *  sizeof(float *));
  
  asize = N1*N2;
  asize = asize*N3;
  
  if(!(phi = (float *) calloc(asize,sizeof(float))))
    {
      printf("error in allocate_fftwf_3d\n");
      exit(0);
    }
  
  for(ii=0;ii<N1;++ii)
    for(jj=0;jj<N2;++jj)
      {
	index = N2*N3;
	index = index*ii + N3*jj;
	phia[ii][jj]=phi+ index;
      }
  return(phia);
}

/****************************************************/

double  ***allocate_double_3d(int N1,int N2,int N3)
{
  int ii,jj;
  long asize,index;
  double ***phia, *phi;
  
  phia=(double ***) malloc (N1 *  sizeof(double **));
  
  
  for(ii=0;ii<N1;++ii)
    phia[ii]=(double **) malloc (N2 *  sizeof(double *));
  
  asize = N1*N2;
  asize = asize*N3;
  
  if(!(phi = (double *) calloc(asize,sizeof(double))))
    {
      printf("error in allocate_fftwf_3d\n");
      exit(0);
    }
  
  for(ii=0;ii<N1;++ii)
    for(jj=0;jj<N2;++jj)
      {
	index = N2*N3;
	index = index*ii + N3*jj;
	phia[ii][jj]=phi+ index;
      }
  return(phia);
}

/****************************************************/

float  **allocate_fftwf_2d(long N1,long N2)
{
  long ii,jj;
  long asize,index;
  float **phia, *phi;
  
  phia=(float **) fftwf_malloc (N1 *  sizeof(float *));
  
  asize = N1*N2;
  
  if(!(phi = (float *) calloc(asize,sizeof(float))))
    {
      printf("error in allocate_fftwf_2d\n");
      exit(0);
    }
  
  for(ii=0;ii<N1;++ii)
    {
      index = N2*ii;
      phia[ii]=phi+ index;
    }
  return(phia);
}

/****************************************************/

double **allocate_double_2d(int N1,int N2)
{
  double **xxa, *xx;
  int ii;
  
  xxa=(double**)malloc(N1 *  sizeof(double*));
  if(!(xx = (double *) calloc((size_t)(N1*N2),sizeof(double))))
    {
      printf("error in allocate_float_2d\n");
      exit(0);
    }
  
  for(ii=0;ii<N1;++ii)
    xxa[ii]=xx + N2*ii ;
  
  return(xxa);
}

/****************************************************/

long **allocate_long_2d(int N1,int N2) 
{
  long **xxa, *xx;
  int ii;
  
  xxa=(long**)malloc(N1 *  sizeof(long*));
  if(!(xx = (long *) calloc((size_t)(N1*N2),sizeof(long))))
    {
      printf("error in allocate_float_2d\n");
      exit(0);
    }
  
  for(ii=0;ii<N1;++ii)
    xxa[ii]=xx + N2*ii ;
  
  return(xxa);
}

/****************************************************/

long  ***allocate_long_3d(int N1,int N2,int N3)
{
  int ii,jj;
  long asize,index;
  long ***phia, *phi;
  
  phia=(long ***) malloc (N1 *  sizeof(long **));
  
  for(ii=0;ii<N1;++ii)
    phia[ii]=(long **) malloc (N2 *  sizeof(long *));
  
  asize = N1*N2;
  asize = asize*N3;
  
  if(!(phi = (long *) calloc(asize,sizeof(long))))
    {
      printf("error in allocate_fftwf_3d\n");
      exit(0);
    }
  
  for(ii=0;ii<N1;++ii)
    for(jj=0;jj<N2;++jj)
      {
	index = N2*N3;
	index = index*ii + N3*jj;
	phia[ii][jj]=phi+ index;
      }
  return(phia);
}

/****************************************************/

void calpow_2d(int Nbin, double* kpar, double* kperp, long **no, double **power) // P(k_perp, k_par)
{
  long i, j, k, a, b, c;
  int d, e;
  fftwf_complex *comp_ro;
  float fac1, fac2, m, mu, scale;
  long index,index1,index2;
  
  for(i=0;i<N1;i++)
    for(j=0;j<N2;j++)
      for(k=0;k<N3;k++)
	ro[i][j][k]=ro[i][j][k]*Lcube;
  
  fftwf_execute(p_ro);  // ro mow contains delta(k)
  
  comp_ro = (fftwf_complex *)&(ro[0][0][0]);
  
  /*********** TO FIND POWER SPECTRUM OF RO. **************/
  
  fac1=1./(1.*N1*N1);
  fac2=1./(1.*N2*N2);
  
  // dor logarithmic bins
  
  scale=log10(0.5*N3)/(1.*Nbin);
  
  //------ Initialize Power spectrum modes --------------// 
  
  for(i=0;i<Nbin;i++)
    for(j=0;j<Nbin;j++)
      {
	power[i][j]=0.0;
	no[i][j]=0;
      }
  
  //------------- Allocate memory for no of k-modes ----------// 
  
  double *npar, *nperp;
  npar=calloc((size_t)Nbin,sizeof(double));
  nperp=calloc((size_t)Nbin,sizeof(double));
  
  for(i=0;i<Nbin;i++)
    {
      kpar[i]=0.0;
      kperp[i]=0.0;
      
      npar[i]=0.0;
      nperp[i]=0.0;
    }
  
  //**************** Calculate the power spectrum *************************//
  
  /*-------------------------- mu=0------------------------- */
  
  for(i=0;i<N1;i++)
    for(j=0;j<N2/2;j++)   
      for(k=0;k<=N3/2;k=k+N3/2)
	{
          a=(i>N1/2)? N1-i: i;
	  b=j;
	  c=k;
	  
	  index = i*N2*(N3/2+1) + j*(N3/2+1) + k;
	  
	  m =(float) sqrt(fac1*a*a + fac2*b*b);     
	  mu = (float)(1.0*c)/(1.0*N3);    
	  
	  d=(int)floorf(log10(m*N3)/scale);  // logarithmic k_perp bins

	  if(k==0)
	    e=0;
	  else
	    e=(int)floorf(log10(mu*N3)/scale);  // k_par bins // along z-axis
	  
	  if(d>=0 && d<Nbin && e>=0 && e<Nbin)             
	    {
	      kperp[d]+= 1.0*m;
	      nperp[d]+= 1.0;
	      
	      kpar[e]+= 1.0*mu;
	      npar[e]+= 1.0;
	      
	      power[d][e]+= 1.*((comp_ro[index][0]*comp_ro[index][0]) + (comp_ro[index][1]*comp_ro[index][1]));
	      no[d][e]+= 1;
	    }
	}     
  
  /*-------------------------- mu!=0------------------------- */
  
  for(i=0;i<N1;i++)
    for(j=0;j<N2;j++)   
      for(k=1;k<N3/2;k++)
	{
          a=(i>N1/2)? N1-i: i;
          b=(j>N2/2)? N2-j: j;
	  c=k;
	  
	  index = i*N2*(N3/2+1) + j*(N3/2+1) + k;
	  
	  m =(float)sqrt(fac1*a*a + fac2*b*b);     
	  mu = (float)(1.0*c)/(N3);
	  
	  //if(i==0 && j==0)
	  //d=0;
	  //else
	  d=(int)floorf(log10(m*N3)/scale); //logarithmic k_perp bins
	  
	  e=(int)floorf(log10(mu*N3)/scale); //logarithmic k_par bins
	  
	  if(d>=0 && d<Nbin && e>=0 && e<Nbin)             
	    {
	      kperp[d]+= m;
	      nperp[d]+= 1.0;
	      
	      kpar[e]+= mu;
	      npar[e]+= 1.0;
	      
	      power[d][e]+= 1.*((comp_ro[index][0]*comp_ro[index][0]) + (comp_ro[index][1]*comp_ro[index][1]));
	      no[d][e]+= 1;
	    }
	  
	} 
  
  //------------ Print the Power Spectrum Moments -----------------------//  
  
  for(i=0;i<Nbin;i++)
    if(npar[i]>0.0)
      kpar[i] =(2.0*pi*kpar[i])/(LL*npar[i]); 
  
  for(i=0;i<Nbin;i++)   
    if(nperp[i]>0.0)
      kperp[i] =(2.0*pi*kperp[i])/(nperp[i]*16.7326*delta_nu);
  
  for(i=0;i<Nbin;i++)
    for(j=0;j<Nbin;j++)
      if(no[i][j]>0)
	power[i][j]=power[i][j]*16.7326*delta_nu*9151.53*9151.53/(1.0*no[i][j]*vol*LL*9301.61*9301.61*9301.61);

  for(i=0;i<Nbin;i++)
    for(j=0;j<Nbin;j++)
      {
	printf("%e %e\n", sqrt(kperp[i]*kperp[i]+kpar[j]*kpar[j]), power[i][j]);
      }
  
}

/****************************************************/

void pk_2d_1d(int Nbin, double* kpar, double* kperp, long **no_2d, double **power_2d, long **no, double **power) // cl(\Delta \nu) --> P(k)
{
  long i, j, k, l, a, b, c;
  int d, e;
  fftwf_complex *comp_ro, *comp_roa;
  float fac1, fac2, m, mu, scale;
  long index;
  
  /*********** TO FIND POWER SPECTRUM OF RO. **************/
  
  fac1=1./(1.*N1*N1);
  fac2=1./(1.*N2*N2);
  
  scale=log10(0.5*N3)/(1.*Nbin);
  
  //------ Initialize Power spectrum modes --------------// 
  
  for(i=0;i<Nbin;i++)
    for(j=0;j<N3+2;j++)
      {
	power_2d[i][j]=0.0;
	no_2d[i][j]=0;
      }
  
  //------------- Allocate memory for no of k-modes ----------//
  
  double *npar, *nperp;
  npar=calloc((size_t)Nbin,sizeof(double));
  nperp=calloc((size_t)Nbin,sizeof(double));
  
  for(i=0;i<Nbin;i++)
    {
      kpar[i]=0.0;
      kperp[i]=0.0;
      
      npar[i]=0.0;
      nperp[i]=0.0;
    }
  
  //**************** Calculate the power spectrum *************************//
  
  for(k=0;k<N3;k++)
    {
      for(i=0;i<N1;i++)
	for(j=0;j<N2;j++)
	  ro_2d[i][j]=ro[i][j][k]*LL*LL;
      
      fftwf_execute(p_2d);
      comp_ro = (fftwf_complex *)&(ro_2d[0][0]);
      
      for(l=0;l<=k;l++)
	{
	  for(i=0;i<N1;i++)
	    for(j=0;j<N2;j++)
	      ro_2da[i][j]=ro[i][j][l]*LL*LL;
	  
          fftwf_execute(p_2da);
	  comp_roa = (fftwf_complex *)&(ro_2da[0][0]);
	  
	  for(i=0;i<N1;i++)
	    for(j=0;j<=N2/2;j++)
	      {
		index = i*(N2/2+1) + j;
		
                a=(i>N1/2)? N1-i: i;
                b=j;
		
		m =(float)sqrt(fac1*a*a + fac2*b*b);
		
		//d=(int)floorf(log10(m*N1)/scale); // logarithmic k_perp bins, (0, 0) mean
		d=(int)floorf(log10(m*N3)/scale); // logarithmic k_perp bins, (0, 0) mean
		
		e=((k-l)>N3/2)? N3-(k-l) : (k-l);  // delta_nu
				  
		if(d>=0 && d<Nbin)
		  {
		    kperp[d]+= m;
		    nperp[d]+= 1.0;
		    
		    power_2d[d][e]+= 1.*((comp_ro[index][0]*comp_roa[index][0]) + (comp_ro[index][1]*comp_roa[index][1]));
		    
		    if((k-l)<=N3/2)
		      power_2d[d][N3/2+1+e]+= 1.*((comp_ro[index][1]*comp_roa[index][0]) - (comp_ro[index][0]*comp_roa[index][1]));
		    else
		      power_2d[d][N3/2+1+e]+= 1.*((-comp_ro[index][1]*comp_roa[index][0]) + (comp_ro[index][0]*comp_roa[index][1]));
		    
		    no_2d[d][e]+= 1;
		    no_2d[d][N3/2+1+e]+= 1;
		  }
	      }
	}
    }
  
  //------------ the Power Spectrum -----------------------//
  
  for(i=0;i<Nbin;i++)
    if(nperp[i]>0.0)
      kperp[i] =(tpibyL*kperp[i])/(nperp[i]*9301.61);
  
  for(i=0;i<Nbin;i++)
    for(j=0;j<=N3/2;j++)
      if(no_2d[i][j]>0)
	{
  	  power_2d[i][j]=power_2d[i][j]/(1.0*no_2d[i][j]*area);  // check
  	  power_2d[i][N3/2+1+j]=power_2d[i][N3/2+1+j]/(1.0*no_2d[i][N3/2+1+j]*area);  // check
	}
  
  /****************************************************/
  
  //for(i=0;i<Nbin;i++)
  //printf("%e %e\n",kperp[i], power_2d[i][0]);
  //printf("%e %e %e %e %e\n",kperp[i], power_2d[i][0], power_2d[i][89], power_2d[i][179], power_2d[i][259]);


  /* for(j=0;j<=N3/2;j++) */
  /*   { */
  /*     printf("%e ",j*1.0); */
  /*     for(i=0;i<Nbin;i++) */
  /* 	printf("%e %e ",power_2d[i][j],power_2d[i][N3/2+1+j]); */
  
  /*     printf("\n"); */
  /*   } */
  
  /****************************************************/
  
  for(i=0;i<Nbin;i++)
    {
      comp_ro = (fftwf_complex *)&(rpar[0]);
      
      for(j=0;j<=N3/2;j++) //check
	{
          comp_ro[j][0]=power_2d[i][j];
	  comp_ro[j][1]=-1.*power_2d[i][N3/2+1+j];
	  //comp_ro[j][1]=0.0;
	}
      
      comp_ro[0][1]=0.0;
      comp_ro[N3/2][1]=0.0;
      
      fftwf_execute(q_1d);  
      
      for(j=0;j<N3;j++)
	power_2d[i][j]=rpar[j];
    }
  
  /****************************************************/
  
  for(i=0;i<Nbin;i++)
    for(j=0;j<Nbin;j++)
      {
	power[i][j]=0.0;
	no[i][j]=0;
      }
  
  /****************************************************/
  
  scale=log10(0.5*N3)/(1.*Nbin);

  for(i=0;i<Nbin;i++)
    for(j=0;j<N3;j++)
      {
        a=(j>N3/2)? N3-j: j;
	mu = (float)(1.0*a)/(N3);    
	
	if(j==0)
	  e=0;
	else
	  e=(int)floorf(log10(mu*N3)/scale); // logarithmic k_par bins
	
	if(e>=0 && e<Nbin)
	  {
	    kpar[e]+=mu;
	    npar[e]+=1.0;
	    
	    power[i][e]+=(power_2d[i][j]); // check
	    no[i][e]+=1;
	  }
      }
  
  /****************************************************/
  
  for(i=0;i<Nbin;i++)
    if(npar[i]>0.0)
      kpar[i] =(2.0*pi*kpar[i])/(1.0*npar[i]*16.7326*delta_nu); 
  
  for(i=0;i<Nbin;i++)
    for(j=0;j<Nbin;j++)
      if(no[i][j]>0)
	power[i][j]=(power[i][j]*16.7326*delta_nu*9151.53*9151.53)/(1.0*no[i][j]); // check
  
  for(i=0;i<Nbin;i++)
    for(j=0;j<Nbin;j++)
      {
	printf("%e %e\n", sqrt(kperp[i]*kperp[i]+kpar[j]*kpar[j]), power[i][j]);
      }

}

/****************************************************/

  /* sprintf(file,"%s%5.2f","cl_nu1nu2_lc",kperp); // check */
  /* outpp=fopen(file,"w"); */
  
  /* for(ii=0;ii<N3;++ii) */
  /*   fprintf(outpp,", \" \""); */
  
  /* fprintf(outpp,"\n"); */
  /* for(ii=0;ii<N3;++ii) */
  /*   { */
  /*     fprintf(outpp,"\" \""); */
  /*     for(jj=0;jj<N3;++jj) */
  /* 	{ */
  /* 	  fprintf(outpp,", %e",power0[ii][jj]*kperp*(kperp+1)/(2.0*pi)); */
  /* 	} */
  /*     fprintf(outpp,"\n"); */
  /*   } */
  
  /* fclose(outpp); */
  /* printf("\n"); */
  
  /****************************************************/
  
  //pk_2d_1d(Nbin, kpar, kperp, no_2d, power_2d, no, power);
  
  //calpow_2d(Nbin, kpar, kperp, no, power);

  /****************************************************/
  
  /* for(ii=0;ii<Nbin;++ii) */
  /*   printf(", \"%3.2lf\"",kperp[ii]); */
  
  /* printf("\n"); */
  /* for(ii=0;ii<Nbin;++ii) */
  /*   { */
  /*     printf("\"%3.2lf\"",kpar[ii]); */
  /*     for(jj=0;jj<Nbin;++jj) */
  /* 	{ */
  /* 	  printf(", %e",power[ii][jj]); */
  /* 	} */
  /*     printf("\n"); */
  /*   } */
  /* printf("\n"); */
  
  /****************************************************/
  
  /* printf("{"); */
  /* for(ii=0;ii<Nbin;++ii) */
  /*   { */
  /*     if(ii==0) */
  /* 	printf("{"); */
  /*     else */
  /* 	printf(", {"); */
      
  /*     for(jj=0;jj<Nbin;++jj) */
  /* 	{ */
  /* 	  if(jj==0) */
  /* 	    printf("%e",power[ii][jj]); */
  /* 	  else */
  /* 	    printf(", %e",power[ii][jj]); */
  /* 	} */
  /*     printf("}"); */
  /*   } */
  /* printf("}"); */
  /* printf("\n"); */
  
  /****************************************************/
