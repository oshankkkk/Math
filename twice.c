#include <stdio.h>
#include <stdlib.h>
#include <time.h>
int matrix [][2]={
	//{x, y}
	{1,2},
	{2,4},
	{3,6},
	{4,8},
	{5,10},
	{6,12},
};
//y =x*parameter

float randfloat(void){
	float num= (float)rand()/RAND_MAX;
	return num;
}

float costfunc(int length, float parameter){
	int x,y;
	float prediction, d, cost = 0.0f; 
	for (int i =0; i<length;++i){
		y=matrix[i][1];
		x=matrix[i][0];
		prediction=x*parameter;
		d=y-prediction;
		cost+= d*d;	
	}
	//MSE
	cost=cost/length;
	return cost;
}
int main(void){
	srand(99);
	float parameter=randfloat()*10.0f;
	//float parameter=1.0f;
	int length=sizeof(matrix)/sizeof matrix[0];
	float ep=1e-3;
	float bais=1e-4;
	float lrate=1e-3;
	float cost= 0.0f;
	do{
	//deravative cause we wanna know where the cost value decreases
    float diravtive = (costfunc(length, parameter + ep) - costfunc(length, parameter)) / ep;
    float baisdir = (costfunc(length, parameter + bais) - costfunc(length, parameter)) / ep;
	// we are reducing the direction it gives us
    parameter -= lrate * diravtive*baisdir;
    cost = costfunc(length, parameter);
	}while(cost>0.3f);
	printf("cost:%f \n",cost);
	printf("bais:%f \n",bais);
	printf("param:%f \n",parameter);
	return 0;
}



