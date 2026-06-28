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
	
	cost=cost/length;
	return cost;
}
int main(void){
	srand(69);
	//float parameter=randfloat()*10.0f;
	float parameter=1.0f;
	int length=sizeof(matrix)/sizeof matrix[0];
	float ep=1e-3;
	for (int j=0;j<5100;j++){
	parameter-=ep;
	}
	float cost=costfunc(length,parameter);
	printf("cost:%f \n",cost);
	printf("param:%f \n",parameter);
	return 0;
}



