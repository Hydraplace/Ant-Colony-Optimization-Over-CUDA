#include<iostream>
#include<fstream>
#include<math.h>
#include<stdlib.h>
using namespace std;
int main()
{
	int adj_size;
	int distance;
	cin>>adj_size;
	ofstream outfile;
	outfile.open("Matrix.txt");
	int adj_matrix[adj_size][adj_size];
	for(int i=0;i<adj_size;i++)
	{
		for(int j=0;j<adj_size;j++)
		{
			if(i!=j)
			{
				distance=(rand()%100)+1;
				adj_matrix[i][j]=distance;
				adj_matrix[j][i]=distance;
			}
			else
			{
				adj_matrix[i][j]=0;
			}
		}
	}
	outfile<<adj_size<<endl;
		for(int i=0;i<adj_size;i++)
	{
		for(int j=0;j<adj_size;j++)
		{
			outfile<<adj_matrix[i][j]<<" ";
			
			
		}
		outfile<<"\n";
	}
	outfile.close();
	return 0;
	
}
