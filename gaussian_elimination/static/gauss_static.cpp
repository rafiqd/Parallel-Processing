#include <iostream>
#include <fstream>
#include <pthread.h>
#include <cstdlib>
#include <sys/time.h>

int getPivotrow(double** matrix, int row, int col, int maxY, int sizeX);
void swapRows(double** matrix, int row1, int row2, int end);
void reduceRows(double** matrix, int row, int end, double pivotVal);
void reduceOtherRows(double** matrix, int baseRow, int changeRow, int xend, int colN);
void *paraReduceRow( void *ptr );

double **matrix;
int *tjobs;
int sizeX, sizeY;
int tcount;

struct Data{
	int id;
	int  topRow;
	int colN;
	int y;
	int sizeX;
	int sizeY;
	int start;
	int end;
};

int main(int argc, char* argv[]){
	
	double val;
	int i, j, x, y;
	std::ifstream file;
	
	tcount = atoi(argv[2]);
	timeval start, finish;
	
	file.open(argv[1]);
	
	if(file.is_open()){
		
		if((file >> sizeX >> sizeY >> val)){
			matrix = new double*[sizeY];
		}
		for(i = 0; i < sizeY; ++i){
			matrix[i] = new double[sizeX];
		}	
		
		while(file >> x >> y >> val){
			if(x && y){
				matrix[y-1][x-1] = val;
			}
		}
		
		
		double pivotVal;
		int pivotPos;
		
		int topRow = 0, row = 0, colN = 0;
		tjobs = new int[tcount];
		pthread_t threads[tcount];
		
		gettimeofday (&start, NULL);
		
		for( i = 0; i < sizeY; ++i, ++topRow, ++row){
		
			if(tcount > 1){
				// calculate how many jobs there are and which thread they go to
				for(j = 0; j < tcount; ++j){
					tjobs[j] = (sizeY-i) / tcount;
				}
			
				for(j = 0; j < (sizeY-i) % tcount; ++j){
					tjobs[j%tcount]++;
				}
			}
			
			pivotPos = getPivotrow(matrix, topRow, colN, topRow, sizeY);
			pivotVal = matrix[pivotPos][i];
			
			while(!pivotVal){
				++colN;
				++i;
				pivotPos = getPivotrow(matrix, topRow, colN, topRow, sizeY);
				pivotVal = matrix[pivotPos][i];
			}
			
			swapRows(matrix, topRow, pivotPos, sizeX);
			reduceRows(matrix, topRow, sizeX, pivotVal);
			
			
			// multithreaded stuff here
			if(tcount > 1){
				int startloc =0;
				int endloc = i;
				Data d;
				d.topRow = topRow;
				d.colN = colN;
				d.sizeX = sizeX;
				d.sizeY = sizeY;
				for(y = 0; y < tcount; ++y){
					if(tjobs[y]){
						startloc = endloc;
						d.id = y;
						d.start = startloc;
						endloc = startloc + (tjobs[y]);
						d.end = endloc;
						pthread_create( &threads[y], NULL, paraReduceRow, new Data(d));
					}
				}
				for(y = 0; y < tcount; ++y){
					if(tjobs[y]){
						pthread_join(threads[y], NULL);
					}
				}
				
			}else{
				//serial version here
				for(y = topRow+1; y < sizeY; ++y){
					reduceOtherRows(matrix, topRow, y, sizeX, colN);
				}
			}
			
			++colN;
		}
		
		gettimeofday (&finish, NULL);

		std::cout << "File: " << argv[1] << " cores: " << tcount << std::endl;
		printf ("Elapsed time: %.5f seconds\n",
	    (((finish.tv_sec * 1000000.0) + finish.tv_usec) -
	     ((start.tv_sec * 1000000.0) + start.tv_usec)) / 1000000.0);
		std::cout << std::endl;
		file.close();
	
	}
}
void *paraReduceRow( void *ptr ){
	Data d;
	int y;
	d = *(Data*)ptr;
	int start = d.start;
	int end = d.end;
	
	for(y = start; y < end; ++y){
		if(y != d.topRow){
			reduceOtherRows(matrix, d.topRow, y, d.sizeX, d.colN);
		}
	}
	return 0;
}

void reduceOtherRows(double** matrix, int baseRow, int changeRow, int xend, int colN){
	int i;
	double pivotVal = matrix[changeRow][colN];

	for(i = 0; i < xend; ++i){
		matrix[changeRow][i] = matrix[changeRow][i] - pivotVal * matrix[baseRow][i];
	}
}

void reduceRows(double** matrix, int row, int end, double pivotVal){
	for(int j = row; j < end; ++j){
		matrix[row][j] = matrix[row][j] / pivotVal;
	}
}

void swapRows(double** matrix, int row1, int row2, int end){
	int j;
	double temp;
	
	for(j = 0; j < end; ++j){
		temp = matrix[row1][j];
		matrix[row1][j] = matrix[row2][j];
		matrix[row2][j] = temp;
	}
}

int getPivotrow(double** matrix, int row, int col, int start, int end){
	
	int i, largest, largestIndex;
	largest = matrix[start][col];
	largestIndex = start;
	
	for(i = start; i < end; ++i){
		if( abs(matrix[i][col] )> abs(largest)){
			largest = matrix[i][col];
			largestIndex = i;
		}
	}
	return largestIndex;

}
