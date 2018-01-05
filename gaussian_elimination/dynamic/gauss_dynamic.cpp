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
void computeGauss();

double **matrix;
int *tjobs;
int sizeX, sizeY;
int tcount;
int jobs;
int counter;
pthread_mutex_t lock;

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
	int i, x, y;
	std::ifstream file;
	tcount = atoi(argv[2]);
	timeval start, finish;;
	
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
		
		gettimeofday (&start, NULL);
		computeGauss();
		gettimeofday (&finish, NULL);
		// GE ends here -------------------
		
		/*
		for(y = 0; y < sizeY; ++y){
			for(x = 0; x < sizeX; ++x){
				std::cout << matrix[y][x] << " ";
			}
			std::cout << std::endl;
		}
		*/
		std::cout << "File: " << argv[1] << " cores: " << tcount << std::endl;
		printf ("Elapsed time: %.5f seconds\n",
	    (((finish.tv_sec * 1000000.0) + finish.tv_usec) -
	     ((start.tv_sec * 1000000.0) + start.tv_usec)) / 1000000.0);
		std::cout << std::endl;
		pthread_mutex_destroy(&lock);
		file.close();
	
	}
}

void computeGauss(){

		double pivotVal;
		int pivotPos;
		int topRow = 0, row = 0, colN = 0;
		pthread_t threads[tcount];
		int i, y, j;
		// GE starts here ----------------
		
		pthread_mutex_init(&lock, NULL); // mutex
		
		for( i = 0; i < sizeY; ++i, ++topRow, ++row){
		
			counter = 0;			
			pivotPos = getPivotrow(matrix, topRow, colN, topRow, sizeY);
			pivotVal = matrix[pivotPos][i];
			while(!pivotVal){
				++colN;
				++i;
				pivotPos = getPivotrow(matrix, topRow, colN, topRow, sizeY);
				pivotVal = matrix[pivotPos][i];
			}
			matrix[pivotPos][i] = 1;
			swapRows(matrix, topRow, pivotPos, sizeX);
			reduceRows(matrix, topRow, sizeX, pivotVal);
			
			// multithreaded stuff here
			if(tcount > 1){
				jobs = sizeY - i;
				Data d;
				d.topRow = topRow;
				d.colN = colN;
				d.sizeX = sizeX;
				for(y = 0; y < tcount; ++y){
					pthread_create( &threads[y], NULL, paraReduceRow, new Data(d));
				}
				for(y = 0; y < tcount; ++y){
					pthread_join(threads[y], NULL);
				}
				
			}else{
				//serial version here
				for(y = topRow+1; y < sizeY; ++y){
					pivotVal = matrix[y][colN];
					matrix[y][0] = 0;
						for( j = 1; j < sizeX; ++j){
							matrix[y][j] -= pivotVal * matrix[topRow][j];
						}
				}
			}
			++colN;
		}
}


void *paraReduceRow( void *ptr ){
	Data d;
	d = *(Data*)ptr;
	int i, j;
	int end = d.sizeX;
	int change;
	while(true){
		pthread_mutex_lock(&lock);
		i = counter;
		counter++;
		pthread_mutex_unlock(&lock);
		if(i >= jobs){
			return 0;
		}
		change = i + d.topRow;
		if(d.topRow != (change)){
			double pivotVal = matrix[change][d.colN];
			matrix[change][0] = 0;
			for(j = 1; j < end; ++j){
				matrix[change][j] -= pivotVal * matrix[d.topRow][j];
			}
		}
	}
	
	delete (Data*) &d;
	return 0;
}

void reduceOtherRows(double** matrix, int baseRow, int changeRow, int xend, int colN){

}

void reduceRows(double** matrix, int row, int end, double pivotVal){
	for(int j = row + 1; j < end; ++j){
		matrix[row][j] /= pivotVal;
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

