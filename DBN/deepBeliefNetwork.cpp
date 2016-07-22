//Implementation of Deep Belief Network in C++
//Author: Pritish Yuvraj
//Designation: Summer Research Fellow
//University: Indian Statistical Institute, Kolkata

#include<iostream>
#include<cmath>
#include<cstdlib>
#include<cstring>
#include<cstdio>
#define MAX 10
using namespace std;
int count = 0;
int check = 0;
int lock = 0;

class Mathematics{
public:

	template <typename C, typename D, typename E>
	void MatrixMultiplication(C **Matrix1, int X1, int Y1, D **Matrix2,
		int X2, int Y2, E **MatrixOuput){
		register int i,j,k;
		if(Y1 != X2){
			cout<<"\nWrong indices for matrix multiplication\n";
			cout<<"X1 Y1 "<<X1<<" "<<Y1<<endl;
			cout<<"X2 Y2 "<<X2<<" "<<Y2<<endl;
			exit(0);
		}
		for(i = 0; i<X1; i++){
			for(j = 0; j<Y2; j++){
				MatrixOuput[i][j] = 0;
				for(k = 0; k<Y1; k++){
					MatrixOuput[i][j] += Matrix1[i][k] * Matrix2[k][j];
				}
			}
		}
	}

	template<typename B, typename C, typename D>
	B **binomial(B **sample,C **matrix, int X, int Y, D No){
		register int i, j, k;
		int c;
		double r;
		for(i = 0; i<X; i++){
			for(j = 0; j<Y; j++){
				if(matrix[i][j] < 0 || matrix[i][j] > 1)
					sample[i][j] = int(0);
				c = 0;
				for(k = 0; k<No; k++){
					r = rand() / (RAND_MAX + 1.0);
					if(r < matrix[i][j])
						c++;
				}
				sample[i][j] = int(c);
			}
		}
		return sample;
	}

	template <typename C, typename D, typename E>
	void MatrixSubtraction(C **matrix1, int X1, int Y1, D **matrix2, int X2, int Y2, E **matrix3){
		register int i, j;
		if((X1 != X2) || (Y1 != Y2)){
			cout<<"\nWrong Indices for subtraction";
			exit(0);
		}
		else{
			for(i = 0; i<X1; i++){
				for(j = 0; j<Y1; j++){
					matrix3[i][j] = matrix1[i][j] - matrix2[i][j];
				}	
			}
		}
	}

	template <typename C, typename D>
	void MatrixAddtion(C **matrix1, int X1, int Y1, D **matrix2, int X2, int Y2){
		register int i, j;
		if((X1 != X2) || (Y1 != Y2)){
			cout<<"\nWrong indices for matrix addition";
			exit(0);
		}
		for(i = 0; i<X1; i++){
			for(j = 0; j<X2; j++){
				matrix1[i][j] += matrix2[i][j];
			}
		}
	}

	template <typename D, typename E>
	void Uniform(D **matrix, E X, E Y,D min, D max){
		register int i, j;
		for(i = 0; i<X; i++){
			for(j = 0; j<Y; j++){
				matrix[i][j] = rand() / (RAND_MAX + 1.0)*(max - min) + min;
			}
		}
	}

	template <typename C, typename D>
	D **Sigmoid(C **matrix1, int X, int Y, D **matrix2){
		register int i, j;
		for(i = 0; i<X; i++){
			for(j = 0; j<Y; j++){
				matrix2[i][j] = (double) 1.0 / (double)(1.0 + exp(-matrix1[i][j]));
			}
		}
		return matrix2;
	}

	template<typename C, typename D>
	void scalarMatrixMultiplication(C **matrix1, int X, int Y, D value){
		register int i, j;
		for(i = 0; i<X; i++){
			for(j = 0; j<Y; j++){
				matrix1[i][j] = (matrix1[i][j] * value);
			}
		}
	}

	template <typename C, typename D, typename E>
	void MatrixVectorAddition(C **Matrix1, int X1, int Y1, D *Vector,
		int X2, E **Matrix3){
		register int i, j;
		for(i = 0; i<X1; i++){
			for(j = 0; j<Y1; j++){
				Matrix3[i][j] = Matrix1[i][j] + Vector[i];
			}
		}
	}

	template <typename C>
	void DisplayMatrix(C **MatrixName, int X, int Y){
		register int i, j;
		cout<<endl;
		for(i =0; i<X; i++){
			for(j = 0; j<Y; j++){
				cout<<MatrixName[i][j]<<"\t";
			}
			cout<<endl;
		}
	cout<<endl<<endl;
	}

	template <typename D>
	D **Create2DMatrix(D **MatrixName, int X, int Y){
		register int k;
		MatrixName = new D*[X];
		for(k = 0; k<X; k++){
			MatrixName[k] = new D[Y];
		}
		return MatrixName;
	}

	template <typename C>
	C **TransposeOfMatrix(C **matrix, int X, int Y, C **TransMatrix){
		register int i, j;
		for(i = 0; i<X; i++){
			for(j = 0; j<Y; j++){
				TransMatrix[j][i] = matrix[i][j];
			}
		}
		return TransMatrix;
	}

	template <typename C>
	void Delete2DMatrix(C **MatrixName, int X, int Y){
		register int i, j;
		for(i = 0; i<X; i++){
			delete[] MatrixName[i];
		}
		delete[] MatrixName;
	}

	template <typename C, typename D>
	void CopyMatrix(C **Matrix1, int X, int Y, D **Matrix2){
		register int i, j;
		for(i = 0; i<X; i++){
			for(j = 0; j<Y; j++){
				Matrix2[i][j] = Matrix1[i][j];
			}
		}
	}

	template <typename C, typename D>
	void initializeWith(C **matrix, int X, int Y, D No){
		register int i, j;
		for(i = 0; i<X; i++){
			for(j = 0; j<Y; j++){
				matrix[i][j] = 0;
			}
		}
	}

	~Mathematics(){
	}
};

Mathematics ob1;

class HiddenLayer{
public:
	int N;
	int Ninput;
	int Noutputs;
	double **weights;
	double *bias;

	HiddenLayer(int size, int in, int out, double **w, 
		double *bp){
		register int i, j;
		N = size;
		Ninput = in;
		Noutputs = out;
		
		if(w == NULL){
			double a;
			weights = new double*[Noutputs];
			for(i = 0; i<Noutputs; i++){
				weights[i] = new double[Ninput];
				a = (float)1 / (float)Ninput;
			}				
			ob1.Uniform(weights, Noutputs, Ninput, -a, a);
		}
		else{
			weights = w;
		}
		if(bp == NULL){
			bias = new double[Noutputs];
		}
		else{
			bias = bp;
		}
	}

	void SampleHiddenGivenVisible(int **input, int X, int Y,int **sample){
		register int i, j;
		double **TransWeight;
		double **linearOutput;
		double **NonLinearOutput;
		if(lock!=1){
		TransWeight = ob1.Create2DMatrix(TransWeight, Ninput, Noutputs);
		TransWeight = ob1.TransposeOfMatrix(weights, Noutputs, Ninput, TransWeight);
		ob1.MatrixMultiplication(input, X, Y, TransWeight, Ninput, Noutputs, sample);
		linearOutput = ob1.Create2DMatrix(linearOutput, X, Noutputs);
		ob1.MatrixVectorAddition(sample, X, Noutputs, bias, Noutputs, linearOutput);
		NonLinearOutput = ob1.Create2DMatrix(NonLinearOutput, X, Noutputs);
		NonLinearOutput = ob1.Sigmoid(NonLinearOutput, X, Noutputs, NonLinearOutput);
		sample = ob1.binomial(sample, NonLinearOutput, X, Noutputs, 1);	
		ob1.Delete2DMatrix(TransWeight, Ninput, Noutputs);
		}
		
	}
	~HiddenLayer(){
		delete[] weights;
		delete[] bias;
	}
};

class RBM{
public:
	int N;
	int Nvisible;
	int Nhidden;
	double **weights;
	double *hiddenBias;
	double *visibleBias;
	int Xsample;
	int Ysample;

	RBM(int size, int nV, int nH, double **w, double *hb, 
		double *vb){
		register int i, j;
		N = size;
		Nvisible = nV;
		Nhidden = nH;

		if(w == NULL){
			double a;
			w = new double*[Nhidden];
			for(i = 0; i<Nhidden; i++){
				weights[i] = new double[Nvisible];
				a = (double)1 / (double)Nvisible;
			}
			ob1.Uniform(weights, Nhidden, Nvisible, -a, a);
			ob1.DisplayMatrix(weights, Nhidden, Nvisible);
		}
		else{
			weights = w;
		}
		
		if(hb == NULL){
			hiddenBias = new double[Nhidden];
			for(i = 0; i<Nhidden; i++){
				hiddenBias[i] = 0;
			}
		}
		else{
			hiddenBias = hb;
		}

		if(vb == NULL){
			visibleBias = new double[Nvisible];
			for(i = 0; i<Nvisible; i++){
				visibleBias[i] = 0;
			}
		}
		else{
			visibleBias = vb;
		}
	}

	void SampleHiddenGivenVisible(int **input, int X, int Y, double **PositiveHiddenMean, int **PositiveHiddenSample){

		int **TransInput;
		double **LinearOutput;
		TransInput = ob1.Create2DMatrix(TransInput,Y , X);
		TransInput = ob1.TransposeOfMatrix(input, X, Y, TransInput);
		LinearOutput = ob1.Create2DMatrix(LinearOutput, Nhidden, X);
		ob1.MatrixMultiplication(weights, Nhidden, Nvisible, TransInput, 
			Y, X, LinearOutput);
		ob1.MatrixVectorAddition(LinearOutput, Nhidden, X, 
			hiddenBias, Nhidden, LinearOutput);
		LinearOutput = ob1.Sigmoid(LinearOutput, Nhidden, X, LinearOutput);
		PositiveHiddenMean = LinearOutput;
		PositiveHiddenSample = ob1.binomial(PositiveHiddenSample,
			PositiveHiddenMean, Nhidden, X, 1);		
		ob1.Delete2DMatrix(TransInput, Y, X);
	}

	void SampleVisibleGivenHidden(int **input, int X, int Y, double **Mean, int **sample){
		int **TransInput;
		TransInput = ob1.Create2DMatrix(TransInput, Y, X);
		TransInput = ob1.TransposeOfMatrix(input, X, Y, TransInput);
		ob1.MatrixMultiplication(TransInput, Y, X, weights, Nhidden, Nvisible, Mean);
		ob1.MatrixVectorAddition(Mean, Y, Nvisible, visibleBias, Nvisible, Mean);
		Mean = ob1.Sigmoid(Mean, Y, Nvisible, Mean);
		sample = ob1.binomial(sample, Mean, Y, Nvisible, 1);
		ob1.Delete2DMatrix(TransInput, Y, X);
	}

	void gibbsHidVisHid(int **Hsample, int X, int Y,double **NegativeVisibleMean,
			int **NegativeVisibleSample, double **NegativeHiddenMean,
			int **NegativeHiddenSample){
		SampleVisibleGivenHidden(Hsample, X, Y, NegativeVisibleMean, NegativeVisibleSample);
		SampleHiddenGivenVisible(NegativeVisibleSample, Y, Nvisible, NegativeHiddenMean, NegativeHiddenSample);
	}

	void constrativeDivergence(int **input, int X, int Y,double LearningRate, int k){
		register int i, j;
		double **PositiveHiddenMean; 
		int **PositiveHiddenSample;
		int **NegativeHiddenSample;
		int **NegativeVisibleSample;
		double **NegativeHiddenMean;
		double **NegativeVisibleMean;
		
		double **PosHidMeanWithIn, **NegBackPropagation;
		PosHidMeanWithIn = ob1.Create2DMatrix(PosHidMeanWithIn, N, N);
		NegBackPropagation = ob1.Create2DMatrix(NegBackPropagation, N, N);

		Xsample = Nhidden;
		Ysample = N;
		int step;

		PositiveHiddenMean = ob1.Create2DMatrix(PositiveHiddenMean, N, N);
		PositiveHiddenSample = ob1.Create2DMatrix(PositiveHiddenSample, N, N);
		NegativeHiddenSample = ob1.Create2DMatrix(NegativeHiddenSample, N, N);
		NegativeHiddenMean = ob1.Create2DMatrix(NegativeHiddenMean, N, N);
		NegativeVisibleSample = ob1.Create2DMatrix(NegativeVisibleSample, N, N);
		NegativeVisibleMean = ob1.Create2DMatrix(NegativeVisibleMean, N, N);
		SampleHiddenGivenVisible(input, X, Y, PositiveHiddenMean, PositiveHiddenSample);
		//Gibbs Sampling
		for(step = 0; step<0; step++){
			if(step == 0){
				gibbsHidVisHid(PositiveHiddenSample, Nhidden, X, NegativeVisibleMean,
					NegativeVisibleSample, NegativeHiddenMean, 
					NegativeHiddenSample);
			}
			else{
				gibbsHidVisHid(NegativeHiddenSample, Nhidden, X,NegativeVisibleMean,
					NegativeVisibleSample, NegativeHiddenMean,
					NegativeHiddenSample);
			}
		}

		PosHidMeanWithIn = ob1.Create2DMatrix(PosHidMeanWithIn, N, N);
		NegBackPropagation = ob1.Create2DMatrix(NegBackPropagation, N, N);
		//Changing weights
		ob1.MatrixMultiplication(PositiveHiddenMean, Nvisible, X, input, X, Y, PosHidMeanWithIn);
		ob1.MatrixMultiplication(NegativeHiddenMean, Nvisible, Nhidden, NegativeVisibleSample, Nhidden, Nvisible, NegBackPropagation);
		ob1.MatrixSubtraction(PosHidMeanWithIn, Nvisible, Y, NegBackPropagation, Nvisible, Nvisible, PosHidMeanWithIn);
		ob1.scalarMatrixMultiplication(PosHidMeanWithIn, Nhidden, Y, (double)N);
		ob1.scalarMatrixMultiplication(PosHidMeanWithIn, Nhidden, Y, LearningRate);
		ob1.MatrixAddtion(weights, Nhidden, Nvisible, PosHidMeanWithIn, Nhidden, Y);

		//Changing Hidden Bias
		for(i = 0; i<N; i++){
			for(j = 0; j<Nhidden; j++){
				hiddenBias[j] += (double)(LearningRate/N)*(PositiveHiddenSample[i][j] - NegativeHiddenSample[i][j]);
			}
		}

		//Changing Visible Bias
		for(i = 0; i<N; i++){
			for(j = 0; j<Nvisible; j++){
				visibleBias[j] += (double)(LearningRate/N)*(input[i][j] - NegativeVisibleSample[i][j]);
			}
		}

		ob1.Delete2DMatrix(PositiveHiddenSample, N, N);
		ob1.Delete2DMatrix(PositiveHiddenMean, N, N);
		ob1.Delete2DMatrix(NegativeHiddenSample, N, N);
		ob1.Delete2DMatrix(NegativeHiddenMean, N, N);
		ob1.Delete2DMatrix(NegativeVisibleSample, N, N);
		ob1.Delete2DMatrix(NegativeVisibleMean, N, N);
		ob1.Delete2DMatrix(PosHidMeanWithIn, N, N);
		ob1.Delete2DMatrix(NegBackPropagation, N, N);
	}

	~RBM(){
		delete[] visibleBias;
	}
};

class LogisticRegression{
public:
	int N;
	int Ninput;
	int Noutputs;
	double **weights;
	double *bias;

	LogisticRegression(int size, int in, int out){
		register int i, j;
		N = size;
		Ninput = in;
		Noutputs = out;
		weights = new double*[Noutputs];
		for(i = 0; i<Noutputs; i++){
			weights[i] = new double[Ninput];
		}
		bias = new double[Noutputs];

		for(i = 0; i<Noutputs; i++){
			for(j = 0; j<Ninput; j++){
				weights[i][j] = 0;
			}
			bias[i] = 0;
		}
	}

	void SoftMax(double **x, int X, int Y){
		register int i, j;
		double *maxi = new double[X];
		double *sum = new double[X];
		//Initializing max and sum matrices with 0
		for(i = 0; i<X; i++){
			maxi[i] = 0;
			sum[i] = 0;
		}
		for(i = 0; i<X; i++){
			for(j = 0; j<Y; j++){
				if(maxi[i] < x[i][j])
					maxi[i] = x[i][j];
			}
		}
		for(i = 0; i<X; i++){
			for(j = 0; j<Y; j++){
				x[i][j] = exp(x[i][j] - maxi[i]);
				sum[i] += x[i][j];
			}
		}
		for(i = 0; i<X; i++){
			for(j = 0; j<Y; j++){
				x[i][j] /= sum[i];
			}
		}
		delete[] maxi;
		delete[] sum;
	}

	void train(int **input, int X, int Y, int **output, int lr){
		double **PosYgivenX;
		double **theta;
		double **temp, **temp2;
		double **Transtemp;
		int **TransInput;
		register int i, j;
		PosYgivenX = ob1.Create2DMatrix(PosYgivenX, N, Noutputs);
		theta = ob1.Create2DMatrix(theta, N, Noutputs);
		//Initialize with 0
		ob1.initializeWith(PosYgivenX, N, Noutputs, 0);
		temp = ob1.Create2DMatrix(temp, Ninput, Noutputs);
		temp2 = ob1.Create2DMatrix(temp2, N, Noutputs);
		TransInput = ob1.Create2DMatrix(TransInput, Y, X);
		TransInput = ob1.TransposeOfMatrix(input, X, Y, TransInput);
		ob1.MatrixMultiplication(weights, Noutputs, Ninput, TransInput, Y, X, temp);
		Transtemp = ob1.Create2DMatrix(Transtemp, N, Noutputs);
		Transtemp = ob1.TransposeOfMatrix(temp, Noutputs, X, Transtemp);
		ob1.MatrixAddtion(PosYgivenX, N, Noutputs, Transtemp, X, Noutputs);
		ob1.MatrixVectorAddition(PosYgivenX, N, Noutputs, bias, Noutputs, PosYgivenX);
		//Turning Outputs to Probability with Softmax Activation
		SoftMax(PosYgivenX, N, Noutputs);
		ob1.MatrixSubtraction(output, N, Noutputs, PosYgivenX, N, Noutputs, theta);
		ob1.MatrixMultiplication(theta, N, Noutputs, TransInput, Noutputs, X, temp);
		ob1.scalarMatrixMultiplication(temp, N, X, (double)(lr / N));
		for(i = 0; i<X; i++){
			ob1.MatrixVectorAddition(weights, Ninput, Noutputs, temp[i], X, weights);
			for(j = 0; j<Y; j++){
				bias[i] += (((double)(lr/N))/theta[i][j]);
			}

		}

	}

	~LogisticRegression(){
		register int i;
		for(i = 0; i<Noutputs; i++){
			delete[] weights[i];
		}
		delete[] bias;
	}
};

class DBN{
public:
	int N;
	int Ninput;
	int *HiddenLayerSize;
	int Noutputs;
	int Nlayers;
	int inputSizes;

	HiddenLayer **sigmoidLayers;
	RBM **rbmLayers;
	LogisticRegression *LogisRegressionLayer;

	DBN(int size, int Nin, int *hls, int Nout, int Nlay){
		register int i,j;
		N = size;
		Ninput = Nin;
		HiddenLayerSize = hls;
		Noutputs = Nout;
		Nlayers = Nlay;

		sigmoidLayers = new HiddenLayer*[Nlayers];
		rbmLayers = new RBM*[Nlayers];

		for(i = 0; i<Nlayers; i++){
			if(i==0){
				inputSizes = Ninput;
			}
			else{
				inputSizes = HiddenLayerSize[i-1];
			}
			//Constructing Sigmoid Layer
			sigmoidLayers[i] = new HiddenLayer(N, inputSizes, 
				HiddenLayerSize[i], NULL, NULL);

			rbmLayers[i] = new RBM(N, inputSizes, HiddenLayerSize[i],
				sigmoidLayers[i]->weights, sigmoidLayers[i]->bias,
				NULL);
		}
		LogisRegressionLayer = new LogisticRegression(N, 
			HiddenLayerSize[Nlayers - 1], Noutputs);
	}

	void preTraining(int input1[MAX][MAX], double learningRate, 
		int k1, int preTrainingEpoch){
		register int i, j, layer, epoch, k;
		int **PresentInput;
		int **prevLayerInput;
		int prevInputSize;
		//Just to convert int input into int **input1 for further 
		//use of Mathematics class
		int **input;

		input = ob1.Create2DMatrix(input, N, Ninput);
		//input[N-1][Ninput-1] = 10;
		
		for(i = 0; i<N; i++){
			for(j = 0; j<Ninput; j++){
				input[i][j] = input1[i][j];
			}
		}
		
		for(layer = 0; layer<Nlayers; layer++){
			for(epoch = 0; epoch<preTrainingEpoch; epoch++){
				for(i = 0; i<=layer; i++){
					if(i == 0){
						PresentInput = ob1.Create2DMatrix(PresentInput, N, Ninput);
						ob1.CopyMatrix(input, N, Ninput, PresentInput);
						prevInputSize = Ninput;
					}
					else{
						if(i == 1){
							prevInputSize = Ninput;
							}
						else{
							prevInputSize = HiddenLayerSize[i - 2];
						}			
						prevLayerInput = ob1.Create2DMatrix(prevLayerInput, N, prevInputSize);
						ob1.CopyMatrix(PresentInput, N, prevInputSize, prevLayerInput);
						ob1.Delete2DMatrix(PresentInput, N, prevInputSize);
						PresentInput = ob1.Create2DMatrix(PresentInput, N, HiddenLayerSize[i - 1]);
						sigmoidLayers[i - 1]->SampleHiddenGivenVisible(prevLayerInput, N, 
							prevInputSize, PresentInput);
						prevInputSize = HiddenLayerSize[i-1];
						ob1.Delete2DMatrix(prevLayerInput, N, prevInputSize);

					}
				}
				rbmLayers[layer]->constrativeDivergence(PresentInput, N, prevInputSize, learningRate, k1);
			}
		}
		ob1.Delete2DMatrix(PresentInput, N, prevInputSize);
		ob1.Delete2DMatrix(input, N, Ninput);
	}

	void finetune(int input[MAX][MAX], int Ntraining, int Ninput, 
		int label[MAX][MAX], int outputs,double lr, double epochs){
		register int i, j, k, epoch, layer;
		int prevLayerSize;
		int **layerInput;
		int **prevLayerInput;
		int **trainX;
		int **trainY;
		trainX = ob1.Create2DMatrix(trainX, Ntraining, Ninput);
		trainY = ob1.Create2DMatrix(trainY, Ntraining, Noutputs);
		//Copying the matrix from int[][] to int**
		for(i = 0; i<Ntraining; i++){
			for(j = 0; j<Ninput; j++){
				trainX[i][j] = input[i][j];
			}
			for(k = 0; k<Noutputs; k++){
				trainY[i][k] = label[i][k];
			}
		}
		//ob1.DisplayMatrix(trainX, Ntraining, Ninput);
		for(epoch = 0; epoch<epochs; epoch++){
			for(layer=0; layer<1; layer++){
				if(layer==0){ //Training matrix is same as Input matrix
					prevLayerInput = ob1.Create2DMatrix(prevLayerInput, Ntraining, Ninput);
					ob1.CopyMatrix(trainX, Ntraining, Ninput, prevLayerInput);
				}
				else{
					prevLayerInput = ob1.Create2DMatrix(prevLayerInput, Ntraining, HiddenLayerSize[layer - 1]);
					prevLayerSize = HiddenLayerSize[layer-1];
					ob1.CopyMatrix(layerInput, Ntraining, prevLayerSize, prevLayerInput);							
					ob1.Delete2DMatrix(layerInput, Ntraining, prevLayerSize);
				}
				layerInput = ob1.Create2DMatrix(layerInput, Ntraining, HiddenLayerSize[layer]);
				sigmoidLayers[i]->SampleHiddenGivenVisible(prevLayerInput, 
					Ntraining, prevLayerSize,layerInput);				
				ob1.Delete2DMatrix(prevLayerInput, Ntraining, prevLayerSize);
			}
		}
		LogisRegressionLayer->train(layerInput, Ntraining, prevLayerSize,trainY, lr);
	}
};

void test_dbn();

int main(){
	test_dbn();
}

void test_dbn(){
	int inputX[MAX][MAX] = {	{1, 1, 1, 0, 0, 0},
					{1, 0, 1, 0, 0, 0},
					{1, 1, 1, 0, 0, 0},
					{0, 0, 1, 1, 0, 0},
					{0, 0, 1, 1, 0, 0},
					{0, 0, 1, 1, 1, 0} };
	int outputY[MAX][MAX] = { {1, 0},
					{1, 0},
					{1, 0},
					{0, 1},
					{0, 1},
					{0, 1} };

	int Ntraining = 6;
	int Ninputs = 6;
	int hiddenLayers[] = {3, 3};
	int Noutputs = 2;
	int Nlayers = 2;
	double learningRate = 0.1;
	int k = 1;
	int preTrainingEpoch = 1;
	double FinetuneLearningRate = 0.1;
	double FinetuneEpochs = 1;

	DBN dbn(Ntraining, Ninputs, hiddenLayers, Noutputs, Nlayers);

	dbn.preTraining(inputX, learningRate, k, preTrainingEpoch);

	dbn.finetune(inputX, Ntraining, Ninputs,outputY, Noutputs,FinetuneLearningRate, FinetuneEpochs);
}	
