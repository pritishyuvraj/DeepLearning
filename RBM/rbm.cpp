#include<iostream>
#include<cstdlib>
#include<ctime>
#include<cmath>
using namespace std;
#define MAX 100

double rand_normal(double mean, double stddev);

class rbm{
	int num_hidden;
	int num_visible;
	int no_of_exmp;
	float alpha;
	float weights[MAX][MAX];
	float data[MAX][MAX];

public:
	rbm() { }
	rbm(int Nhidden, int NVisible, float learning_rate);
	void train(int epoch);
	//void predict1();
	void show_weights();
	void input_data();
	void dotproduct(float result[MAX][MAX], float matrix1[MAX][MAX], int mat1X, int mat1Y, float matrix2[MAX][MAX], int mat2X, int mat2Y);	
	void sigmoid(float result[MAX][MAX], float input[MAX][MAX], int x, int y);
	void compare(float output[MAX][MAX], float input[MAX][MAX], int x, int y);
	void transpose(float output[MAX][MAX], float input[MAX][MAX], int x, int y);
	void fix_bias_unit_as_one(float input[MAX][MAX], int x, int y);
	void subtract(float output[MAX][MAX], float inputF[MAX][MAX], int x1, int y1, float inputT[MAX][MAX], int x2, int y2);
	void ScalarVectorMultiplication(float matrix[MAX][MAX], int x, int y, float no);
	void MatrixAddition(float matrix1[MAX][MAX], int x1, int y1, float matrix2[MAX][MAX], int x2, int y2);
	float AdditionOfAllTermsOfMatrix(float input[MAX][MAX], int x, int y);
	void squareOfMatrix(float input[MAX][MAX], int x, int y);
	void displayMatrix(float input[MAX][MAX], int x, int y);
};	

void rbm::displayMatrix(float input[MAX][MAX], int x, int y){
	register int i, j;
	//cout<<"\nDisplaying Matrix contents\n";
	for(i = 0; i<x; i++){
		for(j = 0; j<y; j++){
			cout<<input[i][j]<<"\t";
		}
		cout<<endl;
	}
}

rbm::rbm(int Nhidden, int NVisible, float learning_rate){
	register int i, j;
	//Initialization of hidden layers, visible layers and learning rate
	num_hidden = Nhidden;
	num_visible = NVisible;
	alpha = learning_rate;
	//Random Function Generator
	srand(time(NULL));

	for(i = 0; i<= num_visible; i++){
		for(j = 0; j <= num_hidden; j++){
			if( (i==0) || (j==0)){
				weights[i][j] = 0;
			}
			else{
				//weights[i][j] = (float)rand() / (float)RAND_MAX;
				weights[i][j] = rand_normal(0, 0.1);
			}
		}
	}
}

void rbm::show_weights(){ //Show weights
	register int i, j;
	cout<<"\nInitialized Random Weights are:\n";
	for( i=0; i<= num_visible; i++){
		for( j = 0; j<=num_hidden; j++){
			cout<<weights[i][j]<<"\t";
		}
		cout<<endl;
	}

	cout<<"\nRead input data is:\n";
	for(i = 0; i<no_of_exmp; i++){
		for(j = 0; j<=num_visible; j++){
			cout<<data[i][j]<<"\t";
		}
		cout<<endl;
	}
}

void rbm::input_data(){ //Take input from user
	register int i, j;
	cout<<"\nVisible No of units: "<<num_visible;
	cout<<"\nHidden No of units: "<<num_hidden;
	cout<<"\nLearning Rate: "<<alpha;
	cout<<"\nEnter the max no of data:\t";
	cin>>no_of_exmp;
	cout<<"\nEnter the data\n";
	for(i = 0; i<no_of_exmp; i++){
		for(j = 0; j<= num_visible; j++){
			//
			if(j == 0){
				data[i][j] = 1;
				//
				continue;
			}
			else{
				//cout<<"\n i:"<<i<<" j:"<<j;
				cin>>data[i][j];
				//data[i][j] = 0;
			}
		}
	}
}

void rbm::train(int epoch){
	register int i, j;
	float sum, divideby, one_by_no_example;
	
	float result1[MAX][MAX], pos_hidden_probs[MAX][MAX], 
	pos_hidden_states[MAX][MAX], pos_association[MAX][MAX], 
	dataTrans[MAX][MAX], weightTrans[MAX][MAX], neg_visible_activations[MAX][MAX], 
	neg_visible_prob[MAX][MAX], neg_hidden_prob[MAX][MAX], 
	neg_hidden_activations[MAX][MAX], neg_visible_probT[MAX][MAX],
	neg_associations[MAX][MAX], weights_subtraction[MAX][MAX], 
	ThetaWeight[MAX][MAX], error[MAX][MAX];

	cout<<"\nTraining begins.....Epoch value is: "<<epoch<<"\n";
	for(i = 0; i<epoch; i++){
		dotproduct(result1, data, no_of_exmp, num_visible+1, weights, num_visible+1, num_hidden + 1);
		
		sigmoid(pos_hidden_probs, result1, no_of_exmp, num_hidden+1);
		
		compare(pos_hidden_states, pos_hidden_probs, no_of_exmp, num_hidden+1);
		
		transpose(dataTrans, data, no_of_exmp, num_visible+1);
		
		dotproduct(pos_association, dataTrans, num_visible+1, no_of_exmp, pos_hidden_probs, no_of_exmp, num_hidden+1);	
		
		transpose(weightTrans, weights, num_visible+1, num_hidden+1);
		
		dotproduct(neg_visible_activations, pos_hidden_states, no_of_exmp, num_hidden+1, weightTrans, num_hidden+1, num_visible+1);
		
		sigmoid(neg_visible_prob, neg_visible_activations, no_of_exmp, num_visible+1);
		
		fix_bias_unit_as_one(neg_visible_prob, no_of_exmp, num_visible+1);
		
		dotproduct(neg_hidden_activations, neg_visible_prob, no_of_exmp, num_visible+1, weights, num_visible+1, num_hidden+1);
		
		sigmoid(neg_hidden_prob, neg_hidden_activations, no_of_exmp, num_hidden+1);
		
		transpose(neg_visible_probT, neg_visible_prob, no_of_exmp, num_visible+1);
		
		dotproduct(neg_associations, neg_visible_probT, num_visible+1, no_of_exmp, neg_hidden_prob, no_of_exmp, num_hidden+1);
		
		subtract(weights_subtraction, pos_association, num_visible+1, num_hidden+1, neg_associations, num_visible+1, num_hidden+1);
		
		one_by_no_example = ((float)1 / (float)no_of_exmp);
		
		ScalarVectorMultiplication(weights_subtraction, num_visible+1, num_hidden+1, one_by_no_example);
		
		ScalarVectorMultiplication(weights_subtraction, num_visible+1, num_hidden+1, (float)alpha);
		
		MatrixAddition(weights, num_visible+1, num_hidden+1, weights_subtraction, num_visible+1, num_hidden+1);
		
		subtract(error, data, no_of_exmp, num_visible+1, neg_visible_prob, no_of_exmp, num_visible+1);
		
		squareOfMatrix(error, no_of_exmp, num_visible+1);
		
		sum  = AdditionOfAllTermsOfMatrix(error, no_of_exmp, num_visible+1);
		
		cout<<"\n"<<"Epoch\t:"<<i<<"\tError is: "<<sum;
	}	
	
	cout<<"\n\nData and Weights together form\n";
	for(i=0; i<=num_visible; i++){
		for(j = 0; j<= num_hidden; j++){
			cout<<weights[i][j]<<"\t";
		}
		cout<<endl;
	}
	cout<<endl;
}

float rbm::AdditionOfAllTermsOfMatrix(float input[MAX][MAX], int x, int y){
	register int i, j;	
	float sum = 0;
	for(i = 0; i<x; i++){
		for(j = 0; j<y; j++){
			sum += input[i][j];
		}
	}
	return sum;
}

void rbm::squareOfMatrix(float input[MAX][MAX], int x, int y){
	register int i, j;
	for(i = 0; i<x; i++){
		for(j = 0; j<y; j++){
			input[i][j] = pow(input[i][j], 2);
		}
	}
}

void rbm::MatrixAddition(float matrix1[MAX][MAX], int x1, int y1, float matrix2[MAX][MAX], int x2, int y2){
	register int i, j;
	if((x1!=x2) || (y1!=y2)){
		cout<<"Wrong matrix indices";
		exit(2);
	}
	for(i = 0; i<x1; i++){
		for(j = 0; j<y1; j++){
			matrix1[i][j] += matrix2[i][j];
		}
	}
}

void rbm::ScalarVectorMultiplication(float matrix[MAX][MAX], int x, int y, float no){
	register int i, j;
	for(i = 0; i<x; i++){
		for(j = 0; j<y; j++){
			//cout<<"\n"<<"No is "<<no<<" Before Value "<<matrix[i][j];
			matrix[i][j] = (float)matrix[i][j] * (float)no;
			//cout<<"\nAfter Value "<<matrix[i][j];
		}
	}
}

void rbm::subtract(float output[MAX][MAX], float inputF[MAX][MAX], int x1, int y1, float inputT[MAX][MAX], int x2, int y2){
	register int i,j;
	if((x1!=x2) && (y1!=y2)){
		cout<<"Wrong inputs for subtraction";
		exit(1);
	}
	for(i = 0; i<x1; i++){
		for(j = 0; j<y1; j++){
			output[i][j] = inputF[i][j] - inputT[i][j];
		}
	}
}

void rbm::fix_bias_unit_as_one(float input[MAX][MAX], int x, int y){
	register int i;
	for(i = 0; i<x; i++){
		input[i][0] = 1;
	}
}

void rbm::transpose(float output[MAX][MAX], float input[MAX][MAX], int x, int y){
	register int i, j;
	for(i = 0; i<x; i++){
		for(j = 0; j<y; j++){
			output[j][i] = input[i][j];
		}
	}
}

void rbm::compare(float output[MAX][MAX], float input[MAX][MAX], int x, int y){
	register int i,j;
	float temp[MAX][MAX];
	srand(time(NULL));
	for(i=0; i<x; i++){
		for(j = 0; j<y; j++){
			output[i][j] = (float)rand() / (float)RAND_MAX;
		}
	}
	for(i = 0; i<x; i++){
		for(j = 0; j<y; j++){
			if(input[i][j] > output[i][j]){
				output[i][j] = 1;
			}
			else{
				output[i][j] = 0;
			}
		}
	}
}

void rbm::dotproduct(float result[MAX][MAX], float matrix1[MAX][MAX], int mat1X, int mat1Y, float matrix2[MAX][MAX], int mat2X, int mat2Y){
	register int i, j, k;
	if( mat2X != mat1Y){
		cout<<"\nError Wrong inputs for Matrix Multiplication";
		exit(0);
	}
	for(i=0; i<mat1X; i++){
		for(j = 0; j<mat2Y; j++){
			result[i][j] = 0;
			for(k = 0; k<mat2X; k++){
				result[i][j] += (matrix1[i][k] * matrix2[k][j]);
			}
		}
	}
}

void rbm::sigmoid(float result[MAX][MAX], float input[MAX][MAX], int x, int y){
	register int i,j;
	for(i =0; i<x; i++){
		for(j = 0; j<y; j++){
			result[i][j] = (1/(1+(1/exp(input[i][j]))));
		}
	}
}

int main(){
	int visible, hidden, epoch;
	float l;
	cout<<"\nRBM";
	cout<<"\nEnter Visible units\t";
	cin>>visible;
	cout<<"\nEnter Hidden units\t";
	cin>>hidden;
	cout<<"\nEnter the learning rate\t";
	cin>>l;
	cout<<"\nEnter value of Epoch";
	cin>>epoch;
	rbm instance(hidden, visible, l);
	instance.input_data();
	//instance.show_weights();
	instance.train(epoch);
	return 0;
}

double rand_normal(double mean, double stddev){
//Box muller method
    static double n2 = 0.0;
    static int n2_cached = 0;
    if (!n2_cached)
    {
        double x, y, r;
        do
        {
            x = 2.0*rand()/RAND_MAX - 1;
            y = 2.0*rand()/RAND_MAX - 1;

            r = x*x + y*y;
        }
        while (r == 0.0 || r > 1.0);
        {
            double d = sqrt(-2.0*log(r)/r);
            double n1 = x*d;
            n2 = y*d;
            double result = n1*stddev + mean;
            n2_cached = 1;
            return result;
        }
    }
    else
    {
        n2_cached = 0;
        return n2*stddev + mean;
    }
}
