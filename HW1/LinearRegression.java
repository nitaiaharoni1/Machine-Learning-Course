//Nitai Aharoni 203626742

package HomeWork1;

import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

public class LinearRegression implements Classifier {

	private int m_ClassIndex;
	private int m_truNumAttributes;
	private double[] m_coefficients;
	private double m_alpha;

	// the method which runs to train the linear regression predictor, i.e.
	// finds its weights.
	@Override
	public void buildClassifier(Instances trainingData) throws Exception {
	//
		trainingData = new Instances(trainingData);
		m_ClassIndex = trainingData.classIndex();
		// since class attribute is also an attribute we subtract 1
		m_truNumAttributes = trainingData.numAttributes() - 1;
		findAlpha(trainingData);
		// initialize the coefficient array
		initCoefficients();
		m_coefficients = gradientDescent(trainingData);
		// print results
		System.out.println("The weights are: ");
		for(int i = 0; i < m_coefficients.length; i++){
			System.out.print(m_coefficients[i] + " ");
		}
		System.out.println();
	}
	//

	private void findAlpha(Instances data) throws Exception {
	//
		double minError = Double.MAX_VALUE;
		double currError = Double.MAX_VALUE;
		double bestAlpha = Double.MAX_VALUE;
		for(int i = -17; i <= 0; i++){
			m_alpha = Math.pow(3, i);
			initCoefficients();
			for(int j = 0; j < 20000; j++){
				singleStepGD(data);
			}
			currError = calculateMSE(data);
			if (currError < minError){
				minError = currError;
				bestAlpha = m_alpha;
			}
		}
		m_alpha = bestAlpha;	
	}
	//

	/**
	 * An implementation of the gradient descent algorithm which should return the
	 * weights of a linear regression predictor which minimizes the average squared
	 * error.
	 * 
	 * @param trainingData
	 * @throws Exception
	 */
	private double[] gradientDescent(Instances trainingData) throws Exception {
//
		double prevError = calculateMSE(trainingData);
		double currError = 0;
		// stop condition
		while (Math.abs(prevError - currError) >= 0.003) {
			for (int i = 0; i < 100; i++) {
				singleStepGD(trainingData);
			}
			prevError = currError;
			currError = calculateMSE(trainingData);
			// check for valid error
			if (Double.isNaN(currError)) {
				currError = Double.MAX_VALUE;
			}
		}
		return m_coefficients;
	}
//

	/**
	 * Returns the prediction of a linear regression predictor with weights given by
	 * m_coefficients on a single instance.
	 *
	 * @param instance
	 * @return
	 * @throws Exception
	 */
	public double regressionPrediction(Instance instance) throws Exception {
	//	
		double sum = m_coefficients[0];
		for(int i = 0; i < m_truNumAttributes; i++){
			sum += this.m_coefficients[i + 1] * instance.value(i);
		}
		return sum;
	}
	//

	/**
	 * Calculates the total squared error over the data on a linear regression
	 * predictor with weights given by m_coefficients.
	 *
	 * @param testData
	 * @return
	 * @throws Exception
	 */
	public double calculateMSE(Instances data) throws Exception {
	//
		double squeredError = 0;
		int numInstences = data.numInstances();
		for(int i = 0; i < numInstences; i++){
			squeredError += Math.pow(regressionPrediction(data.instance(i)) - data.instance(i).classValue(), 2);
		}
		return (squeredError / (2.0 * numInstences));
	}
	//
	
	
	
	
	
	
    
	/**
	 * 
	 * guess coefficient values
	 * initialize all coefficients with random values between 0 to 1
	 * @param data
	 */
	private void initCoefficients() {
		m_coefficients = new double[m_truNumAttributes + 1];
		for (int i = 0; i <= m_truNumAttributes; i++){ 
			m_coefficients[i] = Math.random();	
		}
	}
	
	/**
	 * 
	 * calculate the derivative
	 * @param trainingData
	 * @param index
	 * @return
	 * @throws Exception
	 */
	public double calculateDerivative(Instances trainingData, int index) throws Exception{
		double derivative = 0;
		double featureVal = 0;
		int numInstences = trainingData.numInstances();
		for(int i = 0; i < numInstences; i++){
			// check if theta 0
			if(index == 0) {
				featureVal = 1;
			}
			else {
				featureVal = trainingData.instance(i).value(index - 1);
			}
			derivative += (regressionPrediction(trainingData.instance(i)) - trainingData.instance(i).value(m_ClassIndex)) * featureVal; 
		}
		return derivative / numInstences;
	}
	
	/**
	 * One iteration of gradient descent
	 * @param trainingData
	 * @throws Exception
	 */
	public void singleStepGD(Instances trainingData) throws Exception{
		
		// Saving the coefficients values in temp array
		double[] temp_coefficients = new double[m_truNumAttributes + 1];
		for(int i = 0; i < temp_coefficients.length; i++){
			temp_coefficients[i] = m_coefficients[i];
		}
		
		// Update the coefficients
		for(int i = 0; i <= m_truNumAttributes; i++){
			m_coefficients[i] = temp_coefficients[i] - m_alpha * calculateDerivative(trainingData, i);
		}
	}
	
	
	
	
	
	
	
	
	
	
	
	

	@Override
	public double classifyInstance(Instance arg0) throws Exception {
		// Don't change
		return 0;
	}

	@Override
	public double[] distributionForInstance(Instance arg0) throws Exception {
		// Don't change
		return null;
	}

	@Override
	public Capabilities getCapabilities() {
		// Don't change
		return null;
	}
}
