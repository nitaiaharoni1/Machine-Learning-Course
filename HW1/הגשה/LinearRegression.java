//Nitai Aharoni 203626742

package HomeWork1;

import java.util.Arrays;

import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

public class LinearRegression implements Classifier {

	private int m_ClassIndex;
	private int m_truNumAttributes;
	private double[] m_coefficients;
	private double m_alpha;


	public int getM_ClassIndex() {
		return m_ClassIndex;
	}

	public int getM_truNumAttributes() {
		return m_truNumAttributes;
	}

	public double[] getM_coefficients() {
		return m_coefficients;
	}

	public double getM_alpha() {
		return m_alpha;
	}

	// the method which runs to train the linear regression predictor, i.e.
	// finds its weights.
	@Override
	public void buildClassifier(Instances trainingData) throws Exception {
		this.m_ClassIndex = trainingData.classIndex();
		this.m_truNumAttributes = trainingData.numAttributes() - 1;
		rndCoeff();
		if(this.m_alpha == 0) {
			findAlpha(trainingData);
		} else {
			m_coefficients = gradientDescent(trainingData);			
		}
	}
	
	/**
	 * @param data
	 * @throws Exception
	 */
	private void findAlpha(Instances data) throws Exception {
		double targetAlpha = 0;
		double curErr;
		double prvErr;
		double minErr = Integer.MAX_VALUE;
		double[] alpha_coefficients = new double[m_coefficients.length];
		for (int i = -17; i <= 0; i++) {
			m_alpha = Math.pow(3, i);
			curErr = Integer.MAX_VALUE;
			prvErr = 0;
			rndCoeff();
			for (int j = 0; j < 20000; j++) {
				m_coefficients = stepGrad(data);
				if(j%100==0) {
					prvErr = curErr;
					curErr = calculateMSE(data);
					if(curErr > prvErr) {
						break;
					}
				}
				if (curErr < minErr) {
					minErr = curErr;
					targetAlpha = m_alpha;
					alpha_coefficients = m_coefficients;
				}
			}
		}
		m_alpha = targetAlpha;
		m_coefficients = alpha_coefficients;
	}

	/**
	 * An implementation of the gradient descent algorithm which should return the
	 * weights of a linear regression predictor which minimizes the average squared
	 * error.
	 * 
	 * @param trainingData
	 * @throws Exception
	 */
	private double[] gradientDescent(Instances trainingData) throws Exception {
		double prvErr = 0;
		double curErr = calculateMSE(trainingData);
		while (Math.abs(prvErr - curErr) >= 0.003) {
			for (int i = 0; i < 100; i++) {
				m_coefficients = stepGrad(trainingData);
			}
			prvErr = curErr;
			curErr = calculateMSE(trainingData);
		}
		return m_coefficients;
	}
	
	/**
	 * @param data
	 * @return
	 * @throws Exception
	 */
	public double[] stepGrad(Instances data) throws Exception {
		double[] temp_coefficients = new double[m_truNumAttributes+1];
		for (int thetaIndex = 0; thetaIndex < m_truNumAttributes + 1; thetaIndex++) {
			temp_coefficients[thetaIndex] = m_coefficients[thetaIndex] - m_alpha * calcDevSum(data, thetaIndex);
		}
		return temp_coefficients;
	}
	
	/**
	 * @param data
	 * @param thetaIndex
	 * @return
	 * @throws Exception
	 */
	public double calcDevSum(Instances data, int thetaIndex) throws Exception {
		double attVal = 0;
		double devSum = 0;
		for (int instIndex = 0; instIndex < data.numInstances(); instIndex++) {
			if (thetaIndex == 0) {
				attVal = 1;
			} else {
				attVal = data.instance(instIndex).value(thetaIndex - 1);
			}
			devSum += (regressionPrediction(data.instance(instIndex))
					- data.instance(instIndex).classValue()) * attVal;
		}
		devSum = devSum / data.numInstances();
		return devSum;
	}

	/**
	 * Returns the prediction of a linear regression predictor with weights given by
	 * m_coefficients on a single instance.
	 *
	 * @param instance
	 * @return
	 * @throws Exception
	 */
	public double regressionPrediction(Instance instance) throws Exception {
		double pred = m_coefficients[0];
		for (int i = 1; i < m_truNumAttributes + 1; i++) {
			pred += m_coefficients[i] * instance.value(i-1);
		}
		return pred;
	}

	/**
	 * Calculates the total squared error over the data on a linear regression
	 * predictor with weights given by m_coefficients.
	 *
	 * @param testData
	 * @return
	 * @throws Exception
	 */
	public double calculateMSE(Instances data) throws Exception {
		double Err = 0;
		for (int instIndex = 0; instIndex < data.numInstances(); instIndex++) {
			Err += Math.pow(regressionPrediction(data.instance(instIndex)) - data.instance(instIndex).classValue(),
					2);
		}
		return (Err / (2.0 * data.numInstances()));
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

	// initial coefficients with random values
	private void rndCoeff() {
		m_coefficients = new double[(m_truNumAttributes + 1)];
		for (int i = 0; i < m_truNumAttributes + 1; i++) {
			m_coefficients[i] = Math.random();
		}
	}
	


}
