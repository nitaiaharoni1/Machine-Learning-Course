package HomeWork1;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;
import java.util.concurrent.ThreadLocalRandom;

import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

public class MainHW1 {

	public static BufferedReader readDataFile(String filename) {
		BufferedReader inputReader = null;
		try {
			inputReader = new BufferedReader(new FileReader(filename));
		} catch (FileNotFoundException ex) {
			System.err.println("File not found: " + filename);
		}
		return inputReader;
	}

	/**
	 * Sets the class index as the last attribute.
	 * 
	 * @param fileName
	 * @return Instances data
	 * @throws IOException
	 */
	public static Instances loadData(String fileName) throws IOException {
		BufferedReader datafile = readDataFile(fileName);
		Instances data = new Instances(datafile);
		data.setClassIndex(data.numAttributes() - 1);
		return data;
	}

	public static void main(String[] args) throws Exception {
		// load data
		Instances trainingData = loadData("wind_training.txt");
		Instances testingData = loadData("wind_testing.txt");
		LinearRegression linearRegression = new LinearRegression();
		linearRegression.buildClassifier(trainingData);
		double alpha = linearRegression.getM_alpha();
		System.out.println("The chosen alpha is: " + alpha);
		double trainingErr = linearRegression.calculateMSE(trainingData);
		double testingErr = linearRegression.calculateMSE(testingData);
		System.out.println("Training error with all features is: " + trainingErr);
		System.out.println("Test error with all features is: " + testingErr);
		Instances filtered = null;
		Remove remove = new Remove();
		remove.setInvertSelection(true);
		int targetIndex = linearRegression.getM_ClassIndex();
		int[] arrayOf3 = new int[4];
		arrayOf3[3] = targetIndex;
		int[] best3 = null;
		double minTrainingErr3 = Integer.MAX_VALUE;
		double trainingErr3 = 0;
		double minTestingErr3 = 0;
		for (int i = 0; i < targetIndex - 2; i++) {
			for (int j = i + 1; j < targetIndex - 1; j++) {
				for (int k = j + 1; k < targetIndex; k++) {
					if (i != j && j != k) {
						arrayOf3[0] = i;
						arrayOf3[1] = j;
						arrayOf3[2] = k;
						remove.setAttributeIndicesArray(arrayOf3);
						remove.setInputFormat(trainingData);
						filtered = Filter.useFilter(trainingData, remove);
						linearRegression.buildClassifier(filtered);
						trainingErr3 = linearRegression.calculateMSE(filtered);
						
						System.out.println(trainingData.attribute(i).name() + " " + i + " - "
								+ trainingData.attribute(j).name() + " " + j + " - "
								+ (trainingData.attribute(k).name()) + " " + k + " - Training Error: " + trainingErr3);
						if (trainingErr3 < minTrainingErr3) {
							minTrainingErr3 = trainingErr3;
							best3 = arrayOf3.clone();
							remove.setInputFormat(testingData);
							filtered = Filter.useFilter(testingData, remove);
							minTestingErr3 = linearRegression.calculateMSE(filtered);
						}
					}
				}
			}
		}
		System.out.println("Training error the features " + best3[0] + "," + best3[1] + "," + best3[2] + ": " + minTrainingErr3);
		System.out.println("Test error the features " + best3[0] + "," + best3[1] + "," + best3[2] + ": " + minTestingErr3);
	}
}
