package HomeWork3;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Random;

import HomeWork3.Knn.DistanceCheck;
import weka.core.Instances;

public class MainHW3 {
	private final static double[] pArr = { 1, 2, 3, Double.POSITIVE_INFINITY };
	private final static int[] foldsArr = { -1, 50, 10, 5, 3 };

	private final static DistanceCheck[] distanceCheckArr = { DistanceCheck.Regular, DistanceCheck.Efficient };

	public static BufferedReader readDataFile(String filename) {
		BufferedReader inputReader = null;

		try {
			inputReader = new BufferedReader(new FileReader(filename));
		} catch (FileNotFoundException ex) {
			System.err.println("File not found: " + filename);
		}

		return inputReader;
	}

	public static Instances loadData(String fileName) throws IOException {
		BufferedReader datafile = readDataFile(fileName);
		Instances data = new Instances(datafile);
		data.setClassIndex(data.numAttributes() - 1);
		return data;
	}

	public static void main(String[] args) throws Exception {
		Instances autoData = loadData("auto_price.txt");
		foldsArr[0] = autoData.numInstances();
		autoData.randomize(new Random());
		double[] bestValues = findBestValues(autoData);
		System.out.println("----------------------------");
		System.out.println("Results for original dataset:");
		System.out.println("----------------------------");
		System.out.println("Cross validation error with K = " + (int) bestValues[1] + ", lp = " +  bestValues[2]
				+ ", majority function = " + majorityFunc(bestValues[3]) + " for auto_price data is: " + bestValues[0]
				+ "\n");

		FeatureScaler instancesScaler = new FeatureScaler();
		Instances autoScaled = instancesScaler.scaleData(autoData);
		bestValues = findBestValues(autoScaled);

		System.out.println("----------------------------");
		System.out.println("Results for scaled dataset:");
		System.out.println("----------------------------");
		System.out.println("Cross validation error with K = " + (int) bestValues[1] + ", lp = " + bestValues[2]
				+ ", majority function = " + majorityFunc(bestValues[3]) + " for auto_price data is: " + bestValues[0]
				+ "\n");

		Knn knn = new Knn();
		knn.buildClassifier(autoScaled);
		knn.setK((int) bestValues[2]);
		knn.setP(bestValues[2]);
		knn.setWeighted((int) bestValues[3]);
		double error;
		long time;
		for (int i = 0; i < foldsArr.length; i++) {
			System.out.println("----------------------------");
			System.out.println("Results for " + foldsArr[i] + " folds:");
			System.out.println("----------------------------");
			for (int j = 0; j < distanceCheckArr.length; j++) {
				knn.setDistanceCheck(distanceCheckArr[j]);
				error = knn.crossValidationError(autoScaled, foldsArr[i]);
				if (distanceCheckArr[j] == DistanceCheck.Regular) {
					 System.out.println("Cross validation error of regular knn on auto_price dataset is: " + error + " and the average elapsed time is: " + (knn.getTime()/ foldsArr[i])+ " The total elapsed time is: " +
							 knn.getTime()+"\n");
				} else {
					 System.out.println("Cross validation error of efficient knn on auto_price dataset is: " + error + " and the average elapsed time is: " + (knn.getTime()/ foldsArr[i]) + " The total elapsed time is: " +
							 knn.getTime()+"\n");
				}
			}

		}

	}

	private static String majorityFunc(double majNum) {
		if (majNum == 0) {
			return "uniform";
		} else if (majNum == 1) {
			return "weighted";
		} else {
			return "error";
		}
	}

	private static double[] findBestValues(Instances data) throws Exception {
		double[] bestValues = new double[4];
		double err;
		double minErr = Double.POSITIVE_INFINITY;
		int bestK = -1;
		double bestP = -1;
		int bestWeighted = -1;
		Knn knn = new Knn();
		knn.buildClassifier(data);
		for (int k = 1; k < 21; k++) {
			for (int p = 0; p < pArr.length; p++) {
				for (int weighted = 0; weighted < 2; weighted++) {
					knn.setDistanceCheck(DistanceCheck.Regular);
					knn.setK(k);
					knn.setP(pArr[p]);
					knn.setWeighted(weighted);
					err = knn.crossValidationError(data, 10);
					if (err < minErr) {
						minErr = err;
						bestK = k;
						bestP = pArr[p];
						bestWeighted = weighted;
					}
				}
			}

		}
		bestValues[0] = minErr;
		bestValues[1] = bestK;
		bestValues[2] = bestP;
		bestValues[3] = bestWeighted;
		return bestValues;
	}

}
