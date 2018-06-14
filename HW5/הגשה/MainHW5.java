package HomeWork5;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Random;

import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.RemovePercentage;
import weka.classifiers.functions.supportVector.Kernel;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.classifiers.functions.supportVector.RBFKernel;

public class MainHW5 {
	private static double[] polyDeg = { 2, 3, 4 };
	private static double[] rbfGamma = { 0.005, 0.05, 0.5 };
	private static double[] Ci = { 1, 0, -1, -2, -3, -4 };
	private static double[] Cj = { 3, 2, 1 };

	private final static double ALPHA = 1.5;

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
		Instances cancerData = loadData("cancer.txt");
		cancerData.randomize(new Random());
		Instances[] training_testing = splitData(cancerData);
		Instances training = training_testing[0];
		Instances testing = training_testing[1];
		double[] tprFpr;
		Kernel bestKernel = null;
		String bestKernelString = null;
		double bestParameter = 0;
		double eq;
		double bestEq = 0;

		SVM svm = new SVM();
		for (int i = 0; i < polyDeg.length; i++) {
			PolyKernel polykernel = new PolyKernel();
			polykernel.setExponent(polyDeg[i]);
			svm.setKernel(polykernel);
			svm.buildClassifier(training);
			tprFpr = svm.calcTprFpr(svm.calcConfusion(testing));
			eq = clac(tprFpr, ALPHA);
			System.out.println("For PolyKernel with degree " + polyDeg[i] + " the rates are:");
			System.out.println("TPR = " + tprFpr[0]);
			System.out.println("FPR = " + tprFpr[1]);
			System.out.println("------------------------------------------------");
			if (eq > bestEq) {
				bestEq = eq;
				bestKernel = polykernel;
				bestParameter = polyDeg[i];
				bestKernelString = "PolyKernel";
			}
		}

		for (int i = 0; i < rbfGamma.length; i++) {
			RBFKernel rbfkernel = new RBFKernel();
			rbfkernel.setGamma(rbfGamma[i]);
			svm.setKernel(rbfkernel);
			svm.buildClassifier(training);
			tprFpr = svm.calcTprFpr(svm.calcConfusion(testing));
			eq = clac(tprFpr, ALPHA);
			System.out.println("For RBFKernel with gamma " + rbfGamma[i] + " the rates are:");
			System.out.println("TPR = " + tprFpr[0]);
			System.out.println("FPR = " + tprFpr[1]);
			System.out.println("------------------------------------------------");
			if (eq > bestEq) {
				bestEq = eq;
				bestKernel = rbfkernel;
				bestParameter = rbfGamma[i];
				bestKernelString = "RBFKernel";
			}
		}

		System.out.println("The best kernel is: " + bestKernelString + " " + bestParameter + " " + bestEq);

		double c;
		svm.setKernel(bestKernel);
		for (int i = 0; i < Ci.length; i++) {
			for (int j = 0; j < Cj.length; j++) {
				c = calcC(Ci[i], Cj[j]);
				svm.setC(c);
				svm.buildClassifier(training);
				tprFpr = svm.calcTprFpr(svm.calcConfusion(testing));
				System.out.println("------------------------------------------------");
				System.out.println("For C " + c + " the rates are:");
				System.out.println("TPR: " + tprFpr[0]);
				System.out.println("FPR: " + tprFpr[1]);
			}
		}
	}

	public static double clac(double[] arr, double alpha) {
		double eq = (alpha * arr[0]) - arr[1];
		return eq;
	}

	public static double calcC(double Ci, double Cj) {
		double c = (Math.pow(10, Ci)) * (Cj / 3);
		return c;
	}

	public static Instances[] splitData(Instances instances) throws Exception {
		Instances[] training_testing = new Instances[2];
		Instances training = new Instances(instances, 0);
		Instances testing = new Instances(instances, 0);
		for (int i = 0; i < instances.numInstances(); i++) {
			if(i<= instances.numInstances()*0.8) {
				training.add(instances.instance(i));
			}else {
				testing.add(instances.instance(i));
			}
			
		}
		training_testing[0] = training;
		training_testing[1] = testing;
		return training_testing;
	}
}
