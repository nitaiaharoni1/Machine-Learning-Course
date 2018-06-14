//Nitai Aharoni 203626742
//Nadav Lotan 312346406

package HomeWork2;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

import weka.core.Instances;

public class MainHW2 {

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

	@SuppressWarnings("unused")
	public static void main(String[] args) throws Exception {
		Instances trainingCancer = loadData("cancer_train.txt");
		Instances testingCancer = loadData("cancer_test.txt");
		Instances validationCancer = loadData("cancer_validation.txt");

		// TODO: complete the Main method
		DecisionTree decisionTreeEntropy = new DecisionTree();
		decisionTreeEntropy.setMethod(0);
		DecisionTree decisionTreeGini = new DecisionTree();
		decisionTreeGini.setMethod(1);

		decisionTreeEntropy.buildClassifier(trainingCancer);
		double valErrorEntropy = decisionTreeEntropy.calcAvgError(validationCancer);


		decisionTreeGini.buildClassifier(trainingCancer);
		double valErrorGini = decisionTreeGini.calcAvgError(validationCancer);


		System.out.println("Validation error using Entropy: " + valErrorEntropy);
		System.out.println("Validation error using Gini: " + valErrorGini);

		DecisionTree decisionTree;
		int method = -1;
		if (valErrorGini >= valErrorEntropy)
			method = 0;
		else
			method = 1;
		int bestPValue = -1;
		double bestValidationErr = Integer.MAX_VALUE;
		double TestErrBestTree = Integer.MAX_VALUE;
		double validationErr;
		double testErr;
		for (int i = 0; i < 6; i++) {
			validationErr = 0;
			testErr = 0;
			decisionTree = new DecisionTree();
			decisionTree.setMethod(method);
			decisionTree.setPValue(i);
			decisionTree.buildClassifier(trainingCancer);
			System.out.println("----------------------------------------------------");
			System.out.println("Decision Tree with P_value of: " + decisionTree.getPValue());
			testErr = decisionTree.calcAvgError(trainingCancer);
			System.out.printf("The train error of the decision tree is: %.3f\n", testErr);
			validationErr = decisionTree.calcAvgError(validationCancer);
			System.out.println("Max height on validation data: " + decisionTree.getMaxHeight());
			System.out.println("Average height on validation data: " + decisionTree.getDataHeight()/(validationCancer.numInstances()));
			System.out.println("The validation error of the decision tree is: " + validationErr);
			if (validationErr< bestValidationErr) {
				bestPValue = i;
				bestValidationErr  = validationErr;
				TestErrBestTree = testErr;
			}
		}
		DecisionTree bestDecisionTree = new DecisionTree();
		bestDecisionTree.setMethod(method);
		bestDecisionTree.setPValue(bestPValue);
		bestDecisionTree.buildClassifier(trainingCancer);

		System.out.println("----------------------------------------------------");
		System.out.println("Best validation error at p_value = " + bestDecisionTree.getPValue());
		System.out.println("Test error with best tree: " + TestErrBestTree);
		bestDecisionTree.printTree();
		//bestDecisionTree.printTreeTest(trainingCancer);
	}
}
