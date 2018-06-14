//Nitai Aharoni 203626742
//Nadav Lotan 312346406

package HomeWork2;

import java.util.LinkedList;
import java.util.Queue;

import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.InstanceComparator;
import weka.core.Instances;

class Node {
	Node[] children;
	Node parent;
	int attributeIndex = -1;
	int value = -1;
	double returnValue = -1;

}

public class DecisionTree implements Classifier {
	private Node rootNode;
	private int method = -1;
	private int pValue;
	private int maxHeight;
	private double dataHeight;

	private final double[][] chiTable = {
			/* 1, 0.75, 0.5, 0.25, 0.05, 0.005 */
			/* 1 */{ 0, 0.102, 0.455, 1.323, 3.841, 7.879 }, /* 2 */{ 0, 0.575, 1.386, 2.773, 5.991, 10.597 },
			/* 3 */{ 0, 1.213, 2.366, 4.108, 7.815, 12.838 }, /* 4 */{ 0, 1.923, 3.357, 5.385, 9.488, 14.860 },
			/* 5 */{ 0, 2.675, 4.351, 6.626, 11.070, 16.750 }, /* 6 */{ 0, 3.455, 5.348, 7.841, 12.592, 18.548 },
			/* 7 */{ 0, 4.255, 6.346, 9.037, 14.067, 20.278 }, /* 8 */{ 0, 5.071, 7.344, 10.219, 15.507, 21.955 },
			/* 9 */{ 0, 5.899, 8.343, 11.389, 16.919, 23.589 }, /* 10 */{ 0, 6.737, 9.342, 12.549, 18.307, 25.188 },
			/* 11 */{ 0, 7.584, 10.341, 13.701, 19.675, 26.757 },
			/* 12 */{ 0, 8.438, 11.340, 14.485, 21.026, 28.300 } };

	public void setMethod(int method) {
		this.method = method;
	}

	public void setPValue(int pValue) {
		this.pValue = pValue;
	}

	public int getMaxHeight() {
		return maxHeight;
	}

	public double getPValue() {
		double[] table = { 1, 0.75, 0.5, 0.25, 0.05, 0.005 };
		return table[pValue];
	}

	public double getDataHeight() {
		return dataHeight;
	}

	@Override
	public void buildClassifier(Instances arg0) throws Exception {
		this.rootNode = new Node();
		this.rootNode.value = -1;
		buildTree(arg0);
	}

	@Override
	public double classifyInstance(Instance instance) {
		Node curr = rootNode;
		int height = 0;
		while (curr.children != null) {
			for (int i = 0; i < curr.children.length; i++) {
				if (instance.value(curr.attributeIndex) == (double) i) {
					// if no child - return parent's return value
					if (curr.children[i] == null) {
						this.dataHeight += height;
						return curr.returnValue;
					}
					height++;
					if (height > this.maxHeight) {
						this.maxHeight = height;
					}
					curr = curr.children[i];
					break;
				}
			}

		}
		this.dataHeight += height;
		return curr.returnValue;
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

	private void buildTree(Instances data) {
		Queue<Node> queue = new LinkedList<Node>();
		queue.add(rootNode);
		Node curr;
		double maxGain;
		double gain;
		Instances nodeData = data;
		int bestAttribute;
		double[] valuesArray;
		int deg;
		boolean prune;
		do {
			maxGain = 0;
			deg = 0;
			bestAttribute = -1;
			curr = queue.peek();
			nodeData = distributeInstances(data, curr);
			curr.returnValue = returnValue(nodeData);
			if (perfectlyClassified(nodeData)) {
				queue.poll();
				continue;
			}
			if (sameAtts(nodeData)) {
				queue.poll();
				continue;
			}
			for (int i = 0; i < data.numAttributes() - 1; i++) {
				gain = calcGain(nodeData, i);
				if (gain > maxGain) {
					maxGain = gain;
					bestAttribute = i;
				}
			}
			curr.attributeIndex = bestAttribute;
			Node[] children = new Node[nodeData.attribute(bestAttribute).numValues()];
			curr.children = children;
			valuesArray = nodeData.attributeToDoubleArray(bestAttribute);
			for (int i = 0; i < children.length; i++) {
				if (arrayContains(valuesArray, i)) {
					deg++;
					curr.children[i] = new Node();
					curr.children[i].parent = curr;
					curr.children[i].value = i;
				}
			}
			prune = prune(nodeData, bestAttribute, this.pValue, deg-2);
			if (prune) {
				curr.children = null;
			}else {
				for (int i = 0; i < curr.children.length; i++) {
					if (curr.children[i] != null) {
						queue.add(curr.children[i]);
					}
				}
			}
			queue.poll();
		} while (queue.peek() != null);
	}

	//all instances with the same class value
	private boolean perfectlyClassified(Instances data) {
		double[] prob = calcProb(data);
		if ((prob[0] == 0 && prob[1] == 1) || (prob[0] == 1 && prob[1] == 0))
			return true;
		else
			return false;
	}

	//checks if array contains value
	private boolean arrayContains(double[] array, double value) {
		for (int i = 0; i < array.length; i++) {
			if (array[i] == value) {
				return true;
			}
		}
		return false;
	}
	
	//all attributes are the same except class attribute
	private boolean sameAtts(Instances nodeData) {
		for (int k = 0; k < nodeData.numAttributes() - 1; k++) {
			if (nodeData.numDistinctValues(k) > 1)
				return false;
		}
		return true;
	}

	// calc return value
	private double returnValue(Instances data) {
		double[] prob = calcProb(data);
		// if (prob[1] == 0.5) we chose to returnValue 1.0
		return Math.round(prob[1]);
	}

	// calculates Entropy
	private double calcEntropy(double[] probabilities) {
		double sum = 0;
		for (int i = 0; i < probabilities.length; i++) {
			if (probabilities[i] == 0) {
				continue;
			} else {
				sum += probabilities[i] * Math.log(probabilities[i]);
			}
		}
		return sum * (-1);
	}

	// calculates Gini
	private double calcGini(double[] probabilities) {
		double sum = 0;
		for (int i = 0; i < probabilities.length; i++) {
			sum += Math.pow(probabilities[i], 2);
		}
		return (1 - sum);
	}

	// calculates Gain
	private double calcGain(Instances data, int attIndex) {
		double sum = 0;
		double[] childProb = new double[2];
		double numChildInstances;
		double numDataInstances;
		// info gain
		if (method == 0) {
			double parentEntropy = calcEntropy(calcProb(data));
			for (int i = 0; i < data.attribute(attIndex).numValues(); i++) {
				Instances child = getChild(data, i, attIndex);
				numChildInstances = child.numInstances();
				numDataInstances = data.numInstances();
				if (data.numInstances() != 0 && child.numInstances() != 0) {
					childProb = calcProb(child);
					sum += (numChildInstances / numDataInstances) * (calcEntropy(childProb));
				}
			}
			return parentEntropy - sum;
		} else {
			// gini gain
			double parentGini = calcGini(calcProb(data));
			for (int i = 0; i < data.attribute(attIndex).numValues(); i++) {
				Instances child = getChild(data, i, attIndex);
				numChildInstances = child.numInstances();
				numDataInstances = data.numInstances();
				if (data.numInstances() != 0 && child.numInstances() != 0) {
					childProb = calcProb(child);
					sum += (numChildInstances / numDataInstances) * (calcGini(childProb));
				}
			}
			return parentGini - sum;
		}
	}

	private Instances distributeInstances(Instances data, Node node) {
		Instances distributed = data;
		Node curr = node;
		LinkedList<Node> nodesList = new LinkedList<Node>();
		while (curr.parent != null) {
			nodesList.addFirst(curr);
			curr = curr.parent;
		}
		while (!nodesList.isEmpty()) {
			distributed = getChild(distributed, nodesList.getFirst().value, curr.attributeIndex);
			curr = nodesList.pollFirst();
			if (distributed.numInstances() == 0) {
				break;
			}
		}
		return distributed;
	}

	// set of instances with specific attribute
	private Instances getChild(Instances data, int val, int attIndex) {
		Instances child = new Instances(data, 0);
		for (int i = 0; i < data.numInstances(); i++) {
			if (data.instance(i).value(attIndex) == val || val == -1) {
				child.add(data.instance(i));
			}
		}
		return child;
	}

	// calculates node probabilities
	private double[] calcProb(Instances data) {
		double[] classProbabilities = new double[2];
		if (data.numInstances() != 0) {
			double numRecurrence = 0;
			for (int i = 0; i < data.numInstances(); i++) {
				if (data.instance(i).classValue() == 0) {
					numRecurrence++;
				}
			}
			classProbabilities[0] = numRecurrence / data.numInstances();
			classProbabilities[1] = 1 - classProbabilities[0];
		}
		return classProbabilities;
	}

	public double calcAvgError(Instances data) {
		this.maxHeight = 0;
		this.dataHeight = 0;

		double avgErr = 0;
		int numInstances = data.numInstances();
		for (int i = 0; i < numInstances; i++) {
			avgErr += Math.abs(classifyInstance(data.instance(i)) - data.instance(i).classValue());
		}
		return (avgErr / numInstances);
	}

	public boolean prune(Instances data, int attIndex, int pValue, int deg) {
		double chiTableVal;
		double chiSquare;
		chiTableVal = chiTable(data, attIndex, pValue, deg);
		chiSquare = chiSquare(data, attIndex);
		if (chiSquare < chiTableVal) {
			return true;
		} else {
			return false;
		}
	}

	private double chiSquare(Instances data, int attIndex) {
		int Df;
		int Pf;
		int Nf;
		double E0;
		double E1;
		double chi = 0;
		double[] prob = calcProb(data);
		for (int i = 0; i < data.attribute(attIndex).numValues(); i++) {
			Df = 0;
			Pf = 0;
			Nf = 0;
			E0 = 0;
			E1 = 0;
			for (int j = 0; j < data.numInstances(); j++) {
				if (data.instance(j).value(attIndex) == i) {
					Df++;
					if (data.instance(j).classValue() == 0)
						Pf++;
					if (data.instance(j).classValue() == 1)
						Nf++;
				}
			}
			if (Df != 0) {
				E0 = Df * prob[0];
				E1 = Df * prob[1];
				chi += (Math.pow((Pf - E0), 2) / E0) + (Math.pow((Nf - E1), 2) / E1);
			}
		}
		return chi;
	}

	private double chiTable(Instances data, int attIndex, int pValue, int deg) {
		double val = chiTable[deg][pValue];
		return val;
	}

	public void printTree() {
		printTree("", rootNode);
	}

	private void printTree(String tab, Node node) {
		if (node.children == null) {
			System.out.println(tab + "Leaf. Returning value: " + node.returnValue);
		} else {
			if (node.parent == null) {
				System.out.println("Root");
				System.out.println("Returning value: " + node.returnValue);
				tab = tab + "\t";
			}
			for (int i = 0; i < node.children.length; i++) {
				if (node.children[i] != null) {
					System.out.println(
							tab + "If attribute " + node.attributeIndex + " = " + node.children[i].value);
					if (node.children[i].children == null) {
						System.out.println(tab + "\tLeaf. Returning value: " + node.returnValue);
					} else {
						System.out.println(tab + "Returning value: " + node.children[i].returnValue);
						printTree(tab + "\t", node.children[i]);
					}
				}
			}
		}
	}

	public void printTreeTest(Instances data) {
		printTreeTest(data, "", rootNode);
	}

	private void printTreeTest(Instances data, String tab, Node node) {
		if (node.children != null) {
			if (node.parent == null) {
				System.out.println("Root - " + data.attribute(node.attributeIndex).name() + " - ("
						+ distributeInstances(data, node).numInstances() + ")");
				tab = tab + "|\t";
			}
			for (int i = 0; i < node.children.length; i++) {
				if (node.children[i] != null) {
					System.out.println(tab + data.attribute(node.children[i].parent.attributeIndex).name() + " = "
							+ data.attribute(node.children[i].parent.attributeIndex).value(node.children[i].value)
							+ ": " + data.attribute(9).value((int) node.children[i].returnValue) + " ("
							+ distributeInstances(data, node.children[i]).numInstances() + ")");
					printTreeTest(data, tab + "|\t", node.children[i]);

				}
			}
		}
	}

}
