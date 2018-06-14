package HomeWork2;

import java.util.ArrayList;
import java.util.List;

import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

class BasicRule {
	int attributeIndex;
	int attributeValue;
}

class Rule {
	List<BasicRule> basicRule;
	double returnValue;
}

class Node {
	Node[] children;
	Node parent;
	int attributeIndex;
	double returnValue;
	Rule nodeRule = new Rule();

}

public class DecisionTree implements Classifier {
	private Node rootNode;
	final double THRESHOLD = 15.51;

	public enum PruningMode {
		None, Chi, Rule
	};

	private PruningMode m_pruningMode;
	Instances validationSet;
	private List<Rule> rules = new ArrayList<Rule>();

	/**
	 * The function builds a decision tree from the training data.
	 * 
	 */
	@Override
	public void buildClassifier(Instances arg0) throws Exception {

		// initialization
		rules.clear();
		rootNode = new Node();
		rootNode.parent = null;
		rootNode.nodeRule.basicRule = new ArrayList<BasicRule>();
		buildTree(arg0, rootNode);
		
		// check for rule pruning
		if (m_pruningMode == PruningMode.Rule) {
			rulePruning();
		}
	}

	public void setPruningMode(PruningMode pruningMode) {
		m_pruningMode = pruningMode;
	}

	public void setValidation(Instances validation) {
		validationSet = validation;
	}

	/**
	 * The function return the classification of the instance
	 * 
	 */
	@Override
	public double classifyInstance(Instance instance) {
		int maxMatchedRules = 0;
		int counter = 0;
		double[] count = new double[2];
		for (int i = 0; i < rules.size(); i++) {
			for (int j = 0; j < rules.get(i).basicRule.size(); j++) {
				if (instance.value(rules.get(i).basicRule.get(j).attributeIndex) != rules
						.get(i).basicRule.get(j).attributeValue) {
					// fill count array with classification
					if (counter == maxMatchedRules) {
						count[(int) rules.get(i).returnValue]++;
					// reset count array
					} else if (counter > maxMatchedRules) {
						maxMatchedRules = counter;
						count[0] = 0;
						count[1] = 0;
						count[(int) rules.get(i).returnValue]++;
					}
					break;
					// get max classification, with means the instance is matching the rule
				} else if (counter == rules.get(i).basicRule.size() - 1) {
					return rules.get(i).returnValue;
				}
				counter++;
			}
			counter = 0;
		}
		// select random classification
		if (count[0] == count[1]) {
			return count[0] % 2;
		}
		// yes classification
		if (count[0] > count[1]) {
			return 0;
		}
		// no classification
		return 1;

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

	/**
	 * 
	 * The function builds the decision tree on given data set using recursion
	 * 
	 * @param data
	 * @param node
	 */
	private void buildTree(Instances data, Node node) {
		if (data.numInstances() == 0) {
			return;
		}
		boolean classify = allAttributesHaveTheSameValue(data);
		if (classify) {
			node.returnValue = data.instance(0).classValue();
			node.nodeRule.returnValue = data.instance(0).classValue();
			rules.add(node.nodeRule);
			return;
		}
		double[] probabilities = calcValueClassProbabilities(data,
				data.classIndex());
		// leaf
		if (probabilities[0] == 0.0) {
			rules.add(node.nodeRule);
			node.returnValue = 1;
			node.nodeRule.returnValue = 1;
			return;
			// leaf
		} else if (probabilities[0] == 1.0) {
			rules.add(node.nodeRule);
			node.returnValue = 0;
			node.nodeRule.returnValue = 0;
			return;
			// select calss by probability
		} else if (probabilities[0] >= 0.5) {
			node.returnValue = 0;
			node.nodeRule.returnValue = 0;
		} else {
			node.returnValue = 1;
			node.nodeRule.returnValue = 1;
		}

		int maxInfoGainAttribute = currentMaxInfoGain(data);

		// check for valid attributeIndex
		if (maxInfoGainAttribute == -1) {
			rules.add(node.nodeRule);
			return;
		}

		node.attributeIndex = maxInfoGainAttribute;
		// check for chi pruning
		if (m_pruningMode == PruningMode.Chi) {
			double chiSquare = calcChiSquare(data, maxInfoGainAttribute);
			if (chiSquare < THRESHOLD) {
				rules.add(node.nodeRule);
				return;
			}
		}
		int numOfAttributeValues = data.attribute(maxInfoGainAttribute)
				.numValues();
		node.children = new Node[numOfAttributeValues];
		Instances[] instanceSplitByAttributeValue = splitByAttributeValues(
				data, maxInfoGainAttribute);
		for (int i = 0; i < numOfAttributeValues; i++) {
			node.children[i] = new Node();
			node.children[i].parent = node;
			BasicRule basicRule = new BasicRule();
			basicRule.attributeIndex = maxInfoGainAttribute;
			basicRule.attributeValue = i;
			List<BasicRule> parent = new ArrayList<BasicRule>(
					node.nodeRule.basicRule);
			node.children[i].nodeRule.basicRule = parent;
			node.children[i].nodeRule.basicRule.add(basicRule);
			buildTree(instanceSplitByAttributeValue[i], node.children[i]);
		}
		return;
	}

	/**
	 * The function receives a set of instances, and returns true if all
	 * attributes in all instances have the same value, False otherwise.
	 * 
	 * @param data
	 * @return
	 */
	private boolean allAttributesHaveTheSameValue(Instances data) {
		double currentAttributeValue = 0.0;
		for (int i = 0; i < data.numAttributes(); i++) {
			currentAttributeValue = data.instance(0).value(i);
			for (int j = 1; j < data.numInstances(); j++) {
				if (data.instance(j).value(i) != currentAttributeValue) {
					return false;
				}
			}
		}
		return true;
	}

	/**
	 * 
	 * The function gets a set of instances and an attribute value. The function
	 * returns a new set of instances with the same attribute value.
	 * 
	 * @param data
	 * @param value
	 * @param attributeIndex
	 * @return
	 */
	private Instances getValueSubset(Instances data, int value,
			int attributeIndex) {
		Instances subset = new Instances(data, 0);
		for (int i = 0; i < data.numInstances(); i++) {
			if (data.instance(i).value(attributeIndex) == value) {
				subset.add(data.instance(i));
			}
		}
		return subset;
	}

	/**
	 * 
	 * The function receives a set of data and an attribute index The function
	 * return an array of Instances where every attribute value, i, has set of
	 * instances with the same value in index i.
	 * 
	 * @param data
	 * @param attributeIndex
	 * @return
	 */
	private Instances[] splitByAttributeValues(Instances data, int attributeIndex) {
		int numOfValues = data.attribute(attributeIndex).numValues();
		Instances[] splitValues = new Instances[numOfValues];
		for (int i = 0; i < numOfValues; i++) {
			splitValues[i] = getValueSubset(data, i, attributeIndex);
		}
		return splitValues;
	}

	/**
	 * 
	 * The function receives a set of data and an attribute / class index. The
	 * function returns the probabilities partition of the attribute's values.
	 * 
	 * @param data
	 * @param attributeIndex
	 * @return
	 */
	private double[] calcAttributeProbabilities(Instances data,
			int attributeIndex) {
		double[] valuesPartition = new double[data.attribute(attributeIndex)
				.numValues()];

		for (int i = 0; i < data.numInstances(); i++) {
			// calculate the number of occurrences of each value
			double value = data.instance(i).value(attributeIndex);
			valuesPartition[(int) value]++;
		}

		for (int i = 0; i < valuesPartition.length; i++) {
			// calculate the number of occurrences of each value, divided by num
			// of instances
			valuesPartition[i] = (double) valuesPartition[i]
					/ (double) data.numInstances();
		}
		return valuesPartition;
	}

	/**
	 * 
	 * The function receives a set of instances and an index attribute (could be
	 * classIndex) The function return a classification probabilities.
	 * 
	 * @param data
	 * @param attributeIndex
	 * @return
	 */
	private double[] calcValueClassProbabilities(Instances data,int attributeIndex) {
		double[] classProbabilities = new double[2];
		if (data.numInstances() != 0) {

			int numOfYes = 0;
			for (int i = 0; i < data.numInstances(); i++) {
				if (data.instance(i).classValue() == 0) {
					numOfYes++;
				}
			}
			classProbabilities[0] = ((double) numOfYes)
					/ ((double) data.numInstances());
			classProbabilities[1] = 1 - classProbabilities[0];
		}
		return classProbabilities;
	}

	/**
	 * 
	 * The function calculates the entropy of a random variable where all the
	 * probabilities of all of the possible values it can take are given as
	 * input.
	 * 
	 * @param probabilities
	 * @return
	 */
	private double calcEntropy(double[] probabilities) {
		double result = 0;
		for (int i = 0; i < probabilities.length; i++) {
			if (probabilities[i] == 0) {
				continue;
			} else {
				result += probabilities[i]
						* (Math.log(probabilities[i]) / Math.log(2));
			}
		}
		return (-1.0) * result;
	}

	/**
	 * 
	 * calculates the information gain of splitting the input data according to
	 * the attribute
	 * 
	 * @param data
	 * @param attributeIndex
	 * @return
	 */
	private double calcInfoGain(Instances data, int attributeIndex) {
		double mainEntropy = 0;
		mainEntropy = calcEntropy(calcValueClassProbabilities(data,
				data.classIndex()));

		double[] attributeValuesProbabilities = calcAttributeProbabilities(
				data, attributeIndex);
		double sigma = 0.0;
		for (int i = 0; i < data.attribute(attributeIndex).numValues(); i++) {
			Instances subset = getValueSubset(data, i, attributeIndex);
			double[] subsetProbabilities = calcValueClassProbabilities(subset,
					attributeIndex);
			sigma += attributeValuesProbabilities[i]
					* calcEntropy(subsetProbabilities);
		}
		return mainEntropy - sigma;
	}

	/**
	 * 
	 * The function receives a set of instances, and calculates the attribute
	 * with the max info gain value. The function returns the attribute index.
	 * 
	 * @param data
	 * @return
	 */
	private int currentMaxInfoGain(Instances data) {
		double maxInfoGain = 0.0;
		double currentInfoGain = 0.0;
		int resultIndex = -1;
		for (int i = 0; i < data.numAttributes() - 1; i++) {
			currentInfoGain = calcInfoGain(data, i);
			if (currentInfoGain > maxInfoGain) {
				maxInfoGain = currentInfoGain;
				resultIndex = i;
			}
		}
		return resultIndex;
	}

	/**
	 * 
	 * The function calculates the chi square statistic of splitting the data
	 * according to attributeIndex.
	 * 
	 * @param data
	 * @param attributeIndex
	 * @return
	 */
	private double calcChiSquare(Instances data, int attributeIndex) {
		double x2 = 0.0;
		Instances value_i;
		int sizeOfValue_i;
		int numOfYesForValue;
		int numOfNoForValue;
		double E0, E1;
		double component1, component2;

		double[] probabilities;
		probabilities = calcValueClassProbabilities(data, attributeIndex);

		double probabilityForYes = probabilities[0];
		double probabilityForNo = 1 - probabilityForYes;

		for (int i = 0; i < data.attribute(attributeIndex).numValues(); i++) {
			value_i = getValueSubset(data, i, attributeIndex);
			sizeOfValue_i = value_i.size();
			probabilities = calcValueClassProbabilities(value_i, attributeIndex);
			numOfYesForValue = (int) (probabilities[0] * value_i.numInstances());
			numOfNoForValue = value_i.numInstances() - numOfYesForValue;
			E0 = sizeOfValue_i * probabilityForYes;
			E1 = sizeOfValue_i * probabilityForNo;
			if (E0 == 0) {
				component1 = 0;
			} else {
				component1 = (Math.pow((E0 - numOfYesForValue), 2) / E0);
			}
			if (E1 == 0) {
				component2 = 0;
			} else {
				component2 = (Math.pow((E1 - numOfNoForValue), 2) / E1);
			}
			x2 += component1 + component2;

		}
		return x2;
	}

	/**
	 * The function calculates the average on a given instances set.
	 * 
	 * @param data
	 * @return
	 */
	public double calcAvgError(Instances data) {
		double avgError = 0.0;
		for (int i = 0; i < data.numInstances(); i++) {
			avgError += Math.abs(data.instance(i).classValue()
					- classifyInstance(data.instance(i)));
		}
		return ((double) avgError) / ((double) data.numInstances());
	}

	/**
	 * The function checks if removing a rule will improve the result. Picks the
	 * best rule to remove according to the error on the validation set and
	 * remove it from the rule set. The function stops removing rules when there
	 * is no improvement.
	 * 
	 * @param data
	 */
	private void rulePruning() {
		List<Rule> newRules;
		double allRulesAvgError = calcAvgError(validationSet);
		double currentError = 1, prevError = 2;
		double maxDiffError = -1;
		Rule ruleToRemove = null;

		while (currentError < prevError) {
			newRules = (List<Rule>) (new ArrayList<Rule>(rules));
			allRulesAvgError = calcAvgError(validationSet);
			prevError = currentError;
			maxDiffError = Double.MIN_VALUE;
			ruleToRemove = null;

			for (Rule rule : newRules) {
				rules.remove(rule);
				double tempAvgError = calcAvgError(validationSet);
				double currentDiffError = allRulesAvgError - tempAvgError;
				if (currentDiffError > maxDiffError) {
					ruleToRemove = rule;
					maxDiffError = currentDiffError;
					currentError = tempAvgError;
				}
				rules.add(rule);
			}
			// remove the rule with max diff
			if (currentError < prevError) {
				rules.remove(ruleToRemove);
			}
		}
	}

	/**
	 * The function returns number of rules
	 */
	public int GetNumRules() {
		return rules.size();
	}
}
