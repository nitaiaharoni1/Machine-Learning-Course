package HomeWork2;

import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.InstanceComparator;
import weka.core.Instances;

class Node {
	Node[] children;
	Node parent;
	int attributeIndex;
	double returnValue;

}

public class DecisionTree implements Classifier {
	private Node rootNode;

	@Override
	public void buildClassifier(Instances arg0) throws Exception {

	}
    
    @Override
	public double classifyInstance(Instance instance) {

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
