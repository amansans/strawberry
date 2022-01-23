import java.util.ArrayDeque;
import java.util.Queue;

public class BinaryTree {
    private BinaryNode root;

    private class BinaryNode {
        private int value;
        private BinaryNode leftChild;
        private BinaryNode rightChild;

        public BinaryNode(int value) {
            this.value = value;
        }

        @Override
        public String toString(){
            return "Node: " + value;
        }
    }

    public void insert(int value) {
        BinaryNode node = new BinaryNode(value);

        if(root == null) {
            root = node;
            return;
        }

        var current = root;

        while(true) {
            if (value >  current.value) {
                if(current.rightChild == null) {
                    current.rightChild = node;
                    break;
                }
                current = current.rightChild;
            }
            else {
                if(current.leftChild == null) {
                    current.leftChild = node;
                    break;
                }
                current = current.leftChild;
            }
        }
    }

    public BinaryNode getRoot() {
        return root;
    }

    public boolean find(int value) {
        var current = root;

        while(current != null) {
            if (value >  current.value)  {
                current = current.rightChild;
            }
            else if (value < current.value) {
                current = current.leftChild;
            }
            else return true;
        }
        return false;
    }

    public void postOrderRecursion(BinaryNode node){
        if(node == null) return;

        postOrderRecursion(node.leftChild);
        postOrderRecursion(node.rightChild);
        System.out.println(node.value);
    }

    public void preOrderRecursion(BinaryNode node){
        if(node == null) return;

        System.out.println(node.value);
        preOrderRecursion(node.leftChild);
        preOrderRecursion(node.rightChild);
    }

    public void inOrderRecursion(BinaryNode node){
        if(node == null) return;

        inOrderRecursion(node.leftChild);
        System.out.println(node.value);
        inOrderRecursion(node.rightChild);
    }

    public int minimumValue(){
        if(root == null) throw new IllegalStateException();

        var current = root;
        var last = current;
        while(current != null) {
            last = current;
            current = current.leftChild;
        }
        return last.value;
    }

    public void depthFirstRecursion(BinaryNode node){
        Queue<BinaryNode> queue = new ArrayDeque<>();
        queue.add(node);

        while(!queue.isEmpty()) {
            var previousValue = queue.remove();
            System.out.println(previousValue);

            if(previousValue.leftChild != null) queue.add(previousValue.leftChild);
            if(previousValue.rightChild != null) queue.add(previousValue.rightChild);
        }
    }

    public int height(BinaryNode node) {
        if (node == null) return  -1;
        if (isLeaf(node)) return 0;
        return 1 + Math.max(height(node.leftChild),height(node.rightChild));
    }

    private boolean isLeaf(BinaryNode node) {
        return (node.leftChild == null && node.rightChild == null);
    }

    public boolean equals(BinaryTree tree) {
        return equals(root,tree.root);
    }

    private boolean equals(BinaryNode first, BinaryNode second) {
        if (first == null && second == null) {
            return true;
        }

        if (first != null && second != null) {
            return (first.value == second.value) && equals(first.leftChild,second.leftChild) && equals(first.rightChild, second.rightChild);
        }

        return false;
    }

    public void nodesAtKthDistance(BinaryNode node, int distance) {
        if (node == null)
            return;

        if(distance == 0){
            System.out.println(node.value);
            return;
        }

        nodesAtKthDistance(node.leftChild,distance-1 );
        nodesAtKthDistance(node.rightChild,distance-1 );

    }
//    public int min() {
//        return min(root);
//    }

//    private int min(BinaryNode node){
//        if (isLeaf(node)) return node.value;
//
//        var left = min(node.leftChild);
//        var right = min(node.rightChild);
//
//        return Math.min(Math.min(left,right),node.value);
//    }
}
