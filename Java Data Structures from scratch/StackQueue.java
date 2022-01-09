import java.util.Arrays;
import java.util.Stack;

public class StackQueue {
    private Stack<Integer> forwardStack = new Stack<>();
    private Stack<Integer> reversedStack = new Stack<>();

    public void enqueue(int item) {
        forwardStack.push(item);
    }

    public int dequeue() {
        moveStacks();
        return  reversedStack.pop();
    }

    public int peek() {
        moveStacks();
        return reversedStack.peek();
    }

    private void moveStacks() {
        if (reversedStack.isEmpty()) {
            if (forwardStack.isEmpty()) throw new IllegalStateException();
            while (!forwardStack.isEmpty())
                reversedStack.push(forwardStack.pop());
        }
    }
}
