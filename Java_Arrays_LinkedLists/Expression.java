import java.util.Arrays;
import java.util.List;
import java.util.Stack;

public class Expression {
    private final List<Character> leftBrackets = Arrays.asList('(','[','<','{');
    private final List<Character> rightBrackets = Arrays.asList(')',']','>','}');
    private String input = new String();

    public Expression(String input){
        this.input = input;
    }

    public boolean isBalanced() {
        Stack<Character> stack = new Stack<>();
        for (char ch: input.toCharArray()) {
            if (isLeft(ch))
                stack.push(ch);
            if (isRight(ch)) {
                if (stack.isEmpty()) return false;
                var top = stack.pop();
                if (notBalanced(top,ch)) return false;
            }
        }
        return stack.empty();
    }

    private boolean isLeft(char ch){
        return leftBrackets.contains(ch);
    }

    private boolean isRight(char ch){
        return rightBrackets.contains(ch);
    }

    private boolean notBalanced(char left, char right){
        return leftBrackets.indexOf(left) == rightBrackets.indexOf(right);
    }
}
