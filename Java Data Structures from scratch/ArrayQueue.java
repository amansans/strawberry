import java.util.Arrays;

public class ArrayQueue {
    private int[] items;
    private int count;
    private int front;
    private int rear;

    public ArrayQueue(int capacity) {
        items = new int[capacity];
    }

    public void enqueue(int item){
        if (count == (items.length + 1)) throw new IllegalArgumentException();
        items[rear] = item;
        rear = (rear + 1) % items.length;
        count++;
    }

    public int deque() {
        if (count == 0) throw new IllegalArgumentException();
        var item = items[front];
        items[front] = 0;
        front = (front + 1) % 5;
        count--;
        return item;
    }

    @Override
    public String toString(){
        return Arrays.toString(items);
    }
}
