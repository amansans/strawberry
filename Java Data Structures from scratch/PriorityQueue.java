import java.util.Arrays;

public class PriorityQueue {
    private int count;
    private int [] items = new int[5];
    private int [] newItems;

    public void add(int item) {
        if (isFull()) fixArrayOverflow();
        int i = shiftItemsToInsert(item);

        items[i] = item;
        count++;
    }

    public int remove() {
        if(count == 0) throw new IllegalStateException();
        return items[--count];
    }

    private boolean isFull() {
        return (count == items.length);
    }

    private void fixArrayOverflow() {
        newItems = new int[2 * count];
        for (int i = 0; i < count; i++) {
            newItems[i] = items[i];
        }
        items = newItems;
    }

    private int shiftItemsToInsert(int item){
        int i;
        for (i = count - 1; i >= 0; i--) {
            if (items[i] > item)
                items[i + 1] = items[i];
            else
                break;
        }
        return i + 1;
    }
    @Override
    public String toString(){
        return Arrays.toString(items);
    }
}
