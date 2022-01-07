//Vector grows by 100% of its size when it gets filled - Sync
//Arrays grow by 50% of its size when it gets filled - Async

import java.util.Arrays;

public class DataStructures {
    public static void main(String[] args) {
        Array items = new Array(3);
        items.insert(10);
        items.insert(20);
        items.insert(30);
        items.insert(40);
        items.insert(50);
        items.removeAt(1);
        System.out.println("\n\t\t Arrays \n");
        System.out.println("Index:" + items.indexOf(40));
        items.print();
        System.out.println("\n\t\t Linked Lists \n");

        LinkedList list = new LinkedList();
        list.addLast(10);
        list.addLast(20);
        list.addLast(30);
        list.addFirst(5);
        list.removeFirst();
        list.removeLast();
        System.out.println(list.indexOf(20));
        System.out.println(list.contains(50));
    }
}
