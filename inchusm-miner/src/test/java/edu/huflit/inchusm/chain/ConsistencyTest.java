package edu.huflit.inchusm.chain;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Arrays;
import java.util.Random;
import org.junit.jupiter.api.Test;

/** The slot operations of both implementations against a linear scan of their definitions, with amounts on the conservation bounds. */
class ConsistencyTest {

    private static final double[] AMOUNTS = {48, 60, 64, 75, 80, 96, 100, 120, 125, 128, 150, 156.25};
    private static final double[] THETAS = {0.0, 0.2, 0.25, 0.5, Double.POSITIVE_INFINITY};

    private static boolean cons(double in, double out, double theta) {
        return (1 - theta) * in <= out && out <= (1 + theta) * in;
    }

    private static int[] sample(Random rnd, int m) {
        return Arrays.stream(rnd.ints(rnd.nextInt(m + 1), 0, m).toArray()).distinct().sorted().toArray();
    }

    private static int[] withLabel(TransactionGraph g, int[] set, int lab) {
        return Arrays.stream(set).filter(e -> g.label(e) == lab).toArray();
    }

    @Test
    void predictedWorkMatchesHandCounts() {
        // scan: mark 4 + ranges 30 = 34; sorted: 4 * ceil(log2 5) = 12 to sort, 3 * 3 * 3 = 27 to search, 39 in all
        int[][] ranges = {{0, 10}, {0, 10}, {5, 15}};
        assertTrue(Consistency.indexedIsCheaper(ranges, 4, 3));
        // scan 4 + 42 = 46 > 39
        int[][] longer = {{0, 14}, {0, 14}, {0, 14}};
        assertTrue(!Consistency.indexedIsCheaper(longer, 4, 3));
    }

    /**
     * A hub with many arriving and leaving transactions of one amount, and a single transaction in the other slot: the
     * conserving ranges are long and almost empty of marked transactions, so auto must sort instead, and still be right.
     */
    @Test
    void autoSortsWhenTheRangesAreLongAndTheOtherSlotIsSmall() {
        int hub = 0;
        int k = 300;
        int[] vlab = new int[2 * k + 1];
        int[] s = new int[2 * k];
        int[] t = new int[2 * k];
        int[] l = new int[2 * k];
        double[] a = new double[2 * k];
        for (int i = 0; i < k; i++) {
            s[i] = 1 + i;                // arriving at the hub
            t[i] = hub;
            s[k + i] = hub;              // leaving the hub
            t[k + i] = k + 1 + i;
            a[i] = 100;
            a[k + i] = 100 + (i % 3);
        }
        TransactionGraph g = new TransactionGraph(vlab, s, t, l, a);
        int[] arriving = java.util.stream.IntStream.range(0, k).toArray();
        int[] leaving = java.util.stream.IntStream.range(k, 2 * k).toArray();
        for (double theta : THETAS) {
            int[] oneIn = {arriving[7]};
            int[] oneOut = {leaving[5]};
            int[] wantF = Arrays.stream(leaving).filter(f -> cons(100, g.amount(f), theta)).toArray();
            int[] wantB = Arrays.stream(arriving).filter(e -> cons(g.amount(e), g.amount(leaving[5]), theta)).toArray();
            for (Consistency.Mode mode : Consistency.Mode.values()) {
                Consistency c = new Consistency(g, mode);
                assertArrayEquals(wantF, c.forward(oneIn, 0, leaving, theta), mode + " forward, theta " + theta);
                assertArrayEquals(wantB, c.backward(arriving, oneOut, 0, theta), mode + " backward, theta " + theta);
                if (mode == Consistency.Mode.AUTO) {
                    assertTrue(c.autoChoices[1] == 2 && c.autoChoices[0] == 0,
                            "auto should sort both passes: " + Arrays.toString(c.autoChoices));
                }
            }
        }
    }

    @Test
    void forwardBackwardSlotAndExpandEqualTheirDefinitionsInBothImplementations() {
        Random rnd = new Random(20261018L);
        int cases = 0;
        int nonEmpty = 0;
        long autoIndexed = 0;
        long autoSorted = 0;
        for (int it = 0; it < 400; it++) {
            int n = 4 + rnd.nextInt(4);
            int m = 20 + rnd.nextInt(40);
            int[] vlab = new int[n];
            int[] s = new int[m];
            int[] t = new int[m];
            int[] l = new int[m];
            double[] a = new double[m];
            for (int e = 0; e < m; e++) {
                s[e] = rnd.nextInt(n);
                t[e] = (s[e] + 1 + rnd.nextInt(n - 1)) % n;
                l[e] = rnd.nextInt(2);
                a[e] = AMOUNTS[rnd.nextInt(AMOUNTS.length)];
            }
            TransactionGraph g = new TransactionGraph(vlab, s, t, l, a);
            double theta = THETAS[it % THETAS.length];
            int lab = rnd.nextInt(2);
            int labPrev = rnd.nextInt(2);
            int labNext = rnd.nextInt(2);
            int[] prev = withLabel(g, sample(rnd, m), labPrev);
            int[] next = withLabel(g, sample(rnd, m), labNext);
            int[] from = Arrays.stream(rnd.ints(rnd.nextInt(n + 1), 0, n).toArray()).distinct().sorted().toArray();
            int[] to = Arrays.stream(rnd.ints(rnd.nextInt(n + 1), 0, n).toArray()).distinct().sorted().toArray();

            int[] wantF = Arrays.stream(next).filter(f -> Arrays.stream(prev).anyMatch(
                    e -> g.target(e) == g.source(f) && cons(g.amount(e), g.amount(f), theta))).toArray();
            int[] wantB = Arrays.stream(prev).filter(e -> Arrays.stream(next).anyMatch(
                    f -> g.target(e) == g.source(f) && cons(g.amount(e), g.amount(f), theta))).toArray();
            int[] wantS = java.util.stream.IntStream.range(0, m).filter(e -> g.label(e) == lab
                    && Arrays.binarySearch(from, g.source(e)) >= 0 && Arrays.binarySearch(to, g.target(e)) >= 0).toArray();
            // four calls on one instance, so the stamps of one call are still in the arrays at the next
            for (Consistency.Mode mode : Consistency.Mode.values()) {
                Consistency c = new Consistency(g, mode);
                String impl = mode + " " + it;
                assertArrayEquals(wantF, c.forward(prev, labPrev, next, theta), "forward, case " + impl);
                assertArrayEquals(wantB, c.backward(prev, next, labNext, theta), "backward, case " + impl);
                assertArrayEquals(wantS, c.slot(from, lab, to), "slot, case " + impl);
                assertArrayEquals(wantB, c.backward(prev, next, labNext, theta), "backward again, case " + impl);
                assertTrue(c.ops > 0 || prev.length == 0 || next.length == 0 || from.length == 0, "no work counted");
                autoIndexed += c.autoChoices[0];
                autoSorted += c.autoChoices[1];
            }

            int[] slot = sample(rnd, m);
            int[] wantR = Arrays.stream(slot).filter(f -> g.label(f) == lab && Arrays.stream(prev).anyMatch(
                    e -> g.target(e) == g.source(f) && cons(g.amount(e), g.amount(f), theta))).toArray();
            assertArrayEquals(wantR, Consistency.expand(g, prev, lab, slot, theta, true), "expand right, case " + it);

            int[] wantL = Arrays.stream(slot).filter(e -> g.label(e) == lab && Arrays.stream(prev).anyMatch(
                    f -> g.target(e) == g.source(f) && cons(g.amount(e), g.amount(f), theta))).toArray();
            assertArrayEquals(wantL, Consistency.expand(g, prev, lab, slot, theta, false), "expand left, case " + it);
            cases++;
            if (wantF.length > 0 && wantB.length > 0 && wantR.length > 0 && wantL.length > 0 && wantS.length > 0) nonEmpty++;
        }
        System.out.printf("consistency operations: %d cases, %d with all five results non-empty%n", cases, nonEmpty);
        assertTrue(nonEmpty > 60, "non-empty cases: " + nonEmpty);
        // small random graphs rarely make sorting cheaper; that branch is forced in the hub test below
        System.out.printf("auto mode passes: %d indexed, %d sorted%n", autoIndexed, autoSorted);
        assertTrue(autoIndexed > 50, "auto passes ran indexed: " + autoIndexed);
    }
}
