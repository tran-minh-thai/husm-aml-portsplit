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
    void forwardBackwardSlotAndExpandEqualTheirDefinitionsInBothImplementations() {
        Random rnd = new Random(20261018L);
        int cases = 0;
        int nonEmpty = 0;
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
            for (boolean indexed : new boolean[] {false, true}) {
                Consistency c = new Consistency(g, indexed);
                String impl = (indexed ? "indexed " : "sorted ") + it;
                assertArrayEquals(wantF, c.forward(prev, labPrev, next, theta), "forward, case " + impl);
                assertArrayEquals(wantB, c.backward(prev, next, labNext, theta), "backward, case " + impl);
                assertArrayEquals(wantS, c.slot(from, lab, to), "slot, case " + impl);
                assertArrayEquals(wantB, c.backward(prev, next, labNext, theta), "backward again, case " + impl);
                assertTrue(c.ops > 0 || prev.length == 0 || next.length == 0 || from.length == 0, "no work counted");
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
    }
}
