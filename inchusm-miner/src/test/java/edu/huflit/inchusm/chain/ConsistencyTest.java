package edu.huflit.inchusm.chain;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Arrays;
import java.util.Random;
import org.junit.jupiter.api.Test;

/** The slot operations against a linear scan of their definitions, with amounts on the conservation bounds. */
class ConsistencyTest {

    private static final double[] AMOUNTS = {48, 60, 64, 75, 80, 96, 100, 120, 125, 128, 150, 156.25};

    private static boolean cons(double in, double out, double theta) {
        return (1 - theta) * in <= out && out <= (1 + theta) * in;
    }

    private static int[] sample(Random rnd, int m) {
        return Arrays.stream(rnd.ints(rnd.nextInt(m + 1), 0, m).toArray()).distinct().sorted().toArray();
    }

    @Test
    void forwardBackwardAndExpandEqualTheirDefinitions() {
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
            double theta = new double[] {0.0, 0.2, 0.25, 0.5}[it % 4];
            int[] prev = sample(rnd, m);
            int[] next = sample(rnd, m);
            int lab = rnd.nextInt(2);

            int[] wantF = Arrays.stream(next).filter(f -> Arrays.stream(prev).anyMatch(
                    e -> g.target(e) == g.source(f) && cons(g.amount(e), g.amount(f), theta))).toArray();
            assertArrayEquals(wantF, Consistency.forward(g, prev, next, theta), "forward, case " + it);

            int[] wantB = Arrays.stream(prev).filter(e -> Arrays.stream(next).anyMatch(
                    f -> g.target(e) == g.source(f) && cons(g.amount(e), g.amount(f), theta))).toArray();
            assertArrayEquals(wantB, Consistency.backward(g, prev, next, theta), "backward, case " + it);

            int[] slot = sample(rnd, m);
            int[] wantR = Arrays.stream(slot).filter(f -> g.label(f) == lab && Arrays.stream(prev).anyMatch(
                    e -> g.target(e) == g.source(f) && cons(g.amount(e), g.amount(f), theta))).toArray();
            assertArrayEquals(wantR, Consistency.expand(g, prev, lab, slot, theta, true), "expand right, case " + it);

            int[] wantL = Arrays.stream(slot).filter(e -> g.label(e) == lab && Arrays.stream(prev).anyMatch(
                    f -> g.target(e) == g.source(f) && cons(g.amount(e), g.amount(f), theta))).toArray();
            assertArrayEquals(wantL, Consistency.expand(g, prev, lab, slot, theta, false), "expand left, case " + it);
            cases++;
            if (wantF.length > 0 && wantB.length > 0 && wantR.length > 0 && wantL.length > 0) nonEmpty++;
        }
        System.out.printf("consistency operations: %d cases, %d with all four results non-empty%n", cases, nonEmpty);
        assertTrue(nonEmpty > 100, "non-empty cases: " + nonEmpty);
    }
}
