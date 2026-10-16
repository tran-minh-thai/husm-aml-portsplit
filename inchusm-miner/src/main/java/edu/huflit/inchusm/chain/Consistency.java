package edu.huflit.inchusm.chain;

import java.util.Arrays;
import java.util.function.IntBinaryOperator;

/**
 * Transaction-level consistency for chain patterns.
 *
 * <p>A slot is the set of transactions that may stand at one pattern edge. Ignoring that the
 * accounts of an occurrence must be distinct, a chain pattern is a constraint problem whose
 * constraint graph is a path over its slots: consecutive slots meet at an account and must
 * conserve the amount there. On a path, one forward pass (keep a transaction only if some
 * transaction of the previous slot ends where it starts and conserves into it) followed by one
 * backward pass (keep it only if some transaction of the next slot continues it) leaves exactly the
 * transactions that lie on a complete conserving walk with the pattern's labels and domains. Every
 * array handled here is a set of transaction ids sorted ascending.
 *
 * <p>Two implementations give the same slots. {@code sorted} sorts the other slot by account and
 * amount and binary-searches it for every transaction. {@code indexed} marks the other slot in a
 * stamp array and scans the conserving range of the graph's adjacency lists, which are already
 * sorted by label and amount (the range the pinned search uses). {@code auto} computes those ranges
 * first and, for each forward or backward call, runs whichever of the two its counts predict to be
 * cheaper: the total length of the ranges against sorting the other slot plus three binary searches
 * per transaction. {@link #ops} counts the work of
 * {@link #slot}, {@link #forward} and {@link #backward} in one unit for both: a comparison made by a
 * sort or a binary search, or a membership test.
 */
final class Consistency {

    /** How forward and backward passes are computed. */
    enum Mode { SORTED, INDEXED, AUTO }

    private final TransactionGraph g;
    private final Mode mode;
    private final boolean indexed;      // slot() marks accounts instead of binary-searching them
    private int[] txStamp;
    private int[] accountStamp;
    private int generation;
    /** Work of slot, forward and backward (see the class comment). */
    long ops;

    /** Passes chosen per call by AUTO: {indexed, sorted}. */
    final long[] autoChoices = new long[2];

    Consistency(TransactionGraph g, Mode mode) {
        this.g = g;
        this.mode = mode;
        this.indexed = mode != Mode.SORTED;
    }

    /** Transactions with label lab whose source lies in from and target in to (sorted accounts). */
    int[] slot(int[] from, int lab, int[] to) {
        long probes = g.probes;
        int stamp = indexed ? mark(accountStamp(), to) : 0;
        int[] out = new int[16];
        int n = 0;
        for (int v : from) {
            int[] r = g.outByLabel(v, lab);
            for (int i = r[0]; i < r[1]; i++) {
                int e = g.outSlot(i);
                boolean inTo = indexed ? isMarked(accountStamp, g.target(e), stamp) : contains(to, g.target(e));
                if (inTo) {
                    if (n == out.length) out = Arrays.copyOf(out, 2 * n);
                    out[n++] = e;
                }
            }
        }
        out = Arrays.copyOf(out, n);
        Arrays.sort(out);
        ops += sortCost(n) + (g.probes - probes);
        return out;
    }

    /** The transactions of next that start where some transaction of prev (all with label prevLab) ends and conserve its amount. */
    int[] forward(int[] prev, int prevLab, int[] next, double theta) {
        if (prev.length == 0 || next.length == 0) return new int[0];
        if (mode == Mode.SORTED) return forwardSorted(prev, next, theta);
        int[][] ranges = forwardRanges(prevLab, next, theta);
        if (mode == Mode.AUTO && !indexedIsCheaper(ranges, prev.length, next.length)) {
            autoChoices[1]++;
            return forwardSorted(prev, next, theta);
        }
        if (mode == Mode.AUTO) autoChoices[0]++;
        return scan(prev, next, ranges, false);
    }

    /** The transactions of cur that some transaction of next (all with label nextLab) continues, conserving the amount. */
    int[] backward(int[] cur, int[] next, int nextLab, double theta) {
        if (cur.length == 0 || next.length == 0) return new int[0];
        if (mode == Mode.SORTED) return backwardSorted(cur, next, theta);
        int[][] ranges = backwardRanges(cur, nextLab, theta);
        if (mode == Mode.AUTO && !indexedIsCheaper(ranges, next.length, cur.length)) {
            autoChoices[1]++;
            return backwardSorted(cur, next, theta);
        }
        if (mode == Mode.AUTO) autoChoices[0]++;
        return scan(next, cur, ranges, true);
    }

    // ---------------------------------------------------------------- indexed

    /** For every transaction of next, the range of in-slots into its source that could conserve into it. */
    private int[][] forwardRanges(int prevLab, int[] next, double theta) {
        long probes = g.probes;
        int[][] ranges = new int[next.length][];
        for (int j = 0; j < next.length; j++) {
            int e = next[j];
            double b = g.amount(e);
            // the same two predicates as the pinned search uses when it chooses an arriving transaction
            ranges[j] = g.narrow(false, g.inByLabel(g.source(e), prevLab), a -> (1 + theta) * a >= b,
                    a -> (1 - theta) * a <= b);
        }
        ops += g.probes - probes;
        return ranges;
    }

    /** For every transaction of cur, the range of out-slots from its target that could continue it. */
    private int[][] backwardRanges(int[] cur, int nextLab, double theta) {
        long probes = g.probes;
        int[][] ranges = new int[cur.length][];
        for (int j = 0; j < cur.length; j++) {
            int e = cur[j];
            double a = g.amount(e);
            double lo = (1 - theta) * a;
            double hi = (1 + theta) * a;
            // the same two predicates as the pinned search uses when it chooses a leaving transaction
            ranges[j] = g.narrow(true, g.outByLabel(g.target(e), nextLab), b -> b >= lo, b -> b <= hi);
        }
        ops += g.probes - probes;
        return ranges;
    }

    /** The predicted work of scanning the ranges against sorting the other slot and searching it three times per transaction. */
    static boolean indexedIsCheaper(int[][] ranges, int other, int kept) {
        long scan = other;                                   // marking the other slot
        for (int[] r : ranges) scan += r[1] - r[0];
        long log = 64 - Long.numberOfLeadingZeros(other);    // ceil(log2(other + 1))
        long sorted = other * log + (long) kept * 3 * log;
        return scan <= sorted;
    }

    /** Keep the transactions whose range holds a transaction of other (in-slots for a forward pass, out-slots for a backward one). */
    private int[] scan(int[] other, int[] candidates, int[][] ranges, boolean out) {
        int stamp = mark(txStamp(), other);
        int[] kept = new int[candidates.length];
        int n = 0;
        for (int j = 0; j < candidates.length; j++) {
            for (int i = ranges[j][0]; i < ranges[j][1]; i++) {
                if (isMarked(txStamp, out ? g.outSlot(i) : g.inSlot(i), stamp)) {
                    kept[n++] = candidates[j];
                    break;
                }
            }
        }
        return Arrays.copyOf(kept, n);
    }

    private int[] txStamp() {
        if (txStamp == null) txStamp = new int[g.numTransactions()];
        return txStamp;
    }

    private int[] accountStamp() {
        if (accountStamp == null) accountStamp = new int[g.numAccounts()];
        return accountStamp;
    }

    /** Stamp every id of set with a fresh generation and return it; nothing is ever cleared. */
    private int mark(int[] stamps, int[] set) {
        if (generation == Integer.MAX_VALUE) {
            Arrays.fill(txStamp(), 0);
            Arrays.fill(accountStamp(), 0);
            generation = 0;
        }
        generation++;
        for (int x : set) stamps[x] = generation;
        ops += set.length;
        return generation;
    }

    private boolean isMarked(int[] stamps, int x, int stamp) {
        ops++;
        return stamps[x] == stamp;
    }

    // ---------------------------------------------------------------- sorted

    private int[] forwardSorted(int[] prev, int[] next, double theta) {
        int[] p = sorted(prev, (x, y) -> {
            int c = Integer.compare(g.target(x), g.target(y));
            return c != 0 ? c : Double.compare(g.amount(x), g.amount(y));
        });
        int[] out = new int[next.length];
        int n = 0;
        for (int e : next) {
            int v = g.source(e);
            double b = g.amount(e);
            int lo = lowerBound(p, i -> g.target(p[i]) >= v);
            int hi = lowerBound(p, i -> g.target(p[i]) > v);
            // smallest arriving amount a with (1 + theta) a >= b; it is the only candidate worth testing
            int k = lowerBoundIn(lo, hi, i -> (1 + theta) * g.amount(p[i]) >= b);
            ops++;
            if (k < hi && (1 - theta) * g.amount(p[k]) <= b) out[n++] = e;
        }
        return Arrays.copyOf(out, n);
    }

    private int[] backwardSorted(int[] cur, int[] next, double theta) {
        int[] q = sorted(next, (x, y) -> {
            int c = Integer.compare(g.source(x), g.source(y));
            return c != 0 ? c : Double.compare(g.amount(x), g.amount(y));
        });
        int[] out = new int[cur.length];
        int n = 0;
        for (int e : cur) {
            int v = g.target(e);
            double a = g.amount(e);
            int lo = lowerBound(q, i -> g.source(q[i]) >= v);
            int hi = lowerBound(q, i -> g.source(q[i]) > v);
            // smallest leaving amount b with b >= (1 - theta) a; it is the only candidate worth testing
            int k = lowerBoundIn(lo, hi, i -> g.amount(q[i]) >= (1 - theta) * a);
            ops++;
            if (k < hi && g.amount(q[k]) <= (1 + theta) * a) out[n++] = e;
        }
        return Arrays.copyOf(out, n);
    }

    private boolean contains(int[] sortedSet, int x) {
        int a = 0;
        int b = sortedSet.length;
        while (a < b) {
            ops++;
            int mid = (a + b) >>> 1;
            if (sortedSet[mid] < x) a = mid + 1;
            else b = mid;
        }
        return a < sortedSet.length && sortedSet[a] == x;
    }

    /** Comparisons charged for sorting n ids (both implementations sort a slot once, so this is a shared estimate). */
    private static long sortCost(int n) {
        return n < 2 ? 0 : (long) n * (64 - Long.numberOfLeadingZeros(n - 1));
    }

    private interface IntPredicate {
        boolean test(int i);
    }

    private int lowerBound(int[] a, IntPredicate atLeast) {
        return lowerBoundIn(0, a.length, atLeast);
    }

    /** First index in [lo, hi) where the monotone predicate holds, or hi. */
    private int lowerBoundIn(int lo, int hi, IntPredicate atLeast) {
        int a = lo;
        int b = hi;
        while (a < b) {
            ops++;
            int mid = (a + b) >>> 1;
            if (atLeast.test(mid)) b = mid;
            else a = mid + 1;
        }
        return a;
    }

    private int[] sorted(int[] ids, IntBinaryOperator cmp) {
        int[] a = ids.clone();
        int[] buf = new int[a.length];
        IntBinaryOperator counted = (x, y) -> {
            ops++;
            return cmp.applyAsInt(x, y);
        };
        mergeSort(a, buf, 0, a.length, counted);
        return a;
    }

    private static void mergeSort(int[] a, int[] buf, int from, int to, IntBinaryOperator cmp) {
        if (to - from < 2) return;
        int mid = (from + to) >>> 1;
        mergeSort(a, buf, from, mid, cmp);
        mergeSort(a, buf, mid, to, cmp);
        if (cmp.applyAsInt(a[mid - 1], a[mid]) <= 0) return;
        System.arraycopy(a, from, buf, from, to - from);
        int i = from;
        int j = mid;
        int k = from;
        while (i < mid && j < to) a[k++] = cmp.applyAsInt(buf[i], buf[j]) <= 0 ? buf[i++] : buf[j++];
        while (i < mid) a[k++] = buf[i++];
        while (j < to) a[k++] = buf[j++];
    }

    // ---------------------------------------------------------------- shared helpers (not counted)

    /** The transactions of slot whose source (or target) lies in the sorted account set keep. */
    static int[] byEnd(TransactionGraph g, int[] slot, int[] keep, boolean source) {
        int[] out = new int[slot.length];
        int n = 0;
        for (int e : slot) {
            if (Arrays.binarySearch(keep, source ? g.source(e) : g.target(e)) >= 0) out[n++] = e;
        }
        return Arrays.copyOf(out, n);
    }

    /** Distinct sources (or targets) of a slot, sorted. */
    static int[] ends(TransactionGraph g, int[] slot, boolean source) {
        int[] v = new int[slot.length];
        for (int i = 0; i < slot.length; i++) v[i] = source ? g.source(slot[i]) : g.target(slot[i]);
        return ChainMiner.distinctSorted(v);
    }

    /** Transactions out of each account of from-transactions' targets (or into their sources) that conserve. */
    static int[] expand(TransactionGraph g, int[] from, int lab, int[] slot, double theta, boolean rightward) {
        int[] out = new int[16];
        int n = 0;
        for (int f : from) {
            int[] r;
            if (rightward) {
                double a = g.amount(f);
                double lo = (1 - theta) * a;
                double hi = (1 + theta) * a;
                r = g.narrow(true, g.outByLabel(g.target(f), lab), b -> b >= lo, b -> b <= hi);
            } else {
                double b = g.amount(f);
                r = g.narrow(false, g.inByLabel(g.source(f), lab), a -> (1 + theta) * a >= b, a -> (1 - theta) * a <= b);
            }
            for (int i = r[0]; i < r[1]; i++) {
                int e = rightward ? g.outSlot(i) : g.inSlot(i);
                if (Arrays.binarySearch(slot, e) >= 0) {
                    if (n == out.length) out = Arrays.copyOf(out, 2 * n);
                    out[n++] = e;
                }
            }
        }
        return ChainMiner.distinctSorted(Arrays.copyOf(out, n));
    }
}
