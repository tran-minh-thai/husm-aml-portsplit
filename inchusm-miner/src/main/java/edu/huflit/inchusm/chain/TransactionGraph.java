package edu.huflit.inchusm.chain;

import java.io.BufferedReader;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.function.DoublePredicate;

/**
 * Directed transaction multigraph: accounts carry a label, every transaction carries a source,
 * a target, a label and a positive amount. Several transactions may join the same two accounts.
 *
 * <p>Transactions out of (and into) every account are kept sorted by (label, amount, id), so the
 * transactions that conserve the amount of a given transaction are one contiguous range found by
 * binary search on the conservation predicate itself (no division, no tolerance).
 *
 * <p>Self-transactions (source = target) can never be part of a chain occurrence, whose accounts
 * are pairwise distinct; they are dropped at construction and counted.
 *
 * <p>File format ({@code .tg}), one record per line, {@code #} starts a comment line:
 * <pre>
 *   v ID LABEL
 *   e SOURCE TARGET LABEL AMOUNT
 * </pre>
 * Account ids need not be consecutive; they are remapped to 0..n-1 in order of appearance.
 */
public final class TransactionGraph {

    private final int[] vlabel;
    private final int[] src;
    private final int[] dst;
    private final int[] elabel;
    private final double[] amount;
    private final int[] outStart;
    private final int[] outTx;
    private final int[] inStart;
    private final int[] inTx;
    private final int droppedSelfTransactions;
    private final long[] originalIds;
    /** Steps of the binary searches in {@link #outByLabel}, {@link #inByLabel} and {@link #narrow} (a work counter). */
    long probes;

    public TransactionGraph(int[] vlabel, int[] src, int[] dst, int[] elabel, double[] amount) {
        this(vlabel, src, dst, elabel, amount, null);
    }

    /** @param originalIds the id each account had in the input, or null for 0..n-1 */
    public TransactionGraph(int[] vlabel, int[] src, int[] dst, int[] elabel, double[] amount, long[] originalIds) {
        int m = src.length;
        if (dst.length != m || elabel.length != m || amount.length != m) {
            throw new IllegalArgumentException("transaction arrays differ in length");
        }
        this.vlabel = vlabel.clone();
        if (originalIds != null && originalIds.length != vlabel.length) {
            throw new IllegalArgumentException("one original id per account is needed");
        }
        this.originalIds = new long[vlabel.length];
        for (int i = 0; i < vlabel.length; i++) this.originalIds[i] = originalIds == null ? i : originalIds[i];
        for (int i = 0; i < this.vlabel.length; i++) {
            if (this.vlabel[i] < 0) throw new IllegalArgumentException("account " + i + " has a negative label");
        }
        int kept = 0;
        for (int i = 0; i < m; i++) {
            checkTransaction(i, src[i], dst[i], elabel[i], amount[i]);
            if (src[i] != dst[i]) kept++;
        }
        this.src = new int[kept];
        this.dst = new int[kept];
        this.elabel = new int[kept];
        this.amount = new double[kept];
        int j = 0;
        for (int i = 0; i < m; i++) {
            if (src[i] == dst[i]) continue;
            this.src[j] = src[i];
            this.dst[j] = dst[i];
            this.elabel[j] = elabel[i];
            this.amount[j] = amount[i];
            j++;
        }
        this.droppedSelfTransactions = m - kept;
        int n = this.vlabel.length;
        outStart = new int[n + 1];
        inStart = new int[n + 1];
        outTx = csr(this.src, outStart, n);
        inTx = csr(this.dst, inStart, n);
    }

    private void checkTransaction(int i, int s, int t, int lab, double a) {
        int n = vlabel.length;
        if (s < 0 || s >= n || t < 0 || t >= n) {
            throw new IllegalArgumentException("transaction " + i + " names an account outside 0.." + (n - 1));
        }
        if (lab < 0) throw new IllegalArgumentException("transaction " + i + " has a negative label");
        if (!(a > 0) || Double.isInfinite(a)) {
            throw new IllegalArgumentException("transaction " + i + " has amount " + a
                    + "; amounts must be positive and finite (the conservation ratio is undefined otherwise)");
        }
    }

    /** Transaction ids grouped by account, each group sorted by (label, amount, id). */
    private int[] csr(int[] key, int[] start, int n) {
        int m = key.length;
        for (int e = 0; e < m; e++) start[key[e] + 1]++;
        for (int v = 0; v < n; v++) start[v + 1] += start[v];
        int[] out = new int[m];
        int[] fill = Arrays.copyOf(start, n);
        for (int e = 0; e < m; e++) out[fill[key[e]]++] = e;     // ids ascending within each group
        int[] buf = new int[m];
        for (int v = 0; v < n; v++) mergeSort(out, buf, start[v], start[v + 1]);
        return out;
    }

    private int compareTx(int x, int y) {
        if (elabel[x] != elabel[y]) return Integer.compare(elabel[x], elabel[y]);
        int c = Double.compare(amount[x], amount[y]);
        return c != 0 ? c : Integer.compare(x, y);
    }

    /** Stable merge sort of a[from, to) by (label, amount, id), using buf as scratch. */
    private void mergeSort(int[] a, int[] buf, int from, int to) {
        if (to - from < 2) return;
        if (to - from <= 16) {
            for (int i = from + 1; i < to; i++) {
                int x = a[i];
                int j = i - 1;
                while (j >= from && compareTx(a[j], x) > 0) {
                    a[j + 1] = a[j];
                    j--;
                }
                a[j + 1] = x;
            }
            return;
        }
        int mid = (from + to) >>> 1;
        mergeSort(a, buf, from, mid);
        mergeSort(a, buf, mid, to);
        if (compareTx(a[mid - 1], a[mid]) <= 0) return;
        System.arraycopy(a, from, buf, from, to - from);
        int i = from;
        int j = mid;
        int k = from;
        while (i < mid && j < to) a[k++] = compareTx(buf[i], buf[j]) <= 0 ? buf[i++] : buf[j++];
        while (i < mid) a[k++] = buf[i++];
        while (j < to) a[k++] = buf[j++];
    }

    /** Reads a {@code .tg} file line by line into primitive arrays (files of tens of millions of lines). */
    public static TransactionGraph load(Path path) throws IOException {
        Map<Long, Integer> idMap = new HashMap<>();
        LongList ids = new LongList();
        IntList labels = new IntList();
        LongList rawSrc = new LongList();
        LongList rawDst = new LongList();
        IntList edgeLabels = new IntList();
        DoubleList amounts = new DoubleList();
        long lineNo = 0;
        try (BufferedReader in = Files.newBufferedReader(path, StandardCharsets.UTF_8)) {
            String raw;
            while ((raw = in.readLine()) != null) {
                lineNo++;
                String line = raw.trim();
                if (line.isEmpty() || line.startsWith("#")) continue;
                String[] tok = line.split("\\s+");
                switch (tok[0]) {
                    case "v" -> {
                        if (tok.length != 3) throw bad(path, lineNo, "an account line needs 'v ID LABEL'");
                        long id = Long.parseLong(tok[1]);
                        if (idMap.putIfAbsent(id, labels.size()) != null) {
                            throw bad(path, lineNo, "account " + id + " is declared twice");
                        }
                        ids.add(id);
                        labels.add(Integer.parseInt(tok[2]));
                    }
                    case "e" -> {
                        if (tok.length != 5) throw bad(path, lineNo, "a transaction line needs 'e SOURCE TARGET LABEL AMOUNT'");
                        rawSrc.add(Long.parseLong(tok[1]));
                        rawDst.add(Long.parseLong(tok[2]));
                        edgeLabels.add(Integer.parseInt(tok[3]));
                        amounts.add(Double.parseDouble(tok[4]));
                    }
                    default -> throw bad(path, lineNo, "unknown record '" + tok[0] + "'");
                }
            }
        }
        int m = rawSrc.size();
        int[] s = new int[m];
        int[] t = new int[m];
        for (int i = 0; i < m; i++) {
            Integer u = idMap.get(rawSrc.get(i));
            Integer v = idMap.get(rawDst.get(i));
            if (u == null || v == null) {
                throw new IOException(path.getFileName() + ": transaction " + i + " names an undeclared account");
            }
            s[i] = u;
            t[i] = v;
        }
        return new TransactionGraph(labels.toArray(), s, t, edgeLabels.toArray(), amounts.toArray(), ids.toArray());
    }

    private static IOException bad(Path path, long lineNo, String what) {
        return new IOException(path.getFileName() + ":" + lineNo + ": " + what);
    }

    private static final class IntList {
        private int[] a = new int[16];
        private int n;

        void add(int x) {
            if (n == a.length) a = Arrays.copyOf(a, 2 * n);
            a[n++] = x;
        }

        int size() {
            return n;
        }

        int[] toArray() {
            return Arrays.copyOf(a, n);
        }
    }

    private static final class LongList {
        private long[] a = new long[16];
        private int n;

        void add(long x) {
            if (n == a.length) a = Arrays.copyOf(a, 2 * n);
            a[n++] = x;
        }

        long get(int i) {
            return a[i];
        }

        int size() {
            return n;
        }

        long[] toArray() {
            return Arrays.copyOf(a, n);
        }
    }

    private static final class DoubleList {
        private double[] a = new double[16];
        private int n;

        void add(double x) {
            if (n == a.length) a = Arrays.copyOf(a, 2 * n);
            a[n++] = x;
        }

        double[] toArray() {
            return Arrays.copyOf(a, n);
        }
    }

    public int numAccounts() {
        return vlabel.length;
    }

    public int numTransactions() {
        return src.length;
    }

    public int droppedSelfTransactions() {
        return droppedSelfTransactions;
    }

    /** The id account v had in the input file (or v itself when built from arrays). */
    public long originalId(int v) {
        return originalIds[v];
    }

    /** Account index of every original id. */
    public Map<Long, Integer> indexByOriginalId() {
        Map<Long, Integer> out = new HashMap<>();
        for (int v = 0; v < originalIds.length; v++) out.put(originalIds[v], v);
        return out;
    }

    public int accountLabel(int v) {
        return vlabel[v];
    }

    public int source(int e) {
        return src[e];
    }

    public int target(int e) {
        return dst[e];
    }

    public int label(int e) {
        return elabel[e];
    }

    public double amount(int e) {
        return amount[e];
    }

    /** Slot range [from, to) of {@link #outSlot} for every transaction out of v. */
    public int[] outAll(int v) {
        return new int[] {outStart[v], outStart[v + 1]};
    }

    /** Slot range [from, to) of {@link #inSlot} for every transaction into v. */
    public int[] inAll(int v) {
        return new int[] {inStart[v], inStart[v + 1]};
    }

    /** Slot range [from, to) of {@link #outSlot} for transactions out of v with label lab. */
    public int[] outByLabel(int v, int lab) {
        return labelRange(outTx, outStart[v], outStart[v + 1], lab);
    }

    /** Slot range [from, to) of {@link #inSlot} for transactions into v with label lab. */
    public int[] inByLabel(int v, int lab) {
        return labelRange(inTx, inStart[v], inStart[v + 1], lab);
    }

    public int outSlot(int i) {
        return outTx[i];
    }

    public int inSlot(int i) {
        return inTx[i];
    }

    /**
     * Narrow a label range of out-slots (or in-slots) to the amounts satisfying {@code keep}, which
     * must be monotone along the sorted amounts in the sense that the satisfying amounts form one
     * interval: {@code atLeast} is false-then-true, {@code atMost} is true-then-false.
     */
    public int[] narrow(boolean out, int[] range, DoublePredicate atLeast, DoublePredicate atMost) {
        int[] slots = out ? outTx : inTx;
        int lo = range[0];
        int hi = range[1];
        int a = lo;
        int b = hi;
        while (a < b) {                       // first slot whose amount satisfies atLeast
            probes++;
            int mid = (a + b) >>> 1;
            if (atLeast.test(amount[slots[mid]])) b = mid;
            else a = mid + 1;
        }
        int from = a;
        a = from;
        b = hi;
        while (a < b) {                       // first slot whose amount fails atMost
            probes++;
            int mid = (a + b) >>> 1;
            if (atMost.test(amount[slots[mid]])) a = mid + 1;
            else b = mid;
        }
        return new int[] {from, a};
    }

    private int[] labelRange(int[] slots, int lo, int hi, int lab) {
        int a = lo;
        int b = hi;
        while (a < b) {
            probes++;
            int mid = (a + b) >>> 1;
            if (elabel[slots[mid]] < lab) a = mid + 1;
            else b = mid;
        }
        int from = a;
        b = hi;
        while (a < b) {
            probes++;
            int mid = (a + b) >>> 1;
            if (elabel[slots[mid]] <= lab) a = mid + 1;
            else b = mid;
        }
        return new int[] {from, a};
    }
}
