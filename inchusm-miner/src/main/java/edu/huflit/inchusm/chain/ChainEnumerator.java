package edu.huflit.inchusm.chain;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Baseline: enumerate every valid occurrence, then aggregate image sets.
 *
 * <p>This is how instance-finding methods for transaction cycles and flow motifs proceed: a
 * depth-first walk from every transaction that only follows transactions conserving the amount
 * and never revisits an account, recording each occurrence it completes. Image sets are collected
 * per label sequence and the chains whose smallest image set reaches the threshold are reported.
 * It computes the same output as {@link ChainMiner} and exists to measure what the miner's
 * pattern-level pruning saves.
 */
public final class ChainEnumerator {

    private final TransactionGraph g;
    private final ChainMiner.Params params;
    private final Map<String, Acc> byKey = new HashMap<>();
    private final int[] acct;
    private final int[] tx;
    private long occurrences;

    private static final class Acc {
        final ChainPattern pattern;
        final IntSet[] images;

        Acc(ChainPattern pattern) {
            this.pattern = pattern;
            this.images = new IntSet[pattern.positions()];
            for (int i = 0; i < images.length; i++) images[i] = new IntSet();
        }
    }

    public ChainEnumerator(TransactionGraph g, ChainMiner.Params params) {
        this.g = g;
        this.params = params;
        this.acct = new int[params.maxLength() + 1];
        this.tx = new int[params.maxLength() + 1];
    }

    /** Number of valid occurrences visited by the last {@link #mine()}. */
    public long occurrences() {
        return occurrences;
    }

    public List<ChainMiner.Result> mine() {
        byKey.clear();
        occurrences = 0;
        for (int e = 0; e < g.numTransactions(); e++) {
            acct[0] = g.source(e);
            acct[1] = g.target(e);
            tx[1] = e;
            record(1, false);
            extend(1);
        }
        List<ChainMiner.Result> out = new ArrayList<>();
        for (Acc a : byKey.values()) {
            int[] counts = new int[a.images.length];
            int min = Integer.MAX_VALUE;
            for (int i = 0; i < counts.length; i++) {
                counts[i] = a.images[i].size();
                min = Math.min(min, counts[i]);
            }
            if (min >= params.minSupport()) out.add(new ChainMiner.Result(a.pattern, min, counts, null, null));
        }
        out.sort(Comparator.comparing((ChainMiner.Result r) -> r.pattern().length())
                .thenComparing(r -> r.pattern().key()));
        return out;
    }

    /** Extend an occurrence of length k (accounts acct[0..k], transactions tx[1..k]) by one transaction. */
    private void extend(int k) {
        if (k == params.maxLength()) return;
        double theta = params.theta();
        double a = g.amount(tx[k]);
        int v = acct[k];
        for (int lab : outLabels(v)) {
            int[] r = g.narrow(true, g.outByLabel(v, lab), b -> b >= (1 - theta) * a, b -> b <= (1 + theta) * a);
            for (int i = r[0]; i < r[1]; i++) {
                int e = g.outSlot(i);
                int w = g.target(e);
                tx[k + 1] = e;
                if (w == acct[0]) {
                    if (params.closedChains() && k + 1 >= params.minCycleLength()) record(k + 1, true);
                    continue;
                }
                boolean seen = false;
                for (int j = 1; j <= k; j++) {
                    if (acct[j] == w) {
                        seen = true;
                        break;
                    }
                }
                if (seen) continue;
                acct[k + 1] = w;
                record(k + 1, false);
                extend(k + 1);
            }
        }
    }

    /** Labels present among v's out-transactions, ascending (slots are sorted by label first). */
    private int[] outLabels(int v) {
        int[] r = g.outAll(v);
        int[] labs = new int[8];
        int n = 0;
        for (int i = r[0]; i < r[1]; i++) {
            int lab = g.label(g.outSlot(i));
            if (n == 0 || labs[n - 1] != lab) {
                if (n == labs.length) labs = Arrays.copyOf(labs, 2 * n);
                labs[n++] = lab;
            }
        }
        return Arrays.copyOf(labs, n);
    }

    private void record(int m, boolean closed) {
        occurrences++;
        int[] vlab = new int[m + 1];
        int[] elab = new int[m];
        for (int i = 0; i <= m; i++) vlab[i] = g.accountLabel(closed && i == m ? acct[0] : acct[i]);
        for (int i = 1; i <= m; i++) elab[i - 1] = g.label(tx[i]);
        ChainPattern p = new ChainPattern(vlab, elab, closed);
        Acc a = byKey.computeIfAbsent(p.key(), k -> new Acc(p));
        for (int i = 0; i < p.positions(); i++) a.images[i].add(acct[i]);
    }

    /** Open-addressing set of non-negative ints. */
    private static final class IntSet {
        private int[] slots = new int[8];
        private int size;

        IntSet() {
            Arrays.fill(slots, -1);
        }

        void add(int x) {
            if (2 * (size + 1) > slots.length) grow();
            int mask = slots.length - 1;
            int i = (x * 0x9E3779B1) >>> 1 & mask;
            while (slots[i] != -1) {
                if (slots[i] == x) return;
                i = (i + 1) & mask;
            }
            slots[i] = x;
            size++;
        }

        int size() {
            return size;
        }

        private void grow() {
            int[] old = slots;
            slots = new int[old.length * 2];
            Arrays.fill(slots, -1);
            size = 0;
            for (int x : old) if (x != -1) add(x);
        }
    }
}
