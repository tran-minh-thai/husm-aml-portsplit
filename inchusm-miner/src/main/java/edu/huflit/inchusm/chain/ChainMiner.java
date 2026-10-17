package edu.huflit.inchusm.chain;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.BitSet;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

/**
 * Mines every conserved flow chain whose minimum-image support reaches a threshold.
 *
 * <p><b>Occurrence.</b> An occurrence of a chain pattern of length m is an assignment of distinct
 * accounts to its positions together with one transaction per pattern edge, with the labels of the
 * pattern, such that at every intermediate position i (1..m-1) the transaction that leaves keeps
 * the amount of the one that arrives: {@code (1-theta) a_in <= a_out <= (1+theta) a_in}.
 *
 * <p><b>Support.</b> The image set of a position is the set of accounts it takes over all valid
 * occurrences; the support is the size of the smallest image set (minimum image based support).
 *
 * <p><b>Why the search is exact and level-wise.</b> Restricting a valid occurrence of an open chain
 * to its first m-1 transactions, or to its last m-1, gives a valid occurrence of the prefix
 * (resp. suffix) chain with the same accounts at the shared positions: the conservation conditions
 * kept are a subset of the original ones. Hence the image sets of a chain are contained, position
 * by position, in those of its prefix and of its suffix, and its support is at most theirs. The
 * same holds for a closed chain with respect to the two open chains obtained by removing its first
 * or its last transaction, because the only pair a closed chain does not constrain is the pair at
 * its origin, and removing either of those transactions removes the origin as an intermediate
 * position. So every chain of length m+1 that reaches the threshold is the join of two chains of
 * length m that reach it (prefix and suffix), and its image sets lie in the intersection of theirs.
 * The miner generates exactly these joins and decides every remaining account by a search pinned to
 * that account, so the reported image sets and supports are exact.
 */
public final class ChainMiner {

    public record Params(double theta, int minSupport, int maxLength, int minCycleLength, boolean closedChains) {
        public Params {
            if (!(theta >= 0 && (theta < 1 || theta == Double.POSITIVE_INFINITY))) {
                throw new IllegalArgumentException("theta must lie in [0, 1), or be infinite for no conservation");
            }
            if (minSupport < 1) throw new IllegalArgumentException("minimum support must be at least 1");
            if (maxLength < 1) throw new IllegalArgumentException("maximum length must be at least 1");
            if (minCycleLength < 2) throw new IllegalArgumentException("a closed chain has at least 2 transactions");
        }
    }

    /**
     * One reported chain. markedCounts (accounts of each image set that are marked) is null unless
     * marked accounts were given; images (account indices per position) is null unless kept.
     */
    public record Result(ChainPattern pattern, int support, int[] imageCounts, int[] markedCounts, int[][] images) {
    }

    /** Machine-independent work counters. */
    public static final class Counters {
        public long candidates;
        public long prunedByDomain;
        public long prunedByConsistency;
        public long localRejections;
        public long budgetFallbacks;
        public long searches;
        public long searchesFound;
        public long searchesFailed;
        public long earlyStops;
        /** Work of slot consistency per candidate length (see {@link Consistency#ops}): length -> ops. */
        public final Map<Integer, Long> filterOpsByLength = new TreeMap<>();
        /** Wall time of slot consistency per candidate length, in nanoseconds (machine-dependent; not a work counter). */
        public final Map<Integer, Long> filterNanosByLength = new TreeMap<>();
        /** Failed pinned searches on open chains checked for a conserving walk, and those that had none (should be 0). */
        public long failuresChecked;
        public long failuresWithoutWalk;
        /** Accounts that slot consistency removed on open chains, checked for a conserving walk, and those that had one (should be 0). */
        public long rejectionsChecked;
        public long rejectionsWithWalk;
        /** Transactions examined by pinned searches (every transaction of a scanned range), per candidate length. */
        public final Map<Integer, Long> examinedByLength = new TreeMap<>();
        /** Work of the lazy mode's a priori size (adjacency lookups), per candidate length. */
        public final Map<Integer, Long> lazyOverheadByLength = new TreeMap<>();
        public long lazySwitches;
        /** Checks only: levels found outside their slot or with an eliminated account after a lazy switch (must be 0). */
        public long invalidExpansions;
        /** Per candidate, when recorded: key, then examined, filter ops, lazy overhead, switched (0/1), lazy budget, steps visited, reported (0/1). */
        public final List<String> candidateKeys = new ArrayList<>();
        public final List<long[]> candidateCosts = new ArrayList<>();
        public final Map<Integer, long[]> frequentByLength = new TreeMap<>();   // length -> {open, closed}

        void frequent(ChainPattern p) {
            frequentByLength.computeIfAbsent(p.length(), k -> new long[2])[p.closed() ? 1 : 0]++;
        }
    }

    /** slots (consistent slots per pattern edge 1..m) is kept only by the slot probe. */
    private record Entry(ChainPattern pattern, int[][] images, int[][] slots) {
    }

    /**
     * Slot probe, per candidate length: candidates that reached slot consistency, the total size of
     * the slots built from the domains, of the slots inherited from prefix and suffix (intersection,
     * restricted to the domains), of the consistent slots, consistent-slot transactions outside the
     * inherited slots (zero if inheritance is sound), and the total slot size kept by the frequent open
     * chains of that length. Mining is unchanged by the probe.
     */
    public static final class SlotProbe {
        public final Map<Integer, long[]> byLength = new TreeMap<>();

        long[] at(int length) {
            return byLength.computeIfAbsent(length, k -> new long[6]);
        }
    }

    /** Steps a depth-first attempt on a closed chain may take before local propagation is used. */
    static final long DFS_BUDGET = 10_000;
    private long dfsBudget = DFS_BUDGET;
    private boolean useConsistency = true;
    // slot consistency from this candidate length on: fixed at 4 from the counted work on two probe data sets; any
    // value gives the same output
    private int consistencyFrom = 4;
    private Consistency.Mode filterMode = Consistency.Mode.AUTO;
    private boolean checkFailures;
    private Double lazyLambda;          // lazy slot consistency with budget lambda * (a priori slot size); null: off
    private boolean checkLazy;
    private boolean recordCandidates;
    private Consistency cons;

    /** The conservation predicate between an arriving amount and the amount that leaves. */
    public static boolean conserves(double in, double out, double theta) {
        return out >= (1 - theta) * in && out <= (1 + theta) * in;
    }

    private final TransactionGraph g;
    private final Params params;
    private final Counters counters = new Counters();
    private boolean[] marked;
    private boolean keepImages;
    private SlotProbe slotProbe;
    private final Map<String, BitSet> accounts = new TreeMap<>();

    public ChainMiner(TransactionGraph g, Params params) {
        this.g = g;
        this.params = params;
    }

    /** Accounts to count in every image set (for example accounts known to be laundering); mining ignores them. */
    public ChainMiner marked(boolean[] marked) {
        if (marked.length != g.numAccounts()) throw new IllegalArgumentException("one flag per account is needed");
        this.marked = marked.clone();
        return this;
    }

    /** Switch slot consistency off (plain pinned search), to measure what it contributes; output is the same. */
    public ChainMiner consistency(boolean on) {
        this.useConsistency = on;
        return this;
    }

    /** How slot consistency computes its passes (see {@link Consistency}); the output is the same for every mode. */
    public ChainMiner filterMode(String mode) {
        this.filterMode = Consistency.Mode.valueOf(mode.toUpperCase(java.util.Locale.ROOT));
        return this;
    }

    /** Forward and backward passes that the auto mode ran indexed and sorted. */
    public long[] filterAutoChoices() {
        return cons == null ? new long[2] : cons.autoChoices.clone();
    }

    /**
     * After every failed pinned search on an open chain with slot consistency, search the whole graph for a
     * conserving walk through that account (distinct accounts not required), independently of the slots; a failure
     * without such a walk means slot consistency kept an account it should have removed. Likewise every account that
     * slot consistency removes from an open chain is searched for such a walk; one with a walk means slot consistency
     * removed an account it should have kept. Together they check that the filter is exact for walks. Costs time; for
     * checks only.
     */
    public ChainMiner checkFailures(boolean on) {
        this.checkFailures = on;
        return this;
    }

    /**
     * Lazy slot consistency (rule R2) for open chains: search without slots until lambda times the a
     * priori slot size transactions have been examined, then build the slots and continue. lambda = 0
     * filters first, lambda = infinity never filters. Closed chains are filtered first. Output is the same.
     */
    public ChainMiner lazy(double lambda) {
        if (!(lambda >= 0)) throw new IllegalArgumentException("lambda must be at least 0");
        this.lazyLambda = lambda;
        return this;
    }

    /** Checks only: after a lazy switch, verify at every expansion that the assigned levels are still valid. */
    ChainMiner checkLazy(boolean on) {
        this.checkLazy = on;
        return this;
    }

    /** Keep per-candidate costs in the counters (checks and analysis). */
    public ChainMiner recordCandidates(boolean on) {
        this.recordCandidates = on;
        return this;
    }

    /** Apply slot consistency only to candidates of at least this many transactions (output is the same). */
    public ChainMiner consistencyFrom(int length) {
        if (length < 2) throw new IllegalArgumentException("candidates have at least 2 transactions");
        this.consistencyFrom = length;
        return this;
    }

    /** Budget of the first depth-first attempt on a closed chain; only speed depends on it (tests use 0). */
    ChainMiner dfsBudget(long steps) {
        this.dfsBudget = steps;
        return this;
    }

    /** Record the slot probe (keeps the consistent slots of every open chain of the current level). */
    public ChainMiner slotProbe(boolean on) {
        this.slotProbe = on ? new SlotProbe() : null;
        return this;
    }

    public SlotProbe slotProbeStats() {
        return slotProbe;
    }

    /** Keep the image sets themselves in the results (memory grows with every reported chain). */
    public ChainMiner keepImages(boolean keep) {
        this.keepImages = keep;
        return this;
    }

    /**
     * After {@link #mine()}: for every kind and length, the accounts that occupy an intermediate
     * position (1..m-1) of some reported chain, and for closed chains the accounts at the origin.
     * Keys look like {@code open 3 intermediate} or {@code closed 4 origin}.
     */
    public Map<String, BitSet> accountsByRole() {
        return accounts;
    }

    public Counters counters() {
        return counters;
    }

    public List<Result> mine() {
        cons = new Consistency(g, filterMode);
        List<Result> results = new ArrayList<>();
        List<Entry> level = levelOne();
        for (Entry e : level) results.add(result(e));
        for (int k = 1; k < params.maxLength() && !level.isEmpty(); k++) {
            Map<String, List<Entry>> byPrefix = new HashMap<>();
            for (Entry q : level) byPrefix.computeIfAbsent(prefixKey(q.pattern()), x -> new ArrayList<>()).add(q);
            List<Entry> next = new ArrayList<>();
            for (Entry p : level) {
                List<Entry> qs = byPrefix.get(suffixKey(p.pattern()));
                if (qs == null) continue;
                for (Entry q : qs) {
                    int[] vlab = Arrays.copyOf(p.pattern().vlab(), k + 2);
                    vlab[k + 1] = q.pattern().vlab()[k];
                    int[] elab = Arrays.copyOf(p.pattern().elab(), k + 1);
                    elab[k] = q.pattern().elab()[k - 1];
                    Entry open = evaluate(new ChainPattern(vlab, elab, false), p, q);
                    if (open != null) {
                        next.add(open);
                        results.add(result(open));
                    }
                    if (params.closedChains() && k + 1 >= params.minCycleLength() && vlab[0] == vlab[k + 1]) {
                        Entry closed = evaluate(new ChainPattern(vlab, elab, true), p, q);
                        if (closed != null) results.add(result(closed));   // a closed chain extends no further
                    }
                }
            }
            level = next;
        }
        results.sort(Comparator.comparing((Result r) -> r.pattern().length())
                .thenComparing(r -> r.pattern().key()));
        return results;
    }

    private Result result(Entry e) {
        int[][] img = e.images();
        int[] counts = new int[img.length];
        int[] markedCounts = marked == null ? null : new int[img.length];
        int min = Integer.MAX_VALUE;
        for (int i = 0; i < counts.length; i++) {
            counts[i] = img[i].length;
            min = Math.min(min, counts[i]);
            if (markedCounts != null) {
                for (int v : img[i]) if (marked[v]) markedCounts[i]++;
            }
        }
        ChainPattern p = e.pattern();
        String kind = p.closed() ? "closed " : "open ";
        if (p.length() >= 2) {
            BitSet mid = accounts.computeIfAbsent(kind + p.length() + " intermediate", k -> new BitSet());
            for (int i = 1; i < p.length(); i++) for (int v : img[i]) mid.set(v);
        }
        if (p.closed()) {
            BitSet origin = accounts.computeIfAbsent(kind + p.length() + " origin", k -> new BitSet());
            for (int v : img[0]) origin.set(v);
        }
        counters.frequent(p);
        return new Result(p, min, counts, markedCounts, keepImages ? img : null);
    }

    // ---------------------------------------------------------------- length one

    private List<Entry> levelOne() {
        Map<String, int[]> labels = new TreeMap<>();
        Map<String, List<int[]>> ends = new HashMap<>();
        for (int e = 0; e < g.numTransactions(); e++) {
            int s = g.source(e);
            int t = g.target(e);
            int[] lab = {g.accountLabel(s), g.label(e), g.accountLabel(t)};
            String key = lab[0] + " " + lab[1] + " " + lab[2];
            labels.putIfAbsent(key, lab);
            ends.computeIfAbsent(key, x -> new ArrayList<>()).add(new int[] {s, t});
        }
        List<Entry> out = new ArrayList<>();
        for (Map.Entry<String, int[]> kv : labels.entrySet()) {
            counters.candidates++;
            List<int[]> list = ends.get(kv.getKey());
            int[] s = new int[list.size()];
            int[] t = new int[list.size()];
            for (int i = 0; i < s.length; i++) {
                s[i] = list.get(i)[0];
                t[i] = list.get(i)[1];
            }
            int[][] img = {distinctSorted(s), distinctSorted(t)};
            if (Math.min(img[0].length, img[1].length) < params.minSupport()) {
                counters.prunedByDomain++;
                continue;
            }
            int[] lab = kv.getValue();
            int[][] slots = null;
            if (slotProbe != null) {
                slots = new int[][] {null, cons.slot(img[0], lab[1], img[1])};
                slotProbe.at(1)[5] += slots[1].length;
            }
            out.add(new Entry(new ChainPattern(new int[] {lab[0], lab[2]}, new int[] {lab[1]}, false), img, slots));
        }
        return out;
    }

    // ---------------------------------------------------------------- join keys

    /** Open chain of length k: positions 0..k-1 and transactions 1..k-1. */
    private static String prefixKey(ChainPattern p) {
        return subKey(p, 0, p.length() - 1);
    }

    /** Open chain of length k: positions 1..k and transactions 2..k. */
    private static String suffixKey(ChainPattern p) {
        return subKey(p, 1, p.length() - 1);
    }

    private static String subKey(ChainPattern p, int first, int edges) {
        StringBuilder sb = new StringBuilder().append(p.vlab()[first]);
        for (int i = 0; i < edges; i++) {
            sb.append(" >").append(p.elab()[first + i]).append(' ').append(p.vlab()[first + i + 1]);
        }
        return sb.toString();
    }

    // ---------------------------------------------------------------- candidate evaluation

    /** p is the prefix and q the suffix of the candidate; both have length k. */
    private Entry evaluate(ChainPattern cand, Entry p, Entry q) {
        counters.candidates++;
        int k = p.pattern().length();
        int[][] pi = p.images();
        int[][] qi = q.images();
        int[][] dom = new int[cand.positions()][];
        if (cand.closed()) {
            dom[0] = intersect(pi[0], qi[k]);              // the suffix ends at the origin
        } else {
            dom[0] = pi[0];
            dom[k + 1] = qi[k];
        }
        for (int i = 1; i <= k; i++) dom[i] = intersect(pi[i], qi[i - 1]);
        for (int[] d : dom) {
            if (d.length < params.minSupport()) {
                counters.prunedByDomain++;
                return null;
            }
        }
        boolean filtered = useConsistency && cand.length() >= consistencyFrom;
        Search search = new Search(cand, dom, filtered);
        long overhead = 0;
        boolean lazyHere = filtered && lazyLambda != null && !cand.closed();
        if (lazyHere) {
            long probes = g.probes;
            long u = search.aprioriSize();
            long accounts = 0;
            for (int[] d : dom) accounts += d.length;
            overhead = g.probes - probes + accounts;                 // label lookups plus one addition per domain account
            counters.lazyOverheadByLength.merge(cand.length(), overhead, Long::sum);
            double b = u == 0 ? 0 : lazyLambda * u;              // nothing to scan: filtering first costs nothing
            if (b <= 0) {
                lazyHere = false;                                   // budget 0: filter first
            } else if (Double.isInfinite(b)) {
                lazyHere = false;
                filtered = false;                                   // never filter
            }
            if (lazyHere) search.lazy((long) Math.ceil(b));
        }
        int[][] inherited = slotProbe != null && filtered && !lazyHere && p.slots() != null && q.slots() != null
                ? inherited(cand, dom, p, q) : null;
        boolean consistent = true;
        long opsBefore = cons.ops;
        if (!filtered || lazyHere) {
            search.plain();
        } else {
            long before = cons.ops;
            long t0 = System.nanoTime();
            consistent = search.consistent();
            counters.filterNanosByLength.merge(cand.length(), System.nanoTime() - t0, Long::sum);
            counters.filterOpsByLength.merge(cand.length(), cons.ops - before, Long::sum);
            if (inherited != null) probe(cand, inherited, search);
            if (!consistent) {
                counters.prunedByConsistency++;
                record(cand, search, cons.ops - opsBefore, overhead, false);
                return null;
            }
        }
        int[][] img = search.images();
        counters.examinedByLength.merge(cand.length(), search.examined, Long::sum);
        record(cand, search, cons.ops - opsBefore, overhead, img != null);
        if (img == null) return null;
        int[][] kept = null;
        if (inherited != null && !cand.closed()) {
            kept = search.slots.clone();
            long n = 0;
            for (int s = 1; s < kept.length; s++) n += kept[s].length;
            slotProbe.at(cand.length())[5] += n;
        }
        return new Entry(cand, img, kept);
    }

    private void record(ChainPattern cand, Search search, long filterOps, long overhead, boolean reported) {
        if (!recordCandidates) return;
        counters.candidateKeys.add(cand.key());
        counters.candidateCosts.add(new long[] {search.examined, filterOps, overhead,
                search.lazyBudget > 0 && !search.lazyPending ? 1 : 0, search.lazyBudget, search.stepsVisited,
                reported ? 1 : 0});
    }

    /** Slots of the candidate inherited from its prefix p (edges 1..k) and suffix q (edges 2..k+1), within the domains. */
    private int[][] inherited(ChainPattern cand, int[][] dom, Entry p, Entry q) {
        int m = cand.length();
        int[][] ps = p.slots();
        int[][] qs = q.slots();
        int[][] in = new int[m + 1][];
        for (int s = 1; s <= m; s++) {
            int[] x = s == 1 ? ps[1] : s == m ? qs[m - 1] : intersect(ps[s], qs[s - 1]);
            int to = cand.closed() && s == m ? 0 : s;
            in[s] = Consistency.byEnd(g, Consistency.byEnd(g, x, dom[s - 1], true), dom[to], false);
        }
        return in;
    }

    private void probe(ChainPattern cand, int[][] inherited, Search search) {
        long[] a = slotProbe.at(cand.length());
        a[0]++;
        for (int s = 1; s < inherited.length; s++) {
            a[1] += search.fresh[s];
            a[2] += inherited[s].length;
            a[3] += search.slots[s].length;
            for (int e : search.slots[s]) if (Arrays.binarySearch(inherited[s], e) < 0) a[4]++;
        }
    }

    /** Decides, for every account of every position domain, whether it is in the image set. */
    private final class Search {
        private final int m;
        private final boolean closed;
        private final int[] elab;
        private int[][] dom;
        private final int[][] dom0;      // domains before slot consistency (intersections of prefix and suffix images)
        private BitSet[] rejected;
        private final int[][] slots;    // transactions allowed at each pattern edge 1..m after consistency
        private final int[] fresh;      // slot sizes built from the domains, before propagation (slot probe)
        private int[][] allowed;        // transactions allowed in the current pinned search
        private BitSet[] witnessed;     // per position: accounts already found in an occurrence
        private int[] alive;            // per position: accounts not eliminated
        private boolean slotsActive;    // slots built; searches use them
        private boolean lazyPending;    // lazy mode, slots not built yet
        private long lazyBudget;        // examined transactions after which the lazy mode builds the slots
        long examined;                  // transactions examined by the searches of this candidate (all of a scanned range)
        long stepsVisited;              // steps of the fixed order visited, searched or skipped
        private boolean stopped;        // the candidate cannot reach the minimum support (found at a lazy switch)
        private boolean abortAttempt;   // the pinned account itself was eliminated at a lazy switch
        private int abandonDepth = -1;  // unwinding: frames deeper than this return
        private int pin;                // position of the current pinned search
        private long steps;             // transactions tried in the current depth-first attempt
        private long budget;            // abandon the attempt (not the question) past this many
        private final int[] vert;       // account at each position, -1 if unassigned
        private final int[] tx;         // transaction at each pattern edge 1..m, -1 if unassigned

        private final boolean filtered;   // slot consistency applied to this candidate

        Search(ChainPattern cand, int[][] dom, boolean filtered) {
            this.filtered = filtered;
            this.m = cand.length();
            this.closed = cand.closed();
            this.elab = cand.elab();
            this.dom = dom;
            this.dom0 = dom;
            this.slots = new int[m + 1][];
            this.fresh = new int[m + 1];
            this.vert = new int[dom.length];
            this.tx = new int[m + 1];
        }

        /**
         * Reduce every slot to the transactions that lie on a complete conserving walk with the
         * pattern's labels and the initial domains (and, for a closed chain, one that starts and ends
         * at an account of the origin domain), then record every pair (position, account) of the
         * initial domains that no remaining transaction uses as eliminated; the domains themselves are
         * not replaced, so the order of the steps does not depend on the filter. False when fewer than
         * the minimum support accounts remain at some position.
         */
        boolean consistent() {
            ensureState();
            double theta = params.theta();
            for (int s = 1; s <= m; s++) {
                slots[s] = cons.slot(dom[s - 1], elab[s - 1], dom[pos(s)]);
                fresh[s] = slots[s].length;
            }
            while (true) {
                long before = size();
                for (int s = 2; s <= m; s++) slots[s] = cons.forward(slots[s - 1], elab[s - 2], slots[s], theta);
                for (int s = m - 1; s >= 1; s--) slots[s] = cons.backward(slots[s], slots[s + 1], elab[s], theta);
                if (closed) {
                    int[] origins = intersect(Consistency.ends(g, slots[1], true), Consistency.ends(g, slots[m], false));
                    slots[1] = Consistency.byEnd(g, slots[1], origins, true);
                    slots[m] = Consistency.byEnd(g, slots[m], origins, false);
                }
                if (size() == before) break;
            }
            int[][] d = new int[dom.length][];
            d[0] = Consistency.ends(g, slots[1], true);
            for (int p = 1; p < dom.length; p++) d[p] = Consistency.ends(g, slots[p], false);
            boolean enough = true;
            for (int p = 0; p < dom.length; p++) {
                for (int idx = 0; idx < dom[p].length; idx++) {
                    if (rejected[p].get(idx) || Arrays.binarySearch(d[p], dom[p][idx]) >= 0) continue;
                    if (witnessed[p].get(idx)) throw new IllegalStateException("slot consistency removed an image account");
                    if (checkFailures && !closed) {
                        counters.rejectionsChecked++;
                        if (walkExists(p, dom[p][idx])) counters.rejectionsWithWalk++;
                    }
                    rejected[p].set(idx);
                    alive[p]--;
                }
                if (alive[p] < params.minSupport()) enough = false;
            }
            slotsActive = true;
            return enough;
        }

        /** Without consistency: domains as given, every transaction allowed. */
        void plain() {
            ensureState();
        }

        private void ensureState() {
            if (rejected != null) return;
            rejected = new BitSet[dom.length];
            witnessed = new BitSet[dom.length];
            alive = new int[dom.length];
            for (int i = 0; i < dom.length; i++) {
                rejected[i] = new BitSet(dom[i].length);
                witnessed[i] = new BitSet(dom[i].length);
                alive[i] = dom[i].length;
            }
        }

        /**
         * Lazy slot consistency (rule R2): search without slots until budget transactions have been
         * examined, then build the slots and continue the search in progress on them.
         */
        void lazy(long budget) {
            ensureState();
            lazyBudget = budget;
            lazyPending = true;
        }

        /** Transactions an unconstrained slot build would examine: label-matching out-transactions of every source domain. */
        long aprioriSize() {
            long u = 0;
            for (int s = 1; s <= m; s++) {
                for (int a : dom[s - 1]) {
                    int[] r = g.outByLabel(a, elab[s - 1]);
                    u += r[1] - r[0];
                }
            }
            return u;
        }

        private long size() {
            long n = 0;
            for (int s = 1; s <= m; s++) n += slots[s].length;
            return n;
        }

        /**
         * For a closed chain pinned at account u: the transactions reachable from u going forward to
         * the origin and backward to it, each side conserving, kept only where both sides can meet
         * at one origin account. False when they cannot, which proves u is not in the image set.
         */
        private boolean localClosed(int pin, int u) {
            double theta = params.theta();
            int[][] a = new int[m + 1][];
            int[] at = {u};
            if (pin == 0) {
                a[1] = Consistency.byEnd(g, slots[1], at, true);
                for (int s = 2; s <= m; s++) a[s] = Consistency.expand(g, a[s - 1], elab[s - 1], slots[s], theta, true);
                a[m] = Consistency.byEnd(g, a[m], at, false);
                for (int s = m - 1; s >= 1; s--) a[s] = cons.backward(a[s], a[s + 1], elab[s], theta);
                for (int s = 1; s <= m; s++) if (a[s].length == 0) return false;
                allowed = a;
                return true;
            }
            a[pin + 1] = Consistency.byEnd(g, slots[pin + 1], at, true);
            for (int s = pin + 2; s <= m; s++) a[s] = Consistency.expand(g, a[s - 1], elab[s - 1], slots[s], theta, true);
            a[pin] = Consistency.byEnd(g, slots[pin], at, false);
            for (int s = pin - 1; s >= 1; s--) a[s] = Consistency.expand(g, a[s + 1], elab[s - 1], slots[s], theta, false);
            while (true) {
                long before = 0;
                for (int s = 1; s <= m; s++) before += a[s].length;
                int[] origins = intersect(Consistency.ends(g, a[1], true), Consistency.ends(g, a[m], false));
                int self = Arrays.binarySearch(origins, u);
                if (self >= 0) {                                   // the origin is a different account
                    int[] o = new int[origins.length - 1];
                    System.arraycopy(origins, 0, o, 0, self);
                    System.arraycopy(origins, self + 1, o, self, origins.length - self - 1);
                    origins = o;
                }
                a[1] = Consistency.byEnd(g, a[1], origins, true);
                a[m] = Consistency.byEnd(g, a[m], origins, false);
                for (int s = 2; s <= m; s++) a[s] = cons.forward(a[s - 1], elab[s - 2], a[s], theta);
                for (int s = m - 1; s >= 1; s--) a[s] = cons.backward(a[s], a[s + 1], elab[s], theta);
                long after = 0;
                for (int s = 1; s <= m; s++) {
                    if (a[s].length == 0) return false;
                    after += a[s].length;
                }
                if (after == before) break;
            }
            allowed = a;
            return true;
        }

        /** Exact image sets, or null as soon as one position cannot reach the minimum support. */
        int[][] images() {
            ensureState();
            int n = dom.length;
            Integer[] order = new Integer[n];
            for (int i = 0; i < n; i++) order[i] = i;
            // the step order comes from the domains as given, which slot consistency never replaces
            Arrays.sort(order, Comparator.comparingInt((Integer i) -> dom[i].length).thenComparingInt(i -> i));
            for (int pos : order) {
                for (int idx = 0; idx < dom[pos].length; idx++) {
                    stepsVisited++;
                    if (witnessed[pos].get(idx) || rejected[pos].get(idx)) continue;
                    counters.searches++;
                    boolean found = pinned(pos, dom[pos][idx]);
                    if (stopped) {
                        counters.earlyStops++;
                        return null;
                    }
                    if (found) {
                        counters.searchesFound++;
                        for (int q = 0; q < n; q++) witnessed[q].set(Arrays.binarySearch(dom[q], vert[q]));
                    } else {
                        counters.searchesFailed++;
                        if (rejected[pos].get(idx)) continue;        // eliminated by the filter built during this search
                        if (checkFailures && slotsActive && !closed) {
                            counters.failuresChecked++;
                            if (!walkExists(pos, dom[pos][idx])) counters.failuresWithoutWalk++;
                        }
                        rejected[pos].set(idx);
                        if (--alive[pos] < params.minSupport()) {
                            counters.earlyStops++;
                            return null;
                        }
                    }
                }
            }
            int[][] img = new int[n][];
            for (int q = 0; q < n; q++) {
                img[q] = new int[witnessed[q].cardinality()];
                int j = 0;
                for (int idx = witnessed[q].nextSetBit(0); idx >= 0; idx = witnessed[q].nextSetBit(idx + 1)) {
                    img[q][j++] = dom[q][idx];
                }
            }
            return img;
        }

        /**
         * Is there a conserving walk of an open chain with this pattern's labels through account u at position pin,
         * every other position in its domain before slot consistency? Accounts need not be distinct. Reads the graph
         * directly (every transaction, label and amount tested here), not the slots, so it checks them independently.
         */
        private boolean walkExists(int pin, int u) {
            Map<Long, Boolean> memo = new HashMap<>();
            List<Integer> in = new ArrayList<>();
            List<Integer> out = new ArrayList<>();
            if (pin > 0) {
                for (int i = g.inAll(u)[0]; i < g.inAll(u)[1]; i++) {
                    int f = g.inSlot(i);
                    if (g.label(f) == elab[pin - 1] && member(pin - 1, g.source(f)) && leftOk(pin, f, memo)) in.add(f);
                }
                if (in.isEmpty()) return false;
            }
            if (pin < m) {
                for (int i = g.outAll(u)[0]; i < g.outAll(u)[1]; i++) {
                    int e = g.outSlot(i);
                    if (g.label(e) == elab[pin] && member(pin + 1, g.target(e)) && rightOk(pin + 1, e, memo)) out.add(e);
                }
                if (out.isEmpty()) return false;
            }
            if (pin == 0 || pin == m) return true;
            for (int f : in) {
                for (int e : out) if (conserves(g.amount(f), g.amount(e), params.theta())) return true;
            }
            return false;
        }

        /** Transaction e at edge s (ending at position s) continues to the end of the pattern. */
        private boolean rightOk(int s, int e, Map<Long, Boolean> memo) {
            if (s == m) return true;
            Long key = ((long) s << 32) | e;
            Boolean known = memo.get(key);
            if (known != null) return known;
            boolean ok = false;
            int v = g.target(e);
            for (int i = g.outAll(v)[0]; i < g.outAll(v)[1] && !ok; i++) {
                int f = g.outSlot(i);
                ok = g.label(f) == elab[s] && member(s + 1, g.target(f)) && conserves(g.amount(e), g.amount(f), params.theta())
                        && rightOk(s + 1, f, memo);
            }
            memo.put(key, ok);
            return ok;
        }

        /** Transaction e at edge s (starting at position s - 1) extends back to the start of the pattern. */
        private boolean leftOk(int s, int e, Map<Long, Boolean> memo) {
            if (s == 1) return true;
            Long key = (-(long) s << 32) | e;
            Boolean known = memo.get(key);
            if (known != null) return known;
            boolean ok = false;
            int v = g.source(e);
            for (int i = g.inAll(v)[0]; i < g.inAll(v)[1] && !ok; i++) {
                int f = g.inSlot(i);
                ok = g.label(f) == elab[s - 2] && member(s - 2, g.source(f)) && conserves(g.amount(f), g.amount(e), params.theta())
                        && leftOk(s - 1, f, memo);
            }
            memo.put(key, ok);
            return ok;
        }

        private boolean member(int position, int account) {
            return Arrays.binarySearch(dom0[position], account) >= 0;
        }

        /** Position of pattern vertex s (0..m): the origin when s = m in a closed chain. */
        private int pos(int s) {
            return closed && s == m ? 0 : s;
        }

        /** Is there a valid occurrence with account u at position pin? Leaves it in vert/tx if so. */
        boolean pinned(int pin, int u) {
            if (!slotsActive) {
                allowed = null;
                return attempt(pin, u, Long.MAX_VALUE);
            }
            allowed = slots;
            if (!closed) return attempt(pin, u, Long.MAX_VALUE);
            // A closed chain must come back to its origin, which slot consistency cannot see; a
            // depth-first attempt that runs out of budget is completed after local propagation.
            if (attempt(pin, u, dfsBudget)) return true;
            if (steps <= dfsBudget) return false;                  // the attempt was exhaustive
            counters.budgetFallbacks++;
            if (!localClosed(pin, u)) {
                counters.localRejections++;
                return false;
            }
            return attempt(pin, u, Long.MAX_VALUE);
        }

        /** Depth-first search for an occurrence through u at pin; gives up after budget steps. */
        private boolean attempt(int pin, int u, long budget) {
            this.budget = budget;
            this.steps = 0;
            this.pin = pin;
            abandonDepth = -1;
            abortAttempt = false;
            Arrays.fill(vert, -1);
            Arrays.fill(tx, -1);
            vert[pin] = u;
            return right(pin + 1, pin);
        }

        /**
         * Count one examined transaction; at the lazy budget, build the slots and decide how much of
         * the search in progress survives. False when the frame at this depth must return at once.
         */
        private boolean examine(int depth) {
            examined++;
            // checks only: after a switch, no transaction may be examined below a level that the filter invalidated
            if (checkLazy && !lazyPending && lazyBudget > 0) checkPrefix();
            if (!lazyPending || examined < lazyBudget) return true;
            switchToFilter();
            return !(stopped || abortAttempt || (abandonDepth >= 0 && abandonDepth < depth));
        }

        /**
         * Rule R2: build the slots, record the filtered-out pairs as eliminated, test the early stop,
         * then unwind: find the shallowest assigned level whose transaction is outside its slot or
         * whose account is now eliminated; every deeper frame returns and that level moves on.
         */
        private void switchToFilter() {
            lazyPending = false;
            counters.lazySwitches++;
            long before = cons.ops;
            consistent();
            counters.filterOpsByLength.merge(m, cons.ops - before, Long::sum);
            allowed = slots;
            for (int p = 0; p < dom.length; p++) {
                if (alive[p] < params.minSupport()) {
                    stopped = true;
                    return;
                }
            }
            if (rejected[pin].get(Arrays.binarySearch(dom[pin], vert[pin]))) {
                abortAttempt = true;
                return;
            }
            int depth = 0;
            for (int s = pin + 1; s <= m && tx[s] >= 0; s++, depth++) {
                if (!validStep(s, s)) {
                    abandonDepth = depth;
                    return;
                }
            }
            if (pin + 1 <= m && tx[m] < 0) return;                  // still on the right side
            for (int s = pin; s >= 1 && tx[s] >= 0; s--, depth++) {
                if (!validStep(s, s - 1)) {
                    abandonDepth = depth;
                    return;
                }
            }
        }

        /** Transaction s is in its slot and the account it brought at position p is not eliminated. */
        private boolean validStep(int s, int p) {
            if (Arrays.binarySearch(slots[s], tx[s]) < 0) return false;
            int idx = Arrays.binarySearch(dom[p], vert[p]);
            return idx >= 0 && !rejected[p].get(idx);
        }

        /** After returning from a deeper level: whether this frame continues its loop. */
        private boolean resume(int depth) {
            if (stopped || abortAttempt) return false;
            if (abandonDepth < 0) return true;
            if (abandonDepth < depth) return false;
            if (abandonDepth == depth) abandonDepth = -1;
            return true;
        }

        /** Choose transaction s (from position s-1 to s) for s = pin+1 .. m, then go left. */
        private boolean right(int s, int pin) {
            if (s > m) return left(pin);
            final double theta = params.theta();
            final int depth = s - pin - 1;
            int[] range = g.outByLabel(vert[s - 1], elab[s - 1]);
            if (s - 1 > pin) {                          // intermediate position whose arrival is known
                double a = g.amount(tx[s - 1]);
                double lo = (1 - theta) * a;
                double hi = (1 + theta) * a;
                range = g.narrow(true, range, b -> b >= lo, b -> b <= hi);
            }
            int tp = pos(s);
            for (int i = range[0]; i < range[1]; i++) {
                int e = g.outSlot(i);
                if (!examine(depth)) return false;
                if (allowed != null && Arrays.binarySearch(allowed[s], e) < 0) continue;
                if (++steps > budget) return false;
                int t = g.target(e);
                boolean assign = vert[tp] < 0;
                if (assign ? !admissible(tp, t) : t != vert[tp]) continue;
                if (assign) vert[tp] = t;
                tx[s] = e;
                if (right(s + 1, pin)) return true;
                tx[s] = -1;
                if (assign) vert[tp] = -1;
                if (!resume(depth)) return false;
            }
            return false;
        }

        /** Choose transaction s (from position s-1 to s) for s = pin .. 1. */
        private boolean left(int s) {
            if (s < 1) return true;
            final double theta = params.theta();
            final int depth = (m - pin) + (pin - s);
            int[] range = g.inByLabel(vert[s], elab[s - 1]);
            if (s <= m - 1 && tx[s + 1] >= 0) {        // position s is intermediate; what leaves it is known
                double b = g.amount(tx[s + 1]);
                range = g.narrow(false, range, a -> (1 + theta) * a >= b, a -> (1 - theta) * a <= b);
            }
            int fp = s - 1;
            for (int i = range[0]; i < range[1]; i++) {
                int e = g.inSlot(i);
                if (!examine(depth)) return false;
                if (allowed != null && Arrays.binarySearch(allowed[s], e) < 0) continue;
                if (++steps > budget) return false;
                int f = g.source(e);
                boolean assign = vert[fp] < 0;
                if (assign ? !admissible(fp, f) : f != vert[fp]) continue;
                if (assign) vert[fp] = f;
                tx[s] = e;
                if (left(s - 1)) return true;
                tx[s] = -1;
                if (assign) vert[fp] = -1;
                if (!resume(depth)) return false;
            }
            return false;
        }

        /** Checks only: every assigned level is in its slot with a non-eliminated account (note 07b, check 12). */
        private void checkPrefix() {
            for (int s = pin + 1; s <= m && tx[s] >= 0; s++) if (!validStep(s, s)) counters.invalidExpansions++;
            for (int s = pin; s >= 1 && tx[s] >= 0; s--) if (!validStep(s, s - 1)) counters.invalidExpansions++;
        }

        private boolean admissible(int p, int v) {
            int idx = Arrays.binarySearch(dom[p], v);
            if (idx < 0 || rejected[p].get(idx)) return false;
            for (int w : vert) {
                if (w == v) return false;
            }
            return true;
        }
    }

    // ---------------------------------------------------------------- sorted int sets

    static int[] distinctSorted(int[] a) {
        int[] s = a.clone();
        Arrays.sort(s);
        int j = 0;
        for (int i = 0; i < s.length; i++) {
            if (i == 0 || s[i] != s[i - 1]) s[j++] = s[i];
        }
        return Arrays.copyOf(s, j);
    }

    static int[] intersect(int[] a, int[] b) {
        int[] out = new int[Math.min(a.length, b.length)];
        int i = 0;
        int j = 0;
        int k = 0;
        while (i < a.length && j < b.length) {
            if (a[i] < b[j]) i++;
            else if (a[i] > b[j]) j++;
            else {
                out[k++] = a[i];
                i++;
                j++;
            }
        }
        return Arrays.copyOf(out, k);
    }
}
