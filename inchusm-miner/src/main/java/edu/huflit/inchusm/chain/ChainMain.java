package edu.huflit.inchusm.chain;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * Command line for the conserved flow chain miner.
 *
 * <pre>
 * java -cp inchusm-miner.jar edu.huflit.inchusm.chain.ChainMain --graph G.tg --theta 0.2 \
 *      --min-support 10 --max-length 6 [--min-cycle 3] [--no-closed] --out patterns.tsv
 *      [--mark accounts.txt] [--accounts-out roles.tsv]
 * </pre>
 *
 * {@code --no-consistency} skips slot consistency (same output, for measuring what it contributes).
 * {@code --theta inf} switches conservation off (every occurrence is valid), the reference point for
 * measuring what conservation changes. {@code --mark} names a file of account ids, one per line; the
 * output then also counts, for every image set, how many of its accounts are marked. Marking never
 * changes what is mined. {@code --accounts-out} writes, for every kind and length, the accounts that
 * occupy an intermediate position of some reported chain (and the origins of closed chains).
 *
 * The output table has one row per chain whose support reaches the threshold: kind (open or
 * closed), length, position labels, transaction labels, support and the size of every image set.
 * Work counters are printed to standard output; they do not depend on the machine. The one line
 * that does (elapsed time) is marked as such so recording tools can tell it apart.
 */
public final class ChainMain {

    private ChainMain() {
    }

    public static void main(String[] args) throws IOException {
        Map<String, String> opt = parse(args);
        Path graphPath = Path.of(required(opt, "--graph"));
        Path out = Path.of(required(opt, "--out"));
        ChainMiner.Params params = new ChainMiner.Params(
                parseTheta(required(opt, "--theta")),
                Integer.parseInt(required(opt, "--min-support")),
                Integer.parseInt(required(opt, "--max-length")),
                Integer.parseInt(opt.getOrDefault("--min-cycle", "3")),
                !opt.containsKey("--no-closed"));

        TransactionGraph g = TransactionGraph.load(graphPath);
        System.out.printf("graph %s: %d accounts, %d transactions (%d self-transactions dropped)%n",
                graphPath.getFileName(), g.numAccounts(), g.numTransactions(), g.droppedSelfTransactions());
        System.out.printf("params: theta=%s min_support=%d max_length=%d min_cycle=%d closed_chains=%s consistency=%s%n",
                params.theta(), params.minSupport(), params.maxLength(), params.minCycleLength(),
                params.closedChains(), !opt.containsKey("--no-consistency"));

        ChainMiner miner = new ChainMiner(g, params).consistency(!opt.containsKey("--no-consistency"));
        if (opt.containsKey("--mark")) {
            boolean[] marked = readMarked(Path.of(opt.get("--mark")), g);
            int n = 0;
            for (boolean b : marked) if (b) n++;
            System.out.printf("marked accounts present in the graph: %d%n", n);
            miner.marked(marked);
        }
        long t0 = System.nanoTime();
        List<ChainMiner.Result> results = miner.mine();
        long elapsedMs = (System.nanoTime() - t0) / 1_000_000;

        write(out, results);
        if (opt.containsKey("--accounts-out")) writeAccounts(Path.of(opt.get("--accounts-out")), miner, g);
        ChainMiner.Counters c = miner.counters();
        System.out.printf("candidates=%d pruned_by_domain=%d pruned_by_consistency=%d searches=%d found=%d failed=%d"
                + " budget_fallbacks=%d local_rejections=%d early_stops=%d%n", c.candidates, c.prunedByDomain,
                c.prunedByConsistency, c.searches, c.searchesFound, c.searchesFailed, c.budgetFallbacks,
                c.localRejections, c.earlyStops);
        for (Map.Entry<Integer, long[]> e : c.frequentByLength.entrySet()) {
            System.out.printf("length %d: %d open, %d closed%n", e.getKey(), e.getValue()[0], e.getValue()[1]);
        }
        System.out.printf("patterns written: %d -> %s%n", results.size(), out.getFileName());
        System.out.printf("mining elapsed ms: %d (machine-dependent)%n", elapsedMs);
    }

    static void write(Path out, List<ChainMiner.Result> results) throws IOException {
        try (BufferedWriter w = Files.newBufferedWriter(out, StandardCharsets.UTF_8)) {
            w.write("kind\tlength\taccount_labels\ttransaction_labels\tsupport\timage_counts\tmarked_counts\n");
            for (ChainMiner.Result r : results) {
                ChainPattern p = r.pattern();
                w.write((p.closed() ? "closed" : "open") + "\t" + p.length() + "\t" + join(p.vlab()) + "\t"
                        + join(p.elab()) + "\t" + r.support() + "\t" + join(r.imageCounts()) + "\t"
                        + (r.markedCounts() == null ? "" : join(r.markedCounts())) + "\n");
            }
        }
    }

    static void writeAccounts(Path out, ChainMiner miner, TransactionGraph g) throws IOException {
        try (BufferedWriter w = Files.newBufferedWriter(out, StandardCharsets.UTF_8)) {
            w.write("kind\tlength\trole\taccount\n");
            for (Map.Entry<String, java.util.BitSet> e : miner.accountsByRole().entrySet()) {
                String[] key = e.getKey().split(" ");
                java.util.BitSet b = e.getValue();
                for (int v = b.nextSetBit(0); v >= 0; v = b.nextSetBit(v + 1)) {
                    w.write(key[0] + "\t" + key[1] + "\t" + key[2] + "\t" + g.originalId(v) + "\n");
                }
            }
        }
    }

    static boolean[] readMarked(Path path, TransactionGraph g) throws IOException {
        Map<Long, Integer> index = g.indexByOriginalId();
        boolean[] marked = new boolean[g.numAccounts()];
        for (String raw : Files.readAllLines(path, StandardCharsets.UTF_8)) {
            String line = raw.trim();
            if (line.isEmpty() || line.startsWith("#")) continue;
            Integer v = index.get(Long.parseLong(line));
            if (v != null) marked[v] = true;
        }
        return marked;
    }

    static double parseTheta(String text) {
        return text.equalsIgnoreCase("inf") ? Double.POSITIVE_INFINITY : Double.parseDouble(text);
    }

    private static String join(int[] a) {
        return Arrays.stream(a).mapToObj(Integer::toString).collect(Collectors.joining(","));
    }

    private static Map<String, String> parse(String[] args) {
        Map<String, String> opt = new java.util.HashMap<>();
        for (int i = 0; i < args.length; i++) {
            String a = args[i];
            if (!a.startsWith("--")) throw new IllegalArgumentException("unexpected argument '" + a + "'");
            if (a.equals("--no-closed") || a.equals("--no-consistency")) {
                opt.put(a, "");
            } else {
                if (i + 1 >= args.length) throw new IllegalArgumentException(a + " needs a value");
                opt.put(a, args[++i]);
            }
        }
        return opt;
    }

    private static String required(Map<String, String> opt, String key) {
        String v = opt.get(key);
        if (v == null) throw new IllegalArgumentException("missing " + key);
        return v;
    }
}
