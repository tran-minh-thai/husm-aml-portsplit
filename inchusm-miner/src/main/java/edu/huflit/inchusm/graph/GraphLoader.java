package edu.huflit.inchusm.graph;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.TreeSet;

/**
 * Loads graphs in the .lg format used by GraMi/SPMF single-graph datasets:
 *
 * <pre>
 *   t # 0            (optional header, ignored)
 *   v ID LABEL
 *   e U V [LABEL]    (edge label defaults to 0)
 * </pre>
 *
 * Vertex ids need not be consecutive; they are remapped to 0..n-1 preserving order.
 */
public final class GraphLoader {

    private GraphLoader() {
    }

    public static Graph loadLG(Path path) throws IOException {
        Map<Integer, Integer> idMap = new HashMap<>();
        List<Integer> labels = new ArrayList<>();
        List<int[]> edges = new ArrayList<>();
        List<Double> weights = new ArrayList<>();   // optional 5th token on 'e' lines; 1.0 default
        boolean anyWeight = false;
        int lineNo = 0;
        for (String raw : Files.readAllLines(path)) {
            lineNo++;
            String line = raw.trim();
            if (line.isEmpty() || line.startsWith("#") || line.startsWith("t")) continue;
            String[] tok = line.split("\\s+");
            switch (tok[0]) {
                case "v" -> {
                    int id = Integer.parseInt(tok[1]);
                    idMap.put(id, labels.size());
                    labels.add(Integer.parseInt(tok[2]));
                }
                case "e" -> {
                    Integer u = idMap.get(Integer.parseInt(tok[1]));
                    Integer v = idMap.get(Integer.parseInt(tok[2]));
                    if (u == null || v == null) {
                        throw new IOException(path + ":" + lineNo + ": edge references unknown vertex");
                    }
                    int lab = tok.length > 3 ? Integer.parseInt(tok[3]) : 0;
                    edges.add(new int[] {u, v, lab});
                    if (tok.length > 4) {
                        weights.add(Double.parseDouble(tok[4]));
                        anyWeight = true;
                    } else {
                        weights.add(1.0);
                    }
                }
                default -> throw new IOException(path + ":" + lineNo + ": unknown record '" + tok[0] + "'");
            }
        }
        int[] vlabel = new int[labels.size()];
        for (int i = 0; i < vlabel.length; i++) vlabel[i] = labels.get(i);
        double[] w = null;
        if (anyWeight) {
            w = new double[weights.size()];
            for (int i = 0; i < w.length; i++) w[i] = weights.get(i);
        }
        return new Graph(vlabel, edges, w);
    }

    /** Utility table file: one "LABEL VALUE" pair per line, '#' comments allowed. */
    public static Map<Integer, Double> loadUtilityTable(Path path) throws IOException {
        Map<Integer, Double> out = new LinkedHashMap<>();
        for (String raw : Files.readAllLines(path)) {
            String line = raw.trim();
            if (line.isEmpty() || line.startsWith("#")) continue;
            String[] tok = line.split("\\s+");
            out.put(Integer.parseInt(tok[0]), Double.parseDouble(tok[1]));
        }
        return out;
    }

    /**
     * External utilities drawn from N(mu, sigma), clamped at 0.01  -  the convention the
     * HUSM paper uses for datasets without native utilities (mu=3, sigma=1, p. 13).
     * Deterministic per seed; labels are processed in ascending order.
     */
    public static Map<Integer, Double> gaussianUtilities(Collection<Integer> labels,
                                                         double mu, double sigma, long seed) {
        Random rnd = new Random(seed);
        Map<Integer, Double> out = new LinkedHashMap<>();
        for (int label : new TreeSet<>(labels)) {
            out.put(label, Math.max(0.01, mu + sigma * rnd.nextGaussian()));
        }
        return out;
    }
}
