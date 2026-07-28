package edu.huflit.inchusm.csp;

import java.util.List;
import java.util.Set;

/**
 * CMNI table (HUSM Def. 7 / Table 1): for every pattern vertex (column), the set
 * of DISTINCT data vertices it maps to over all MER-valid occurrences.
 *
 * CMNI(P) = min over columns of |images|. Column 0 (the initial vertex I0) also
 * feeds the ESV construction for the utility upper bounds.
 */
public final class CMNITable {

    private final List<Set<Integer>> images;

    public CMNITable(List<Set<Integer>> images) {
        this.images = images;
    }

    public int nColumns() {
        return images.size();
    }

    public Set<Integer> images(int v) {
        return images.get(v);
    }

    public int cmni() {
        int min = Integer.MAX_VALUE;
        for (Set<Integer> col : images) min = Math.min(min, col.size());
        return min == Integer.MAX_VALUE ? 0 : min;
    }
}
