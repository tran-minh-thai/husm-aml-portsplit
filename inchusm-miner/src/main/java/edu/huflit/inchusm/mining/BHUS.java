package edu.huflit.inchusm.mining;

import edu.huflit.inchusm.mining.HUSM.PatternResult;

import java.util.Collection;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.Set;

/**
 * Boundary High Utility Subgraphs (HUSM Def. 15, Alg. 5  -  post-processing form).
 *
 * MaxBHUS: no child in the result set (child = code one edge longer having this
 * code as prefix). MinBHUS: parent (code minus last edge) not in the result set;
 * 1-edge patterns are children of the root, which is a HUS by convention, so
 * they can only be MaxBHUS.
 *
 * Note: because utility is not monotone on the enumeration
 * tree, HUS may form BROKEN paths; lossless recovery (Alg. 6) is only guaranteed
 * along all-HUS segments, which is why extraction reports both boundary sets.
 */
public final class BHUS {

    public record Sets(Set<String> maxBoundary, Set<String> minBoundary) {

        public Set<String> all() {
            Set<String> union = new LinkedHashSet<>(maxBoundary);
            union.addAll(minBoundary);
            return union;
        }
    }

    private BHUS() {
    }

    public static Sets extract(Collection<PatternResult> hus) {
        Set<String> keys = new HashSet<>();
        for (PatternResult p : hus) keys.add(p.code().key());

        Set<String> withChild = new HashSet<>();
        for (PatternResult p : hus) {
            if (p.code().size() >= 2) {
                String parent = p.code().parentKey();
                if (keys.contains(parent)) withChild.add(parent);
            }
        }

        Set<String> maxB = new LinkedHashSet<>();
        Set<String> minB = new LinkedHashSet<>();
        for (PatternResult p : hus) {
            String key = p.code().key();
            if (!withChild.contains(key)) maxB.add(key);
            if (p.code().size() >= 2 && !keys.contains(p.code().parentKey())) minB.add(key);
        }
        return new Sets(maxB, minB);
    }
}
