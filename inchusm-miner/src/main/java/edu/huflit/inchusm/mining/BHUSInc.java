package edu.huflit.inchusm.mining;

import edu.huflit.inchusm.dfs.DFSCode;

import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.Set;

/**
 * Incremental Boundary High-Utility Subgraphs: the streaming form of {@link BHUS}.
 *
 * {@link BHUS#extract} recomputes both boundaries from the full HUS set. Under add-only
 * snapshots the high set only grows (IncHUSM invariant: once high, always high), so the
 * boundaries can be maintained per snapshot from N_t (the patterns that reached the
 * threshold for the first time at this snapshot), touching only the newly-high patterns
 * and their parents and children instead of scanning the whole HUS set.
 *
 * Definitions match {@link BHUS}:
 *   MaxBHUS  a HUS with no child (a code one edge longer, this code as prefix) in the set;
 *   MinBHUS  a HUS of at least two edges whose parent (last edge removed) is not in the set.
 * A 1-edge code is a child of the root (a HUS by convention), so it can only be MaxBHUS.
 *
 * Because utility is not monotone on the enumeration tree, a child may become high before
 * its parent, so absorb handles either order. Absorbing all of HUS in any order yields
 * exactly {@link BHUS#extract} of that set; BHUSIncTest asserts this.
 *
 * Cost: each newly-high pattern updates a constant number of sets, so a snapshot costs
 * O(|N_t|) (plus the reverse-index children touched once over the pattern lifetime),
 * against O(|HUS|) for a full recompute.
 */
public final class BHUSInc {

    private final Set<String> keys = new HashSet<>();                      // all HUS keys so far
    private final Map<String, Integer> childCount = new HashMap<>();       // key -> #children in keys
    private final Map<String, Set<String>> childrenOf = new HashMap<>();   // parentKey -> child keys
    private final Set<String> maxBoundary = new LinkedHashSet<>();
    private final Set<String> minBoundary = new LinkedHashSet<>();

    /**
     * Fold one snapshot's newly-high pattern codes (N_t) into the maintained boundaries.
     * IncHUSM.update returns List&lt;PatternState&gt;; pass {@code ps.code()} of each.
     */
    public void absorb(Collection<DFSCode> newlyHigh) {
        for (DFSCode code : newlyHigh) {
            String key = code.key();
            if (!keys.add(key)) continue;                 // monotone: never re-add

            int size = code.size();
            String parent = size >= 2 ? code.parentKey() : null;

            // (1) this pattern's own boundary status
            if (childCount.getOrDefault(key, 0) == 0) maxBoundary.add(key);     // no child yet -> Max
            if (parent != null && !keys.contains(parent)) minBoundary.add(key); // parent absent -> Min

            // (2) the parent gains a child: record it, and it is no longer Max
            if (parent != null) {
                childCount.merge(parent, 1, Integer::sum);
                childrenOf.computeIfAbsent(parent, k -> new HashSet<>()).add(key);
                if (keys.contains(parent)) maxBoundary.remove(parent);
            }

            // (3) children of this pattern already in the set are no longer Min (parent now present)
            Set<String> kids = childrenOf.get(key);
            if (kids != null) for (String child : kids) minBoundary.remove(child);
        }
    }

    /** Current compact representation (a copy, safe to keep as per-snapshot telemetry). */
    public BHUS.Sets snapshot() {
        return new BHUS.Sets(new LinkedHashSet<>(maxBoundary), new LinkedHashSet<>(minBoundary));
    }

    public int maxSize() {
        return maxBoundary.size();
    }

    public int minSize() {
        return minBoundary.size();
    }

    public int highTotal() {
        return keys.size();
    }
}
