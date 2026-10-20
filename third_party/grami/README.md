# GraMi as an independent reference and as a same-output speed baseline

GraMi (Elseidy, Abdelhamid, Skiadopoulos, Kalnis, "GRAMI: Frequent Subgraph and Pattern Mining in a Single Large
Graph", PVLDB 7(7), 2014) is not part of this repository. It is used from its public repository, commit
`d2875ae80afa2f141323a3f6b89a387787fcda38` of https://github.com/ehab-abdelhamid/GraMi, under its own licence (LGPL).

Building the directed-subgraph miner with a current JDK needs two empty stand-ins, in `stubs/`, for types that GraMi
imports but never uses and that left the JDK in Java 11 (`javax.jws.Oneway`, `javax.rmi.CORBA.Util`):

    git clone https://github.com/ehab-abdelhamid/GraMi && cd GraMi && git checkout d2875ae
    mkdir GRAMI_DIRECTED_SUBGRAPHS/bin
    javac -nowarn -encoding ISO-8859-1 -d GRAMI_DIRECTED_SUBGRAPHS/bin \
        $(find GRAMI_DIRECTED_SUBGRAPHS/src <this directory>/stubs -name "*.java")

GraMi-path is the same miner restricted to directed paths of at most `grami.maxEdges` edges (system property,
default 5): `grami-path.patch` adds one pruning step, placed before frequency evaluation, and changes nothing else.
Apply it to a copy of `GRAMI_DIRECTED_SUBGRAPHS/src` (`patch -p1` from the parent of `GraMi`) and build as above.

`experiments/scripts/grami_crosscheck.py` compares the chain miner (conservation switched off) with GraMi and with
GraMi-path; set `GRAMI_BIN` and `GRAMI_PATH_BIN` to the two `bin` directories.

GraMi-path-nocache is GraMi-path with GraMi's own constant `Settings.CACHING` set to `false` (`grami-nocache.patch`,
one line). With caching on, GraMi passes one shared table of values found not to extend (`nonCandidates`) from a
pattern to the patterns grown from it, instead of a copy; on prefixes of AMLworld HI-Small under labeling A the
published setting reports fewer frequent directed paths than the chain miner and than an independent count from
the definition, and the setting off reports the same paths. `experiments/scripts/grami_speed.py` times the chain
miner against both variants.

`build_grami.py` applies the patches of this directory by exact replacement (no `patch` program; the line endings of
the checkout do not matter), refuses a checkout other than commit `d2875ae`, and compiles both variants:

    git clone https://github.com/ehab-abdelhamid/GraMi && git -C GraMi checkout d2875ae
    python third_party/grami/build_grami.py --grami GraMi --out <directory>
