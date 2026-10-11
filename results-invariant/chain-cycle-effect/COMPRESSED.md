# Compressed account-role files

In every run directory here, `accounts_theta_0.2.tsv` and `accounts_theta_inf.tsv` (the accounts at
every position of the mined chains, open and closed) were written by the run uncompressed and then
stored as `accounts_theta_*.tsv.gz` (`gzip -n -9`, no timestamp, so recompressing gives the same
bytes), because the uncompressed files reach 76 MB. `large_files.sha256` and `large_files.size` give
the sha256 and size of the **uncompressed** files as the run wrote them; every `.gz` was checked to
decompress to exactly those bytes before the uncompressed file was removed:

    gunzip -c accounts_theta_inf.tsv.gz | shasum -a 256

`experiments/scripts/chain_cycle_effect.py` reads only the files its own run writes, so it never
needs the compressed form. The run `20261011T202905` (LI-Medium) was stopped while mining with no
amount constraint; it has the theta = 0.2 files only and no result table.
