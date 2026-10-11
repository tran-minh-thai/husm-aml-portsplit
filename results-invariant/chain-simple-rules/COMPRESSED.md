# Compressed account-role files

In every run directory here, the `accounts_*.tsv` files (the accounts at
intermediate positions of the mined chains) were written by the run uncompressed and then stored as
`accounts_*.tsv.gz` (`gzip -n -9`, no timestamp, so recompressing gives the same bytes), because
the uncompressed files reach 75 MB each. `large_files.sha256` and `large_files.size` give the sha256 and
size of the **uncompressed** files as the run wrote them; every `.gz` was checked to decompress to
exactly those bytes before the uncompressed file was removed:

    gunzip -c accounts_<name>.tsv.gz | shasum -a 256

Readers in `experiments/scripts/` accept either form.
