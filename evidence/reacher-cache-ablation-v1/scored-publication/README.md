# Verified release assets

This package preserves the completed scored study and its original scientific gate unchanged. Engineering archives are separate assets and are not scientific results. No upload or release creation has occurred.

The `.partNNN` files are consecutive byte segments, not individually readable tar archives. Concatenate them in the exact `release_assets` order for the scored archive, verify its SHA-256 from `receipt.json`, then open the recovered gzip tar archive.

Archive, part, engineering-asset and manifest hashes are recorded in `receipt.json`. `manifest.json` binds every original member. Upstream historical lineage artifacts may still be required to rerun the saved-output audit.
