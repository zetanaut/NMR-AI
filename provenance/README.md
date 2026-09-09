# Measurement provenance archive

All working examples in `examples/` use the confirmed 500-bin grid.
This directory preserves original source evidence for reproducibility.

`uva-nd3-measured.json` is the byte-identical original UVA-ND3 numeric excerpt,
with SHA-256 `af5a4358bae2aa48cc3c7d16967816679ae15f9c863511c9c96daa729885e4bb`.
Its original acquisition has 512 measured samples. Historical wording within
the immutable archive describes its earlier use; it is not an active tutorial
or supported fitter input.

Run `python tools/prepare_uva_nd3.py` from the repository root to reproduce
the current [500-bin working example](../examples/uva-nd3.json). The derivation
interpolates phase and recorded baseline in frequency before subtraction;
its exact method and hashes are embedded in the working artifact.
