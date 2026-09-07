# Intermediate source reconstruction

`sums-stale-registry.patch` applies to commit `d3081f9fe3597bcecad1c9b3109e3c038427dc1b` and reproduces the changed source files recorded in `full-native-sums-stale-registry.json`. Its three resulting files were hash-checked against that result. This preserves the intermediate implementation without replacing the final corrected runtime. The native binary remains identified separately by its build record.
