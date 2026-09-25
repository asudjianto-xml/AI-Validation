"""Vendored prompt-factorization subset of Frontier-Discovery.

Chapter 9 optimizes a report-generation instruction by factoring it into declared
behavioral dimensions and designing over them. Those three modules are copied here
so `aiv` carries the method without depending on the Frontier-Discovery repository:

  factorization  -- the versioned prompt document, its validation contract and
                    `build_document_population`, which renders complete system
                    prompts from a design over the admitted factors.
  factor_doe     -- factor encoding, constraint compilation and design generation.
  spacefilling   -- phi_p space-filling optimization over the unit cube.

Two changes were made to the copies. `spacefilling` keeps the optimizer and drops
`SpaceFillingDesigner`, whose base classes live in `frontier_discovery.design` and
which nothing here calls. `factor_doe` imports the optimizer from this package
instead of the original one. Nothing else was edited, so the validation contract is
the upstream contract.

The default design path needs only numpy and scipy. Torch is imported lazily inside
the phi_p optimizer and is reached only by `method="sobol+refine"`; pyyaml is needed
only by the YAML entry points. Knowlytix is not a dependency of any of them.
"""
