# Third-party materials

DiscoRL is an external dependency from Google DeepMind, pinned to commit
`9059a29f7121d60948f25ef165e08e050e9399c8`.
Its software is Apache-2.0; its README licenses other materials under CC BY 4.0.
Copyright 2025 Google LLC. This is not an official Google product.
The supplied `artifacts/teachers/disco_103.npz` is the released Disco103
checkpoint, redistributed unchanged with attribution:

Junhyuk Oh et al. (2025), *Discovering state-of-the-art reinforcement learning
algorithms*, Nature 648, 312–319.
https://doi.org/10.1038/s41586-025-09761-x

Source and license statement:
https://github.com/google-deepmind/disco_rl/tree/9059a29f7121d60948f25ef165e08e050e9399c8

CC BY 4.0 legal terms: https://creativecommons.org/licenses/by/4.0/legalcode
Apache-2.0 legal terms: https://www.apache.org/licenses/LICENSE-2.0

DiscoRL source is not vendored. The frozen `gremlin_rl` code contains our
adapters to the public implementation; retain existing attribution comments.
DiscoRL-derived portions of those adapters retain Apache-2.0, not MIT; a copy
is supplied in `LICENSE-APACHE-2.0`. The adapter docstrings describe our changes,
including immediate-autoreset loss handling and the preview differentiation
window. Our MIT notice does not replace upstream copyright or license terms.
Dependencies listed in the requirements keep their respective licenses.
No Atari ROMs are included; users are responsible for the terms of benchmark
assets they install. GREMLIN's generated games and original/trained teacher
weights are separate artifacts, not copies of the released Disco103 rule.
