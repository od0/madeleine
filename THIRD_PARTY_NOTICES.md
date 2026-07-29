# Third-party notices

**NitroGen action labels (NVIDIA, 2026).** The mapped supervision in this
project derives from the action annotations of the
[NitroGen dataset](https://huggingface.co/datasets/nvidia/NitroGen)
([versioned card](https://huggingface.co/datasets/nvidia/NitroGen/tree/b171bc8ed2e3c311e9305ebb993c56ef565ab509),
accessed 2026-07-26), whose card identifies the action labels as
**CC BY-NC 4.0** ([legal code](https://creativecommons.org/licenses/by-nc/4.0/legalcode.en)),
for research and development use. The license terms apply to the annotations
and to adaptations of them; as a conservative policy, this project treats
its label-derived artifacts as non-commercial. The declaration covers the
dataset annotations; it is distinct from any license on NitroGen model
weights or code (neither is used or redistributed here) and conveys no
rights in the underlying gameplay videos, which remain with their owners
and platforms.

**Frame exhibits in figures.** Six figures include single annotated frames or
small crops from public gameplay videos: one frame per point or a short
evidence sequence, low resolution, credited in-figure, no recognizable people.
The project's position is that this limited use is fair use for research
commentary; that is a working posture, and final clearance is a
repository-owner decision, not a settled legal conclusion. Sources are
machine-recorded in
[results/figures/wild_exhibit_sources.json](results/figures/wild_exhibit_sources.json)
and
[results/figures/wild_decoder_sources.json](results/figures/wild_decoder_sources.json)
— `fig_wild_panels`: YouTube `ss3nhAUaScE`; `fig_wild_styles`: YouTube
`fJcUr6CXD1I`, `b43KAaem61g`, `elDsFg-S8YA` (keyboard region only), Twitch
`v1509603803`, `v378693976` (controller glyph only); the three
`fig_wild_decoder_*` exhibits: Twitch `v498642684`, cropped to gameplay and
keyboard evidence; `fig_offset_review_row`: YouTube `kdQbIoMxzZw` (one
offset-review contact-sheet row of masked gameplay crops); and
`fig_layout_review_geometry`: Twitch `v1068970940` (one annotated
layout-review frame with the proposed viewport, timer, and key-cell
rectangles drawn over the stream layout). The published
boundary-review packet under `results/wild20/ss3nhAUaScE/` additionally
retains its spot-check and wall-clock evidence pages — exact annotated
frames from YouTube `ss3nhAUaScE` — kept as the reviewed record of a human
admission decision and credited on the same posture. If you are a rights
holder and want an exhibit removed or credited differently, open an issue
and it will be replaced. The repository excludes downloaded videos and bulk
frame derivatives.

*Celeste* and its assets are owned by Extremely OK Games; this repository does
not include the game and is not affiliated with or endorsed by its creators,
[Everest](https://everestapi.github.io/), NVIDIA, or the authors of VPT.

The original MADELEINE code is released under the MIT License (see
[LICENSE](LICENSE)). The data-rights statements above are unaffected: the
license covers the code, not third-party labels, videos, or game assets.
