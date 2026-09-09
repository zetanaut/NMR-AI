# Exercise settings

The tutorial commands load their default settings automatically. Inspect these
files alongside the relevant lesson to see which quantities are known, assumed,
or fitted. If you change settings for an investigation, keep your copy with
the resulting data and model.

| File | Controls |
| --- | --- |
| [deuteron-acquisition.json](deuteron-acquisition.json) | The exact 500-bin frequency grid for every exercise |
| [deuteron-baseline-setup.json](deuteron-baseline-setup.json) | The baseline's confirmed cable setup and remaining assumptions |
| [experimental-matching.json](experimental-matching.json) | The single-site raw-sweep fit |
| [butanol-matching.json](butanol-matching.json) | The two-site butanol fit |
| [uva-nd3-matching.json](uva-nd3-matching.json) | The ND3 raw fit and its assumed circuit constants |
| [matched-generator-coverage.json](matched-generator-coverage.json) | Controlled variations around the single-site fits |
| [lineshape-training.json](lineshape-training.json) | The reference experiment-based network run |
| [model-comparison.json](model-comparison.json) | The optional MLP/DNN/CNN benchmark |

The [baseline setup template](baseline-setup.template.json) is for another
apparatus; fill its required fields before using it. The
[butanol theory source record](butanol-theory-source.json) identifies the
supplied model conventions.

Search bounds and simulated parameter ranges are declared exercise choices.
They are not measured uncertainties or fitted population distributions.
