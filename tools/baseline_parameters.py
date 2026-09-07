"""Units and physical meanings of every active baseline-fit parameter."""
from html import escape
import numpy as np
from circuit import Circuit, cable_parameters

# name -> symbol, display-unit scale, unit, physical meaning
CATALOG = {
    "tune_capacitance_f": ("C_tune", 1e12, "pF", "Series tuning capacitance; controls cancellation of the transformed coil reactance. Use a calibrated knob setting or capacitor measurement."),
    "cable_length_m": ("ℓ", 1, "m", "Cable length between the modeled reference planes; controls electrical phase and impedance transformation. Constrain from the known half-wave branch and trim."),
    "cable_delta_length_m": ("δℓ", 1e3, "mm", "Small length correction around the independently known half-wave multiple, including supported trim and reference-plane corrections."),
    "stray_capacitance_f": ("C_stray", 1e12, "pF", "Parasitic capacitance shunting the coil; changes the terminal impedance. Not an extra series tuning capacitor."),
    "readout_gain": ("A", 1, "recorded units/V", "Effective detector/DAQ amplitude scale multiplying node voltage. Baseline alone cannot separate an unknown RF drive from an unknown gain."),
    "readout_phase_rad": ("φ₀", 180/np.pi, "degrees", "Detector reference phase mixing real and imaginary node voltage; stored in radians. This is not cable electrical length or EFG orientation."),
    "readout_offset": ("d", 1, "recorded units", "Additive detector/DAQ offset. No subtraction of the sweep minimum is performed."),
    "reference_hz": ("f_ref", 1e-6, "MHz", "Reference frequency for the circuit/phase expansion. Confirm it from the acquisition/tuning record; the plotted voltage maximum is not automatically resonance."),
    "drive_v": ("U", 1, "V", "RF source amplitude used with R₀ to calculate the drive current scale."),
    "drive_resistance_ohm": ("R₀", 1, "Ω", "Current-limiting source resistor; it also contributes finite-source loading at the detection node."),
    "input_resistance_ohm": ("R_input", 1, "Ω", "Resistive amplifier input load in parallel with the resonator leg."),
    "damping_resistance_ohm": ("R_D", 1, "Ω", "Series damping resistance in the resonator leg; limits resonance sharpness."),
    "coil_resistance_ohm": ("R_coil", 1, "Ω", "Effective RF series loss of the sensing coil, before nuclear absorption is included."),
    "coil_inductance_h": ("L₀", 1e9, "nH", "Unloaded coil inductance; together with the cable and tuning capacitor sets the resonant condition."),
    "cable_resistance_ohm_per_m": ("R_c", 1, "Ω/m", "Distributed conductor resistance, contributing cable attenuation and phase behavior."),
    "cable_inductance_h_per_m": ("L_c", 1e9, "nH/m", "Distributed cable inductance; with C_c determines characteristic impedance and propagation speed."),
    "cable_conductance_s_per_m": ("G_c", 1, "S/m", "Distributed dielectric leakage/loss conductance; zero is an explicit negligible-leakage approximation."),
    "cable_capacitance_f_per_m": ("C_c", 1e12, "pF/m", "Distributed cable capacitance; with L_c determines characteristic impedance and propagation speed."),
    "detector_phase_slope_rad_per_hz": ("φ₁", 1e6, "rad/MHz", "Linear detector-phase variation about f_ref, not a polynomial voltage baseline."),
    "detector_phase_curvature_rad_per_hz2": ("φ₂", 1e12, "rad/MHz²", "Quadratic detector-phase variation about f_ref; hold fixed unless supported by measurement."),
}


def parameter_reference_html():
    """Parameter meanings without displaying invalid or unmeasured fit values."""
    rows = []
    for name, (symbol, scale, unit, meaning) in CATALOG.items():
        rows.append(f'<tr><td><strong>{escape(symbol)}</strong><br><code>{escape(name)}</code></td>'
                    f'<td>{escape(unit)}</td><td>{escape(meaning)}</td></tr>')
    return ('<div id="fit-parameters"><h3>Every active baseline parameter</h3>'
            '<p>The deuteron reference is approximately 32.68 MHz. The table lists physical meanings and display units, '
            'not new component measurements. The setup file records each parameter as fixed or fitted, with its source, '
            'supported bounds, and any measurement constraint. Internal units follow the code names; readout phase is stored in radians.</p>'
            '<div class="table-wrap"><table><caption>Physical parameter reference; no measured-fit values are claimed</caption>'
            '<thead><tr><th>Parameter</th><th>Display units</th><th>Physical meaning</th></tr></thead><tbody>'
            + ''.join(rows) + '</tbody></table></div>'
            '<p>Use either physical cable length ℓ or the known integer half-wave branch plus δℓ, not two independent lengths. '
            'Filling factor and susceptibility scale multiply χ = 0 and cannot be determined from a baseline. '
            'Pake splitting, broadening, asymmetry, and polarization belong to the nuclear signal model.</p></div>')


def parameter_table_html(metadata):
    fit = metadata["selected_fit"]
    values = dict(fit["circuit"])
    a, b, offset = fit["quadrature_coefficients"]
    values.update(readout_gain=float(np.hypot(a, b)), readout_phase_rad=float(np.arctan2(-b, a)), readout_offset=offset)
    fitted = {"tune_capacitance_f", "cable_length_m", "stray_capacitance_f", "readout_gain", "readout_phase_rad", "readout_offset"}
    if "resolved_parameters" in fit:
        values.update(fit["resolved_parameters"])
        fitted = set(fit["free_parameters"])
    specs = fit.get("setup", {}).get("parameters", {})
    derived = {"cable_length_m"} if "cable_tuning" in fit.get("setup", {}) else set()
    groups = []
    for title, names in (("Fitted unknowns", [k for k in CATALOG if k in fitted and k in values]),
                         ("Fixed inputs and assumptions", [k for k in CATALOG if k not in fitted|derived and k in values]),
                         ("Derived from the tuning record", [k for k in CATALOG if k in derived])):
        if not names:
            continue
        rows = []
        for name in names:
            symbol, scale, unit, meaning = CATALOG[name]
            context = ''
            if name in specs:
                spec = specs[name]
                context = '<br><small>Source/assumption: '+escape(spec['source'])+'</small>'
                if 'bounds' in spec:
                    context += f'<br><small>Bounds: {spec["bounds"][0]*scale:.6g} to {spec["bounds"][1]*scale:.6g} {escape(unit)}</small>'
                if 'prior' in spec:
                    context += f'<br><small>Constraint: {spec["prior"]["mean"]*scale:.6g} ± {spec["prior"]["sigma"]*scale:.6g} {escape(unit)} (1σ)</small>'
            rows.append(f'<tr><td><strong>{escape(symbol)}</strong><br><code>{escape(name)}</code></td>'
                        f'<td>{values[name]*scale:.6g} {escape(unit)}</td><td>{escape(meaning)}{context}</td></tr>')
        groups.append(f'<h4>{title}</h4><div class="table-wrap"><table><caption>{title} for the displayed measured fit</caption>'
                      '<thead><tr><th>Parameter</th><th>Value</th><th>Physical meaning</th></tr></thead><tbody>'
                      +''.join(rows)+'</tbody></table></div>')
    c = Circuit(**fit["circuit"])
    tuning_frequency = fit.get("setup", {}).get("cable_tuning", {}).get("reference_hz", c.reference_hz)
    beta = float(cable_parameters(tuning_frequency, c)[1].imag)
    if 'resolved_parameters' in fit:
        provenance = (f'This tuning-informed fit optimizes only {len(fitted)} declared unknowns. '
                      'Fixed entries, bounds, and independent measurement constraints come from the supplied setup file; '
                      'their source or assumption is recorded with each parameter.')
        length_note = 'The setup specifies whether length is directly known/fitted or derived from a known half-wave branch plus a declared correction.'
        if "cable_tuning" in fit["setup"]:
            tuning = fit["setup"]["cable_tuning"]
            length_note += (f' The fixed branch is n = {tuning["half_wave_multiple"]} at '
                            f'{tuning["reference_hz"]/1e6:g} MHz. Source: {escape(tuning["source"])}.')
    else:
        provenance = ('For this exploratory example, six quantities were fitted: three circuit coordinates and three readout coordinates. '
                      'The fixed values are nominal assumptions; confirmed tuning records have not yet been supplied for these scans.')
        length_note = ('An integer branch was not enforced. Do not round this value and call the result a known tuning setting. '
                       'Use the recorded branch and its supported trim tolerance for a tuning-informed refit.')
    return ('<div id="fit-parameters"><h3>Every parameter behind the displayed fit</h3>'
            '<p>Distinguish fitted estimates from fixed inputs and their sources; a numerical fit value is not itself an independent component measurement or uncertainty. '
            +provenance+'</p>'
            +''.join(groups)
            +f'<p>At {tuning_frequency/1e6:g} MHz, this cable model has λ/2 = {np.pi/beta:.6g} m. '
            f'The resulting length corresponds to <strong>{c.cable_length_m*beta/np.pi:.4f} half-waves</strong>. '
            +length_note+'</p>'
            f'<p>The saved linear coefficients are a = {a:.6g}, b = {b:.6g}, d = {offset:.6g}, with '
            'a = A cos φ₀ and b = −A sin φ₀. They are an alternative representation of the same three readout parameters, '
            'not three extra fitted degrees of freedom. The table shows the actual readout mapping, '
            'not the unused detector defaults inside the stored circuit object. With frequency-dependent phase, '
            'φ₁ and φ₂ must also be retained when reconstructing the trace.</p>'
            '<p><strong>Not identifiable from a baseline:</strong> filling factor η_fill and susceptibility amplitude scale '
            'multiply χ, which is zero here. Neither is fitted; neither can be obtained from this baseline. '
            'Pake splitting, broadening, EFG asymmetry, and polarization are signal-model quantities, not baseline-fit parameters.</p></div>')
