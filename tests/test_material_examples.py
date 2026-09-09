"""Independent material normalization, circuit limits and data-contract checks."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from scipy.integrate import quad_vec

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
from baseline_data import acquisition_grid,load_acquisition
from circuit import Circuit
from lineshape import pake_susceptibility
from material_lineshapes import butanol_components
from match_butanol import node_basis as butanol_basis, susceptibility
from match_butanol import reconstruct as butanol_reconstruct
from match_experimental_signals import node_basis as single_basis, digest
from experimental_data import load_signal_csv
from match_uva_nd3 import basis, match_scan, normalized_shape, reconstruct
from uva_nd3_data import (SOURCE, SOURCE_SHA256, confirmed_frequency, linear_resample,
                          load_nd3, load_nd3_source)


class MaterialExamples(unittest.TestCase):
    def test_butanol_agrees_with_independent_complex_orientation_integral(self):
        f=np.array([-.23,-.091,-.017,.03,.082,.21])
        polarization=.42
        q=2-np.sqrt(4-3*polarization**2)
        weights=[(polarization+q)/2,(polarization-q)/2]
        fractions=[.83,.17];splits=[.062,.105];etas=[.07,.19];width=.005
        total=np.zeros_like(f,dtype=complex)
        for fraction,split,eta in zip(fractions,splits,etas):
            for eps,w in zip([1,-1],weights):
                def at_phi(phi):
                    c=np.cos(2*phi)
                    def at_orientation(t):
                        resonance=eps*split*(1-eta*c-(3-eta*c)*t*t)
                        return 1/(np.pi*(resonance-f+1j*width))
                    return quad_vec(at_orientation,0,1,epsabs=2e-9,epsrel=2e-9)[0]*2/np.pi
                total+=fraction*w*quad_vec(at_phi,0,np.pi/2,epsabs=2e-9,epsrel=2e-9)[0]
        actual=butanol_components(f,polarization,0,*splits,width,*etas,.17,nphi=64)[0]
        np.testing.assert_allclose(actual,total,rtol=2e-8,atol=2e-9)

    def test_site_fraction_weights_frequency_integrals_and_zero_p_vanishes(self):
        def integrand(f):
            return -np.imag(butanol_components(np.array([f]),.4,0,.062,.102,.003,0,0,.2))[:,0]
        area=sum(quad_vec(integrand,lo,hi,epsabs=1e-8)[0] for lo,hi in
                 [(-np.inf,-.3),(-.3,0),(0,.3),(.3,np.inf)])
        np.testing.assert_allclose(area,[.4,.32,.08],rtol=1e-7,atol=1e-8)
        zero=butanol_components(np.linspace(32.3,33.1,500),0,32.7,.062,.102,.003,.06,.2,.2)
        np.testing.assert_array_equal(zero,np.zeros((3,500)))

    def test_zero_od_fraction_reproduces_original_full_circuit(self):
        report=json.loads((ROOT/"docs/assets/experimental-matching.json").read_text())
        f=acquisition_grid(load_acquisition())/1e6
        for fit in report["fits"]:
            p={**fit["parameters"],"od_split_ratio":1.7,"eta_od":.17,"od_fraction":0.}
            c=Circuit(**fit["fixed_circuit"])
            np.testing.assert_allclose(butanol_basis(f,c,p,nphi=32),single_basis(f,c,fit["parameters"]),atol=1e-16,rtol=1e-13)
            chi,cd,od=susceptibility(f,p)
            np.testing.assert_array_equal(chi,cd)
            np.testing.assert_array_equal(od,np.zeros_like(chi))

    def test_equal_sites_and_window_restriction_do_not_renormalize(self):
        f=np.linspace(-.5,.5,500)
        for p in [-.4,.4]:
            expected=pake_susceptibility(f/.062,p,.08,.003/.062)[0]/.062
            for fraction in [0,.25,1]:
                result=butanol_components(f,p,0,.062,.062,.003,.08,.08,fraction)[0]
                np.testing.assert_allclose(result,expected,rtol=1e-13,atol=1e-13)
                cropped=butanol_components(f[100:200],p,0,.062,.062,.003,.08,.08,fraction)[0]
                np.testing.assert_array_equal(cropped,result[100:200])
        for fraction in [-.01,1.01,float('nan')]:
            with self.assertRaises(ValueError):butanol_components(f,.4,0,.062,.1,.003,0,0,fraction)

    def test_uva_nd3_uses_confirmed_grid_and_preserves_source_provenance(self):
        data,audit=load_nd3(ROOT/"examples/uva-nd3.json")
        source=load_nd3_source()
        self.assertEqual(digest(SOURCE),SOURCE_SHA256)
        self.assertEqual(audit["bins"],500)
        self.assertEqual(audit["source_record_numbers_1based"],[1,126,251,376,501])
        self.assertEqual(data["source_record_count"],501)
        for r,original in zip(data["records"],source["records"]):
            f=np.asarray(r["frequency_mhz"])
            np.testing.assert_array_equal(f,confirmed_frequency())
            for channel in ("phase","baseline"):
                np.testing.assert_array_equal(r[channel],np.interp(f,original["frequency_mhz"],original[channel]))
            # Changing subtraction order is equal up to roundoff on the much
            # larger phase/reference values, not only on their small difference.
            roundoff=4*np.finfo(float).eps*max(np.max(np.abs(original["phase"])),np.max(np.abs(original["baseline"])))
            np.testing.assert_allclose(r["basesub"],np.interp(f,original["frequency_mhz"],original["basesub"]),atol=roundoff,rtol=0)
            np.testing.assert_array_equal(np.asarray(r["phase"])-r["baseline"],r["basesub"])
            self.assertNotIn("pol",r);self.assertNotIn("cc",r)
        with self.assertRaisesRegex(ValueError,"500"):load_nd3(SOURCE)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"bad.json"
            bad=deepcopy(data);bad["records"][0]["basesub"][0]+=.001
            path.write_text(json.dumps(bad))
            with self.assertRaisesRegex(ValueError,"Subtraction"):load_nd3(path)
            bad=deepcopy(data);bad["records"][0]["frequency_mhz"]=np.linspace(32.3,33.1,500).tolist()
            path.write_text(json.dumps(bad))
            with self.assertRaisesRegex(ValueError,"exact confirmed"):load_nd3(path)
            bad=deepcopy(data);bad["records"][0]["phase"][1]+=.001;bad["records"][0]["basesub"][1]=bad["records"][0]["phase"][1]-bad["records"][0]["baseline"][1]
            path.write_text(json.dumps(bad))
            with self.assertRaisesRegex(ValueError,"source resampling"):load_nd3(path)

    def test_nd3_resampling_operator_and_no_extrapolation(self):
        source=np.array([0.,.7,1.8,3.])
        target=np.array([0.,.3,1.,2.,3.])
        values=2*source+3
        np.testing.assert_allclose(linear_resample(source,values,target),2*target+3,rtol=0,atol=1e-15)
        # Independent explicit piecewise-linear observation matrix.
        clipped=np.clip(target,source[0],source[-1])
        right=np.searchsorted(source,clipped,side="right").clip(1,len(source)-1)
        left=right-1;weight=(clipped-source[left])/(source[right]-source[left])
        operator=np.zeros((len(target),len(source)))
        operator[np.arange(len(target)),left]=1-weight
        operator[np.arange(len(target)),right]=weight
        matrix=np.column_stack([np.sin(source),values,source**2])
        np.testing.assert_allclose(linear_resample(source,matrix,target),operator@matrix,atol=1e-15)
        for bad in (np.array([-.001,1.]),np.array([1.,3.001])):
            with self.assertRaisesRegex(ValueError,"extrapolate"):linear_resample(source,values,bad)
        with self.assertRaisesRegex(ValueError,"increasing"):linear_resample(source[::-1],values,target)

    def test_nd3_conditional_shape_and_known_synthetic_recovery(self):
        config=json.loads((ROOT/"configs/uva-nd3-matching.json").read_text())
        data,_=load_nd3(ROOT/"examples/uva-nd3.json")
        f=np.asarray(data["records"][0]["frequency_mhz"])
        p={n:s["initial"] for n,s in config["fit_parameters"].items()}
        p.update(P_model=-.37,eta=.065,g=.045)
        shape=normalized_shape(f,p)
        expected=pake_susceptibility((f-p["center_mhz"])/p["split_mhz"],p["P_model"],p["eta"],p["g"])[0]/(p["P_model"]*p["split_mhz"])
        np.testing.assert_allclose(shape,expected,rtol=1e-13)
        coeff=np.array([.00015,.000005,-.02,.0002,-.0001,.00005])
        y=basis(f,p)@coeff
        trace,fit=match_scan(f,y,config,starts=2)
        self.assertAlmostEqual(fit["parameters"]["P_model"],p["P_model"],delta=1e-4)
        np.testing.assert_allclose(reconstruct(f,fit),y,atol=2e-9)
        np.testing.assert_allclose(trace[:,2],y,atol=2e-9)
        self.assertEqual(fit["n_free"],11)
        source=np.asarray(load_nd3_source()["records"][0]["frequency_mhz"])
        # Generate on measured source coordinates independently, then fit the
        # resampled observation with exactly the same observation operator.
        sampled=np.interp(f,source,basis(source,p)@coeff)
        _,resampled_fit=match_scan(f,sampled,config,starts=2,source_record_1based=1)
        self.assertAlmostEqual(resampled_fit["parameters"]["P_model"],p["P_model"],delta=1e-4)
        np.testing.assert_allclose(reconstruct(f,resampled_fit),sampled,atol=2e-9)
        self.assertNotIn("wing_second_difference_sigma_proxy",resampled_fit["diagnostics"])
        for wrong in (source,np.linspace(32.3,33.1,500)):
            with self.assertRaisesRegex(ValueError,"500"):match_scan(wrong,np.zeros_like(wrong),config)
        p["P_model"]=0.
        self.assertTrue(np.isfinite(normalized_shape(f,p)).all())

    def check_publication(self, report, figure, prefix, ids, bins):
        import xml.etree.ElementTree as ET
        for name,expected in report["code_sha256"].items():
            self.assertEqual(digest(ROOT/"tools"/name),expected)
        publication=report["publication"]
        self.assertEqual(digest(ROOT/"tools/export_material_examples.py"),publication["exporter_sha256"])
        for name,expected in publication["figure_sha256"].items():
            self.assertEqual(digest(ROOT/"docs/assets"/name),expected)
        tree=ET.fromstring((ROOT/"docs/assets"/figure).read_text())
        ns={"s":"http://www.w3.org/2000/svg"}
        for identifier in ids:
            points=tree.find(f".//s:g[@id='{prefix}{identifier}']",ns)
            self.assertIsNotNone(points)
            self.assertEqual(len(points.findall('.//s:use',ns)),bins)

    def test_published_butanol_comparison_reconstructs_all_bins(self):
        report=json.loads((ROOT/"docs/assets/butanol-matching.json").read_text())
        reference=ROOT/"docs/assets"/report["publication"]["single_site_reference_asset"]
        original=json.loads(reference.read_text())
        records,audit=load_signal_csv(ROOT/"examples/Sample_RawSignal.csv")
        f=acquisition_grid(load_acquisition())/1e6
        self.assertEqual(report["single_site_report_sha256"],digest(reference))
        current=json.loads((ROOT/"docs/assets/experimental-matching.json").read_text())
        self.assertEqual(original["fits"],current["fits"])
        self.assertEqual(report["config_sha256"],digest(ROOT/"configs/butanol-matching.json"))
        self.assertEqual(report["publication"]["theory_source_sha256"],digest(ROOT/"configs/butanol-theory-source.json"))
        self.assertEqual(report["signal_parsing"],audit)
        self.assertEqual([fit["record_1based"] for fit in report["fits"]],list(range(1,6)))
        for fit,old,row in zip(report["fits"],original["fits"],records):
            prediction=butanol_reconstruct(f,fit)
            rms=np.sqrt(np.mean((row[1:]-prediction)**2))
            np.testing.assert_allclose(rms,fit["diagnostics"]["rms_recorded_units"],rtol=1e-7)
            np.testing.assert_allclose(100*(1-rms/old["diagnostics"]["rms_recorded_units"]),fit["rms_reduction_percent"],atol=1e-6)
            self.assertGreater(fit["rms_reduction_percent"],0)
            self.assertEqual(fit["n_free"],15)
            self.assertEqual(len(fit["candidates"]),6)
            self.assertTrue(fit["all_500_bins_fitted"])
            # Independent numerical refinement must be small against the residual.
            refined=butanol_reconstruct(f,fit,nphi=256)
            self.assertLess(np.max(abs(refined-prediction)),.02*rms)
        self.check_publication(report,"butanol-comparison.svg","butanol-scan-",range(1,6),500)

    def test_published_nd3_reconstructs_all_500_resampled_bins(self):
        report=json.loads((ROOT/"docs/assets/uva-nd3-matching.json").read_text())
        data,audit=load_nd3(ROOT/"examples/uva-nd3.json")
        self.assertEqual(report["audit"],audit)
        self.assertEqual(report["config_sha256"],digest(ROOT/"configs/uva-nd3-matching.json"))
        self.assertEqual(len(report["fits"]),5)
        for fit,row in zip(report["fits"],data["records"]):
            self.assertEqual(fit["source_record_1based"],row["source_record_1based"])
            f=np.asarray(row["frequency_mhz"])
            prediction=reconstruct(f,fit)
            rms=np.sqrt(np.mean((np.asarray(row["basesub"])-prediction)**2))
            np.testing.assert_allclose(rms,fit["diagnostics"]["rms_recorded_units"],rtol=1e-8)
            self.assertEqual(fit["n_free"],11)
            self.assertTrue(fit["all_500_bins_fitted"])
            self.assertTrue(fit["success"])
            self.assertLess(np.max(abs(reconstruct(f,fit,nphi=64)-prediction)),.001*rms)
        self.check_publication(report,"uva-nd3-matches.svg","nd3-record-",[1,126,251,376,501],500)


if __name__=="__main__":unittest.main()
