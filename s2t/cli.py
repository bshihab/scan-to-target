"""Command line: s2t <command> ...

  s2t ingest DICOM_DIR -o out.nii.gz [--series UID]   DICOM series -> NIfTI (prints geometry)
  s2t fmri --bold B --events E --out DIR              realign + GLM -> mean_epi, tmap_finger
  s2t register --fixed F --moving M -o M.npy [--metric mi|ssd]
  s2t subject SUB SES [--data data/ds000114]          whole pipeline for one ds000114 session -> out/scenes/SUB_SES
  s2t serve SCENE_DIR [--port 8770]                   planning GUI on http://127.0.0.1:PORT
  s2t robot PLAN.json ROBOT_DEMO.json                 plan -> robot frame, FRE, predicted TRE
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

# Data and outputs are found relative to the working directory, so an installed
# copy (for example in the Docker image) works from any checkout or mount.
ROOT = Path.cwd()


def cmd_ingest(a):
    from . import dicom
    vol = dicom.load_series(a.dicom_dir, a.series)
    vol.to_nifti(a.output)
    m = {k: v for k, v in vol.meta.items() if k != "affine_lps"}
    print(json.dumps({"shape": list(vol.data.shape), "spacing_mm": vol.spacing.round(4).tolist(), **m}, indent=2, default=str))
    np.set_printoptions(precision=4, suppress=True)
    print("voxel -> RAS mm:\n", vol.affine)


def cmd_fmri(a):
    from . import fmri
    print(json.dumps(fmri.run(a.bold, a.events, a.out), indent=2))


def cmd_register(a):
    from .register import register_rigid
    from .volume import Volume
    r = register_rigid(Volume.from_nifti(a.fixed), Volume.from_nifti(a.moving).mean(), metric=a.metric)
    np.save(a.output, r.matrix)
    np.set_printoptions(precision=5, suppress=True)
    print("moving world -> fixed world:\n", r.matrix)


def subject_paths(data: Path, sub: str, ses: str):
    base = data / sub / f"ses-{ses}"
    return (base / "anat" / f"{sub}_ses-{ses}_T1w.nii.gz",
            base / "func" / f"{sub}_ses-{ses}_task-fingerfootlips_bold.nii.gz",
            data / "task-fingerfootlips_events.tsv")


EPI_T1_METHOD = "bbr-two-starts-v4"  # cache key for epi_to_t1.npy


def run_subject(sub: str, ses: str, data: Path = ROOT / "data/ds000114", out_root: Path = ROOT / "out", log=print) -> Path:
    from . import fmri, scene
    from . import template
    from .register import epi_to_t1_qc, register_epi_to_t1_bbr
    from .volume import Volume, sha256
    t1_path, bold_path, events = subject_paths(Path(data), sub, ses)
    fdir = out_root / sub / f"ses-{ses}" / "fmri"
    summary_file = fdir / "fmri_summary.json"
    if not summary_file.exists() or not (fdir / "motion_matrices.npy").exists():
        fmri.run(bold_path, events, fdir, log=log)
    t1 = Volume.from_nifti(t1_path)
    mni_file = fdir.parent / "t1_to_mni.npy"
    mni_info_file = fdir.parent / "t1_to_mni.json"
    mni_info = json.loads(mni_info_file.read_text()) if mni_info_file.exists() else {}
    t1_sha = sha256(t1_path)
    # cached results are keyed on their inputs' content as well as the method,
    # so a changed T1 or mean EPI can never reuse an old matrix or gate verdict
    if not mni_file.exists() or mni_info.get("method") != template.METHOD or mni_info.get("t1_sha256") != t1_sha:
        log("aligning the MNI152 template to the T1 (affine, for the brain mask and MNI coordinates)")
        N, mni_info = template.align_template(t1)
        mni_info["method"] = template.METHOD
        mni_info["t1_sha256"] = t1_sha
        np.save(mni_file, N)
        mni_info_file.write_text(json.dumps(mni_info, indent=2))
    t1_to_mni = np.load(mni_file)
    if not mni_info.get("qc_ok"):
        raise RuntimeError(f"{sub} ses-{ses}: template fit failed its plausibility check "
                           f"(brain inside head {mni_info.get('qc_brain_inside_head', 0):.2f}, "
                           f"volume {mni_info.get('qc_brain_volume_ml', 0):.0f} mL); refusing to build a scene on it")
    reg_file = fdir / "epi_to_t1.npy"
    reg_key_file = fdir / "epi_to_t1.json"
    reg_key = {"t1_to_mni_sha256": hashlib.sha256(t1_to_mni.tobytes()).hexdigest(), "method": EPI_T1_METHOD,
               "mean_epi_sha256": sha256(fdir / "mean_epi.nii.gz")}
    reg_info = json.loads(reg_key_file.read_text()) if reg_key_file.exists() else {}
    if not reg_file.exists() or reg_info.get("key") != reg_key:
        log("registering mean EPI to T1 (boundary-based, from the scanner alignment and from mutual information)")
        brain = template.brain_mask_in_t1(t1, t1_to_mni)
        mean_epi = Volume.from_nifti(fdir / "mean_epi.nii.gz")
        r = register_epi_to_t1_bbr(t1, mean_epi, brain)
        np.save(reg_file, r.matrix)
        reg_info = {"key": reg_key, "start": r.info["start"], "costs": r.info["costs"],
                    **epi_to_t1_qc(r.matrix, mean_epi, r.cost)}
        reg_key_file.write_text(json.dumps(reg_info, indent=2))
    if not reg_info.get("qc_ok"):
        raise RuntimeError(f"{sub} ses-{ses}: EPI-to-T1 registration failed its plausibility check "
                           f"(rotation {reg_info.get('rotation_from_scanner_deg', 0):.1f} deg, moved "
                           f"{reg_info.get('moved_from_scanner_max_mm', 0):.1f} mm, cost {reg_info.get('bbr_cost', 0):.2f}); "
                           "refusing to build a scene on it")
    M = np.load(reg_file)
    summ = json.loads(summary_file.read_text())
    sdir = out_root / "scenes" / f"{sub}_ses-{ses}"
    scene.build(t1_path, fdir / "mean_epi.nii.gz", fdir / "tmap_finger.nii.gz", M, sdir, t1_to_mni=t1_to_mni,
                title=f"ds000114 {sub} ses-{ses} · finger tapping",
                overlay_label="finger > mean(foot, lips), t (AR(1) model, uncorrected)",
                extra_inputs={"bold_sha256": summ["bold_sha256"], "contrast": summ["contrast"],
                              "smooth_fwhm_mm": summ["smooth_fwhm_mm"], "dropped_volumes": summ.get("dropped_volumes")})
    log(f"scene written to {sdir}")
    return sdir


def cmd_subject(a):
    run_subject(a.sub, a.ses, Path(a.data))


def cmd_serve(a):
    from .serve import serve
    serve(a.scene_dir, host=a.host, port=a.port)


def cmd_robot(a):
    from . import robot
    plan = json.loads(Path(a.plan).read_text())
    demo = json.loads(Path(a.robot_demo).read_text())
    print(json.dumps(robot.to_robot(plan, demo["fiducials_image_ras"], demo["fiducials_robot_measured"], demo["sigma_mm"],
                                    np.array(demo["true_image_to_robot"]) if "true_image_to_robot" in demo else None), indent=2))


def main(argv=None):
    p = argparse.ArgumentParser(prog="s2t", description="Scan to Target: research demo, not a medical device.")
    sp = p.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("ingest"); s.add_argument("dicom_dir"); s.add_argument("-o", "--output", required=True); s.add_argument("--series"); s.set_defaults(f=cmd_ingest)
    s = sp.add_parser("fmri"); s.add_argument("--bold", required=True); s.add_argument("--events", required=True); s.add_argument("--out", required=True); s.set_defaults(f=cmd_fmri)
    s = sp.add_parser("register"); s.add_argument("--fixed", required=True); s.add_argument("--moving", required=True); s.add_argument("-o", "--output", required=True); s.add_argument("--metric", default="mi", choices=["mi", "ssd"]); s.set_defaults(f=cmd_register)
    s = sp.add_parser("subject"); s.add_argument("sub"); s.add_argument("ses"); s.add_argument("--data", default=str(ROOT / "data/ds000114")); s.set_defaults(f=cmd_subject)
    s = sp.add_parser("serve"); s.add_argument("scene_dir"); s.add_argument("--port", type=int, default=8770); s.add_argument("--host", default="127.0.0.1", help="0.0.0.0 only inside a container whose port is published to 127.0.0.1"); s.set_defaults(f=cmd_serve)
    s = sp.add_parser("robot"); s.add_argument("plan"); s.add_argument("robot_demo"); s.set_defaults(f=cmd_robot)
    a = p.parse_args(argv)
    a.f(a)


if __name__ == "__main__":
    sys.exit(main())
