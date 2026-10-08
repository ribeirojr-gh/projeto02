#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyMOL render driver for representative-structure snapshots (run inside PyMOL's
own interpreter: `pymol -cq scripts/pymol_render_structures.py`).

Reads the role-split PDB files written by script_14_prepare_structure_renders.py
(structures/render_inputs/) and ray-traces publication-style PNGs to
figures/renders/. Framework atoms are rendered muted grey/thin so the
chemisorbed intermediate (full CPK colour, thicker sticks) and, where present,
explicit pore water (blue-tinted, thicker sticks) stand out.
"""

from pathlib import Path
from pymol import cmd

BASE_DIR = Path("/home/luiz/SIMULACOES/her-oer-co2rr-mofs")
IN_DIR = BASE_DIR / "structures" / "render_inputs"
OUT_DIR = BASE_DIR / "figures" / "renders"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def reset_view_settings():
    cmd.set("ray_opaque_background", 1)
    cmd.set("antialias", 2)
    cmd.set("ray_shadows", 0)
    cmd.set("specular", 0.25)
    cmd.set("depth_cue", 0)
    cmd.bg_color("white")


def auto_bond_metals(obj, cutoff=2.5):
    """PyMOL's default distance-based auto-bonding on load uses a per-element
    covalent-radius table that is too short for Co/Cu coordination bonds to
    framework oxygens (~1.9-2.3 A here), so those bonds are silently not
    drawn and the rendered framework looks like disconnected fragments even
    though the underlying geometry is correct. Explicitly adds a bond for
    every metal-to-non-metal pair within `cutoff`."""
    model = cmd.get_model(obj)
    atoms = model.atom
    # PyMOL's parsed PDB element symbol is upper-cased ("CU", not "Cu").
    metal = [a for a in atoms if a.symbol.upper() in ("CO", "CU")]
    other = [a for a in atoms if a.symbol.upper() not in ("CO", "CU")]
    for m in metal:
        mx, my, mz = m.coord
        for o in other:
            ox, oy, oz = o.coord
            d = ((mx - ox) ** 2 + (my - oy) ** 2 + (mz - oz) ** 2) ** 0.5
            if d <= cutoff:
                cmd.bond(f"{obj} and index {m.index}", f"{obj} and index {o.index}")


def style_framework(obj):
    cmd.hide("everything", obj)
    cmd.show("sticks", obj)
    cmd.set("stick_radius", 0.11, obj)
    cmd.color("grey75", f"{obj} and elem C")
    cmd.color("grey60", f"{obj} and elem O")
    cmd.color("grey90", f"{obj} and elem H")
    cmd.show("spheres", f"{obj} and (elem Co+Cu)")
    cmd.set("sphere_scale", 0.42, f"{obj} and (elem Co+Cu)")
    cmd.color("slate", f"{obj} and elem Co")
    cmd.color("deepsalmon", f"{obj} and elem Cu")
    cmd.set("stick_transparency", 0.35, obj)
    cmd.set("sphere_transparency", 0.15, f"{obj} and (elem Co+Cu)")


def style_highlight(obj, o_color, h_color, c_color="yellow"):
    cmd.hide("everything", obj)
    cmd.show("sticks", obj)
    cmd.show("spheres", obj)
    cmd.set("stick_radius", 0.22, obj)
    cmd.set("sphere_scale", 0.26, obj)
    cmd.color(o_color, f"{obj} and elem O")
    cmd.color(h_color, f"{obj} and elem H")
    cmd.color(c_color, f"{obj} and elem C")


def orient_and_ray(objs, out_path, width=1400, height=1100, zoom_buffer=2.0):
    cmd.orient(" or ".join(objs))
    cmd.zoom(" or ".join(objs), buffer=zoom_buffer)
    cmd.ray(width, height)
    cmd.png(str(out_path), dpi=300)


def render_framework_overview(label):
    cmd.reinitialize()
    reset_view_settings()
    obj = "framework"
    cmd.load(str(IN_DIR / f"{label}_framework_overview.pdb"), obj)
    auto_bond_metals(obj)
    style_framework(obj)
    cmd.set("stick_transparency", 0.0, obj)
    cmd.set("sphere_transparency", 0.0, f"{obj} and (elem Co+Cu)")
    orient_and_ray([obj], OUT_DIR / f"{label}_framework_overview.png", zoom_buffer=1.0)


def render_active_site(label, has_water=False):
    cmd.reinitialize()
    reset_view_settings()
    fw = f"{label}_fw"
    ads = f"{label}_ads"
    cmd.load(str(IN_DIR / f"{label}_active_site_framework.pdb"), fw)
    cmd.load(str(IN_DIR / f"{label}_active_site_adsorbate.pdb"), ads)
    auto_bond_metals(fw)
    style_framework(fw)
    style_highlight(ads, o_color="red", h_color="white", c_color="forest")
    objs = [fw, ads]
    if has_water:
        wat = f"{label}_wat"
        cmd.load(str(IN_DIR / f"{label}_active_site_water.pdb"), wat)
        style_highlight(wat, o_color="marine", h_color="lightblue")
        objs.append(wat)
    orient_and_ray(objs, OUT_DIR / f"{label}_active_site.png", zoom_buffer=1.5)


print("PYMOL_SCRIPT_STARTED", __name__)
champions = ["oer_champion_CoMOF74", "her_champion_CuMOF74", "co2rr_champion_qmofdc7e5a3"]
for label in champions:
    render_framework_overview(label)
    render_active_site(label, has_water=False)

render_active_site("solvation_example_CoMOF74_OH_2H2O", has_water=True)

print("RENDER_DONE")
