
import bpy
import math
import mathutils
import xml.etree.ElementTree as ET
import numpy as np

# ----------------- USER PATHS -----------------
MJCF_XML = r"/Users/leyangwen/Developer/vicon-read/conversion_scripts/IsaacGym/imitation_motion_process/phys_humanoid_v3_box_foot.xml"      # <--- set this to your MJCF
MOTION_NPY = r"/Users/leyangwen/Documents/Isaac/MMH/box01.51470934.20250919201305+__+clip_01/phys_humanoid_v3/ref_motion.npy"         # <--- your uploaded motion
SCENE_FPS_FROM_MOTION = True
# Limit frames for quick test (set to None to use full length)
MAX_FRAMES = None  # e.g., 200
# Quaternion order of motion['rotation']['arr']:
# - "xyzw" (default) or "wxyz"
QUAT_ORDER = "xyzw"
# ----------------------------------------------


EPS = 1e-9

def vec3(s):
    return [float(x) for x in s.split()] if s else [0.0, 0.0, 0.0]

def axis_angle_from_to(src, dst):
    a = mathutils.Vector(src).normalized()
    b = mathutils.Vector(dst).normalized()
    if a.length < EPS or b.length < EPS:
        return mathutils.Quaternion((1.0, 0.0, 0.0, 0.0))
    dot = max(min(a.dot(b), 1.0), -1.0)
    if abs(dot - 1.0) < 1e-8:
        return mathutils.Quaternion((1.0, 0.0, 0.0, 0.0))
    if abs(dot + 1.0) < 1e-8:
        return mathutils.Quaternion((0.0, 1.0, 0.0, 0.0))
    axis = a.cross(b)
    angle = math.acos(dot)
    return mathutils.Quaternion(axis, angle)

def quaternion_from_xyzw_or_wxyz(arr4, order="xyzw"):
    if order == "xyzw":
        x, y, z, w = arr4
        return mathutils.Quaternion((w, x, y, z))
    elif order == "wxyz":
        w, x, y, z = arr4
        return mathutils.Quaternion((w, x, y, z))
    else:
        raise ValueError("QUAT_ORDER must be 'xyzw' or 'wxyz'")

class MJCFBuilder:
    def __init__(self, xml_path):
        self.tree = ET.parse(xml_path)
        self.root = self.tree.getroot()
        self.body_to_empty = {}
        self.world_col = bpy.data.collections.new("MJCF_World")
        bpy.context.scene.collection.children.link(self.world_col)

    def build(self):
        worldbody = self.root.find("worldbody")
        if worldbody is None:
            raise RuntimeError("No <worldbody> in MJCF")
        for body in worldbody.findall("body"):
            self._build_body_recursive(body, parent_empty=None)

    def _build_body_recursive(self, body, parent_empty):
        name = body.get("name", "unnamed")
        pos  = vec3(body.get("pos"))
        bpy.ops.object.empty_add(type='PLAIN_AXES', location=pos)
        empty = bpy.context.active_object
        empty.name = f"Body::{name}"
        empty.rotation_mode = 'QUATERNION'
        if parent_empty:
            empty.parent = parent_empty
        self.body_to_empty[name] = empty
        for g in body.findall("geom"):
            self._add_geom(empty, g)
        for child in body.findall("body"):
            self._build_body_recursive(child, empty)

    def _add_geom(self, parent_empty, g):
        gtype = g.get("type", "capsule")
        fromto = g.get("fromto")
        if fromto:
            a = [float(x) for x in fromto.split()]
            p1 = mathutils.Vector(a[0:3]); p2 = mathutils.Vector(a[3:6])
            center = 0.5*(p1 + p2)
            axis = (p2 - p1); L = axis.length
            if L < EPS: return
            axis_n = axis / (L + EPS)
            radius = float(g.get("size", "0.05").split()[0])
            bpy.ops.mesh.primitive_cylinder_add(radius=radius, depth=L, location=center)
            obj = bpy.context.active_object
            obj.name = f"Geom::{g.get('name','capsule')}"
            q = axis_angle_from_to((0,0,1), axis_n)
            obj.rotation_mode = 'QUATERNION'
            obj.rotation_quaternion = q
            obj.parent = parent_empty
            return
        pos = vec3(g.get("pos"))
        quat = g.get("quat")
        if quat:
            qvals = [float(x) for x in quat.split()]  # assume (w x y z) in MJCF
            q_bl = mathutils.Quaternion((qvals[0], qvals[1], qvals[2], qvals[3]))
        else:
            q_bl = mathutils.Quaternion((1.0, 0.0, 0.0, 0.0))
        if gtype == "sphere":
            r = float(g.get("size").split()[0])
            bpy.ops.mesh.primitive_uv_sphere_add(radius=r, location=pos)
            obj = bpy.context.active_object
        elif gtype == "box":
            sx, sy, sz = [float(x) for x in g.get("size").split()]
            bpy.ops.mesh.primitive_cube_add(location=pos)
            obj = bpy.context.active_object
            obj.scale = (sx, sy, sz)
        else:
            r = float(g.get("size", "0.05").split()[0])
            bpy.ops.mesh.primitive_cylinder_add(radius=r, depth=2*r, location=pos)
            obj = bpy.context.active_object
        obj.name = f"Geom::{g.get('name', gtype)}"
        obj.rotation_mode = 'QUATERNION'
        obj.rotation_quaternion = q_bl
        obj.parent = parent_empty

class MotionApplier:
    def __init__(self, npy_path, builder: MJCFBuilder):
        motion_0d = np.load(npy_path, allow_pickle=True)
        self.motion = motion_0d.item() if hasattr(motion_0d, "item") else motion_0d
        self.builder = builder
        self.fps = int(self.motion.get("fps", 20))
        self.is_local = bool(self.motion.get("is_local", True))
        skel = self.motion.get("skeleton_tree")
        # pull arrays from 'arr' if wrapped
        def arr_of(v):
            return v.get("arr") if isinstance(v, dict) and "arr" in v else v
        self.node_names = list(skel.get("node_names"))
        self.parent_indices = arr_of(skel.get("parent_indices"))
        self.local_translation = arr_of(skel.get("local_translation"))
        self.rot = arr_of(self.motion.get("rotation"))
        self.root_t = arr_of(self.motion.get("root_translation"))
        if self.rot is None or self.root_t is None:
            raise RuntimeError("rotation['arr'] or root_translation['arr'] missing")
        self.frames = self.rot.shape[0]
        self.num_joints = self.rot.shape[1]
        # mapping from joint name to MJCF body empty
        self.joint_to_obj = self._build_name_mapping()
        # report mapping
        mapped = sorted([jn for jn in self.node_names if jn in self.joint_to_obj])
        missing = sorted([jn for jn in self.node_names if jn not in self.joint_to_obj])
        print(f"[Mapping] mapped {len(mapped)}/{len(self.node_names)} joints.")
        if missing:
            print("[Mapping] missing:", missing)

    def _build_name_mapping(self):
        mapping = {}
        body_names = self.builder.body_to_empty.keys()
        canon_body = {self._canon(b): b for b in body_names}
        for jn in self.node_names:
            cj = self._canon(jn)
            if cj in canon_body:
                mapping[jn] = self.builder.body_to_empty[canon_body[cj]]
                continue
            # heuristic variants
            for v in (
                cj.replace("lefthand","left_hand").replace("righthand","right_hand")
                  .replace("leftupperarm","left_upper_arm").replace("leftlowerarm","left_lower_arm")
                  .replace("rightupperarm","right_upper_arm").replace("rightlowerarm","right_lower_arm")
                  .replace("leftthigh","left_thigh").replace("rightthigh","right_thigh")
                  .replace("leftshin","left_shin").replace("rightshin","right_shin")
                  .replace("leftfoot","left_foot").replace("rightfoot","right_foot")
            ,):
                if v in canon_body:
                    mapping[jn] = self.builder.body_to_empty[canon_body[v]]
                    break
        return mapping

    @staticmethod
    def _canon(s):
        return s.lower().replace(" ", "").replace("::","")

    def apply(self, max_frames=None):
        if SCENE_FPS_FROM_MOTION:
            bpy.context.scene.render.fps = self.fps
        F = self.frames if max_frames is None else min(max_frames, self.frames)
        bpy.context.scene.frame_start = 1
        bpy.context.scene.frame_end = F
        pelvis_obj = self.builder.body_to_empty.get("pelvis") \
                     or self.builder.body_to_empty.get("Pelvis") \
                     or list(self.builder.body_to_empty.values())[0]
        for ob in self.builder.body_to_empty.values():
            ob.rotation_mode = 'QUATERNION'
        for f in range(F):
            bpy.context.scene.frame_set(f+1)
            # root translation
            tx, ty, tz = self.root_t[f].tolist()
            pelvis_obj.location = (tx, ty, tz)
            pelvis_obj.keyframe_insert(data_path="location")
            # per-joint rotations
            for ji, jn in enumerate(self.node_names):
                ob = self.joint_to_obj.get(jn)
                if ob is None:
                    continue
                qarr = self.rot[f, ji, :].tolist()
                q = quaternion_from_xyzw_or_wxyz(qarr, order=QUAT_ORDER)
                ob.rotation_quaternion = q
                ob.keyframe_insert(data_path="rotation_quaternion")
        print("Done keyframing.")

def main():
    builder = MJCFBuilder(MJCF_XML)
    builder.build()
    applier = MotionApplier(MOTION_NPY, builder)
    applier.apply(max_frames=MAX_FRAMES)

if __name__ == "__main__":
    main()
