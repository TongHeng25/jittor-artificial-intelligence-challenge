from typing import List

from collections import defaultdict

from .asset import Asset

def collapse(asset: Asset, keep: List[str]) -> Asset:
    assert asset.names is not None
    assert asset.check_order()
    new_parents = []
    new_skin = asset.skin.copy()
    deg = defaultdict(int)
    for i in range(asset.J):
        p = asset.parents[i]
        if p is not None:
            deg[p] += 1
    removed = {}
    for i in reversed(range(asset.J)):
        name = asset.names[i]
        if name not in keep:
            assert deg[i] == 0
            new_skin[:, asset.parents[i]] += new_skin[:, i]
            deg[asset.parents[i]] -= 1
            removed[i] = True
    new_ids = []
    new_names = []
    map_to = {}
    tot = 0
    for i in range(asset.J):
        if removed.get(i) is True:
            continue
        map_to[i] = tot
        tot += 1
        new_ids.append(i)
        new_names.append(asset.names[i])
        new_parents.append(None if asset.parents[i] is None else map_to[asset.parents[i]])
    new_joints = asset.joints[new_ids]
    new_skin = new_skin[:, new_ids]
    new_matrix_local = asset.matrix_local[new_ids]
    return Asset(
        cls=asset.cls,
        id=asset.id,
        vertices=asset.vertices,
        vertex_normals=asset.vertex_normals,
        faces=asset.faces,
        face_normals=asset.face_normals,
        joints=new_joints,
        skin=new_skin,
        parents=new_parents,
        names=new_names,
        matrix_local=new_matrix_local,
    )