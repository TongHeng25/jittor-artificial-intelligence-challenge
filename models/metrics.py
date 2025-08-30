import jittor as jt

def J2J(
    joints_a: jt.Var,
    joints_b: jt.Var,
) -> jt.Var:
    '''
    calculate J2J loss in [-1, 1]^3 cube
    
    joints_a: (J1, 3) joint

    joints_b: (J2, 3) joint
    '''
    assert isinstance(joints_a, jt.Var)
    assert isinstance(joints_b, jt.Var)
    assert joints_a.ndim == 2, "joints_a should be shape (J1, 3)"
    assert joints_b.ndim == 2, "joints_b should be shape (J2, 3)"
    dis1 = ((joints_a.unsqueeze(0) - joints_b.unsqueeze(1))**2).sum(dim=-1).sqrt()
    loss1 = dis1.min(dim=-1)
    dis2 = ((joints_b.unsqueeze(0) - joints_a.unsqueeze(1))**2).sum(dim=-1).sqrt()
    loss2 = dis2.min(dim=-1)
    return (loss1.mean() + loss2.mean()) / 2 / 2