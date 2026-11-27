"""
GR1T2 双臂交接分析：用 URDF 做 FK，统计示教里右手保持位置 / 左手闭合位置的分布，并和测试 action 对比。
用法：
  python handover_fk_analysis.py --urdf gr1t2_fourier_hand_6dof.urdf \
      --demos "data/chunk-000/episode_*.parquet" --tests 1519.parquet 1520.parquet
URDF: https://github.com/FFTAI/Wiki-GRx-Models  (GRX/GR1/gr1t2/urdf/gr1t2_fourier_hand_6dof.urdf)
示教: 26 维 [L臂7, R臂7, L手6, R手6]；测试 dump: 36 维 Isaac Lab 顺序（前14维左右交错）。
位置均为 torso 坐标系下 hand_base，单位 cm。

python assets/handover_fk_analysis.py --urdf assets/gr1t2_fourier_hand_6dof.urdf --demos "datasets/gr1_two_arm_can_sort_random_recorded_300_replay/lerobot/data/chunk-000/episode_*.parquet" --test datasets/1519.parquet datasets/1520.parquet
"""
import argparse, glob, numpy as np, pandas as pd, xml.etree.ElementTree as ET

def load_fk(urdf):
    J = {j.get('name'): j for j in ET.parse(urdf).getroot().findall('joint')}
    def rpy(a):
        x, y, z = a; cx, sx, cy, sy, cz, sz = np.cos(x), np.sin(x), np.cos(y), np.sin(y), np.cos(z), np.sin(z)
        return np.array([[cz*cy, cz*sy*sx-sz*cx, cz*sy*cx+sz*sx], [sz*cy, sz*sy*sx+cz*cx, sz*sy*cx-cz*sx], [-sy, cy*sx, cy*cx]])
    def tf(j):
        o = j.find('origin'); T = np.eye(4)
        T[:3, :3] = rpy([float(v) for v in o.get('rpy', '0 0 0').split()]); T[:3, 3] = [float(v) for v in o.get('xyz', '0 0 0').split()]; return T
    def rot(ax, q):
        a = np.array(ax) / np.linalg.norm(ax); K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
        T = np.eye(4); T[:3, :3] = np.eye(3) + np.sin(q)*K + (1-np.cos(q))*K@K; return T
    names = lambda s: [f'{s}_{k}_joint' for k in ['shoulder_pitch', 'shoulder_roll', 'shoulder_yaw', 'elbow_pitch', 'wrist_yaw', 'wrist_roll', 'wrist_pitch']]
    ch = {s: [(tf(J[n]), [float(v) for v in J[n].find('axis').get('xyz').split()]) for n in names(s)] for s in ('left', 'right')}
    hb = {'left': tf(J['L_hand_base_joint']), 'right': tf(J['R_hand_base_joint'])}
    def fk(side, q):
        T = np.eye(4)
        for (To, ax), qi in zip(ch[side], q): T = T @ To @ rot(ax, qi)
        return (T @ hb[side])[:3, 3] * 100
    return fk

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--urdf', required=True); ap.add_argument('--demos'); ap.add_argument('--tests', nargs='*', default=[])
    a = ap.parse_args(); fk = load_fk(a.urdf)
    P = lambda L, R: (np.array([fk('left', q) for q in L]), np.array([fk('right', q) for q in R]))
    model = None
    if a.demos:
        RH, LC, T = [], [], []
        for f in sorted(glob.glob(a.demos, recursive=True)):
            A = np.stack(pd.read_parquet(f)['action']); lh = A[:, 14:19].mean(1) < -0.7; rh = A[:, 20:25].mean(1) < -0.7
            if not lh.any() or not rh.any(): continue
            tl = np.where(lh)[0][0]; tr = np.where(rh)[0][0]
            if tl - 60 < tr: continue
            pl, pr = P(A[:, 0:7], A[:, 7:14]); RH.append(pr[tl-60:tl-10].mean(0)); LC.append(pl[tl]); T.append((tr, tl))
        RH, LC, T = np.array(RH), np.array(LC), np.array(T)
        print(f'示教 {len(RH)} 条 | 右手闭合步 {T[:,0].mean():.0f}±{T[:,0].std():.0f}，左手闭合步 {T[:,1].mean():.0f}±{T[:,1].std():.0f}')
        print('右手保持位置 xyz 均值', RH.mean(0).round(1), 'std(cm)', RH.std(0).round(1))
        print('左手闭合位置 xyz 均值', LC.mean(0).round(1), 'std(cm)', LC.std(0).round(1))
        d = LC - RH; print('左相对右 xyz 均值', d.mean(0).round(1), 'std', d.std(0).round(1))
        X = np.c_[RH, np.ones(len(RH))]; coef, *_ = np.linalg.lstsq(X, LC, rcond=None); res = LC - X@coef
        print('用右手保持位置线性预测左手闭合位置：R² =', (1 - res.var(0)/np.maximum(LC.var(0), 1e-9)).round(2), '（接近0=左手目标基本不随右手变化）')
        model = (RH, LC, coef, d.mean(0), d.std(0))
    for f in a.tests:
        A = np.stack(pd.read_parquet(f)['action']); L = A[:, [0, 2, 4, 6, 8, 10, 12]]; R = A[:, [1, 3, 5, 7, 9, 11, 13]]
        pl, pr = P(L, R); lh = A[:, 14:19].mean(1) < -0.7
        if lh.any():
            tl = np.where(lh)[0][0]; w = slice(max(tl-30, 0), max(tl-5, 1)); lpos = pl[tl]; tag = f'左手闭合步 {tl}'
        else:
            w = slice(len(A)-100, len(A)); lpos = pl[w].mean(0); tag = '左手未闭合，取末100步'
        rpos = pr[w].mean(0)
        print(f'\n{f} ({tag}): 右手 {rpos.round(1)} 左手 {lpos.round(1)} 两手距离 {np.linalg.norm(lpos-rpos):.1f} cm')
        if model is None: continue
        RH, LC, coef, dm, ds = model
        z = (rpos - RH.mean(0)) / np.maximum(RH.std(0), 1e-9); dist = np.linalg.norm(RH - rpos, axis=1)
        print('  右手位置相对示教分布的 z 分数 xyz', z.round(1), '| 距最近示教右手位置', dist.min().round(1), 'cm，5cm内示教条数', int((dist < 5).sum()))
        cond = np.r_[rpos, 1] @ coef; uncond = LC.mean(0)
        print('  左手条件目标(按右手位置回归)', cond.round(1), '| 无条件均值', uncond.round(1))
        print('  左手实际 离 条件目标', np.linalg.norm(lpos-cond).round(1), 'cm | 离 无条件均值', np.linalg.norm(lpos-uncond).round(1), 'cm (后者更小=左手在输出平均位置，没跟右手)')
        print('  左相对右 实际', (lpos-rpos).round(1), '| 示教', dm.round(1), '±', ds.round(1))

if __name__ == '__main__':
    main()
