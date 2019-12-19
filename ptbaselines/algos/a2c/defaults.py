def mujoco():
    return dict(
        nsteps=2048,
        lam=0.95,
        gamma=0.99,
        noptepochs=10,
        log_interval=1,
        ent_coef=0.0,
        value_network='copy'
    )

def atari():
    return dict(
        nsteps=128, 
        lam=0.95, gamma=0.99, noptepochs=4, log_interval=1,
        ent_coef=.01,
    )

def retro():
    return atari()
