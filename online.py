from numpy import sin, cos, arctan2, array
from numpy.linalg import norm
from env import Racing, Config
from env.vehicle_model import VehicleModel
from time import sleep
from mpc import MPC

HORIZON_LENGTH = 0.5

def policy(track, state):

    theta0 = track.get_theta(*state[:2])
    thetaN = theta0 + HORIZON_LENGTH
    XY = track.center_spline([theta0, thetaN])
    dpsi = arctan2(*track.center_spline(thetaN, 1)[::-1]) - state[2]

    cpsi0 = cos(state[2])
    spsi0 = sin(state[2])
    p3 = array([[cpsi0, spsi0], [-spsi0, cpsi0]]).dot(XY[-1, :] - XY[0, :]) 
    p1 = array([HORIZON_LENGTH/3, 0])
    p2 = p3 - HORIZON_LENGTH/3 * array([cos(dpsi), sin(dpsi)])
    phi12 = arctan2(*(p2 - p1)[::-1])
    phi23 = dpsi - phi12

    return array([
        HORIZON_LENGTH/3,
        norm(p2 - p1),
        HORIZON_LENGTH/3,
        phi12,
        phi23,
        0.75
    ])

if __name__ == "__main__":

    config = Config()

    vehicle_model = VehicleModel(**config.vehicle.get_dict())

    controller = MPC(vehicle_model, config)

    env = Racing(controller, config, render_mode="human")

    state = env.reset()

    print("Game starts in 5 seconds..")
    #sleep(5)

    #env.start()
    
    for i in range(4000):
        #u1 = controller_theta.send([i / 100, env.e, 0])
        #u2 = controller_throttle.send([i / 100, env.vx, 0.5])

        action = policy(env.track, state)
        state, reward, done, info = env.step(action)

        if done == True:
            break

