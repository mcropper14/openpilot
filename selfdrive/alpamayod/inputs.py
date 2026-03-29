def build_alpamayo_observation(sm):
    v_ego = 0.0
    if sm.alive["carState"]:
        v_ego = float(sm["carState"].vEgo)

    calib_ready = bool(sm.alive["liveCalibration"])

    obs = {
        "v_ego": v_ego,
        "calib_ready": calib_ready,
        "timestamp": None,
        "images": None,
    }
    return obs