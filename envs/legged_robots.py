import gym.spaces
import numpy as np
import gym


class LeggedRobot():

    def __init__(
        self,
        task: str,
        discount: int,
        robot_type: str,
        repeat: int = 1,
        enable_rendering: bool = True
    ):
        assert robot_type in ("A1", "Go1","Aliengo"), "Incorrect robot type"
        assert task in ("sim", "real"), task
        self._discount = discount

        # don't move the import from this place! It works only like this
        from motion_imitation.envs import env_builder

        self._env = env_builder.build_env(
            enable_rendering=enable_rendering,
            num_action_repeat=repeat,
            use_real_robot=bool(task == "real"),
            robot_type = robot_type
        )
        
    @property
    def observation_space(self):
        spaces: dict[str, gym.spaces.Box] = {}
        for key, value in self._env.observation_space.items():
            spaces[key] = value
        spaces["image"] = gym.spaces.Box(0, 255, (64, 64, 3), dtype=np.uint8)
        spaces["is_first"] = gym.spaces.Box(0, 1, (), dtype=bool)
        spaces["is_last"] = gym.spaces.Box(0, 1, (), dtype=bool)
        spaces["is_terminal"] = gym.spaces.Box(0, 1, (), dtype=bool)
        return gym.spaces.Dict(spaces)

    @property
    def action_space(self):
        return gym.spaces.Box(-1.0, 1.0, shape=(12,), dtype=np.float32)
    
    def reset(self):
        obs = self._env.reset()
        obs["image"] = self._env.render("rgb_array")
        obs["is_first"] = True
        obs["is_last"] = False
        obs["is_terminal"] = False
        return obs
    
    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        obs["image"] = self._env.render("rgb_array")
        obs["is_first"] = False
        obs["is_last"] = done
        obs["is_terminal"] = False
        info["discount"] = np.array(self._discount).astype(np.float32)
        assert obs["image"].shape == (64, 64, 3), obs["image"].shape
        assert obs["image"].dtype == np.uint8, obs["image"].dtype
        return obs, reward, done, info
    
    def close(self):
        print("Closing env")
        self._env.close()
