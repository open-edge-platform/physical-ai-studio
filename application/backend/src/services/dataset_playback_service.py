import time

from lerobot.utils.robot_utils import precise_sleep

from robots.widowxai.trossen_widowx_ai_follower import TrossenWidowXAIFollower
from schemas import Episode, NetworkIpRobotConfig


class DatasetPlaybackService:

    @staticmethod
    async def playback_episode(episode: Episode):
        config = NetworkIpRobotConfig(type="follower",
                                      robot_type="Trossen_WidowXAI_Follower",
                                      connection_string="192.168.1.3")
        robot = TrossenWidowXAIFollower(config)
        await robot.connect()

        actions = episode.actions
        # joint_names = [k for k in episode.action_keys if k.endswith(".pos")]
        joint_names = episode.action_keys

        for action in actions:
            start_loop_t = time.perf_counter()
            action_dict = {joint_names[i]: action[i] for i in range(len(joint_names))}

            await robot.set_joints_state(action_dict)

            dt_s = time.perf_counter() - start_loop_t
            wait_time = 1 / 30 - dt_s
            precise_sleep(wait_time)
        await robot.disconnect()